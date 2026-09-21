import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from tools.flight_tool import search_flights

from mcp_client import (
    tavily_mcp_search,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM - original model kept
# =========================
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    # The integration prompts below reserve enough of Groq's 8k request budget
    # for a complete multi-day plan, rather than ending partway through a day.
    max_tokens=3000,
)

# =========================
# State - original fields kept, new control fields added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]

PLAN_COMPLETION_MARKER = "<!-- TRIPMATE_PLAN_COMPLETE -->"


def _prompt_context(value: Any, limit: int) -> str:
    """Keep a complete state value while sending a bounded excerpt to an LLM."""
    text = str(value)
    if len(text) <= limit:
        return text

    head_length = limit // 2
    tail_length = limit - head_length
    return (
        f"{text[:head_length]}\n"
        "[...additional source data omitted from this prompt...]\n"
        f"{text[-tail_length:]}"
    )


def _format_hotel_search_results(value: Any, max_results: int = 3) -> str:
    """Convert Tavily MCP's JSON content blocks into readable hotel evidence."""
    payloads: list[str] = []

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                payloads.append(item["text"])
    elif isinstance(value, dict) and isinstance(value.get("text"), str):
        payloads.append(value["text"])
    else:
        payloads.append(str(value))

    for payload in payloads:
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            continue

        results = data.get("results", []) if isinstance(data, dict) else []
        if not isinstance(results, list) or not results:
            continue

        lines = []
        for index, result in enumerate(results[:max_results], start=1):
            if not isinstance(result, dict):
                continue
            title = str(result.get("title", "Hotel result"))
            url = str(result.get("url", "")).strip()
            content = " ".join(
                str(result.get("content", "")).split()
            )
            excerpt = _prompt_context(content, 420)
            link = f"\n   Source: {url}" if url else ""
            lines.append(f"{index}. **{title}**\n   {excerpt}{link}")

        if lines:
            return "\n\n".join(lines)

    return _prompt_context(value, 1_800)


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _complete_plan_text(system_prompt: str, user_prompt: str) -> tuple[str, int]:
    """Generate a complete plan and continue once if the output was truncated."""
    completion_contract = f"""
{system_prompt}

Completion contract:
- Finish every requested day and every required section before stopping.
- Prefer concise wording over leaving a section, total, sentence, or table unfinished.
- Silently verify that the duration, budget total, and final recommendations are complete.
- End the response with this exact marker: {PLAN_COMPLETION_MARKER}
"""
    messages = [
        SystemMessage(content=completion_contract),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    text = str(response.content)
    calls = 1

    if PLAN_COMPLETION_MARKER not in text:
        continuation = llm.bind(max_tokens=1200).invoke(
            [
                *messages,
                AIMessage(content=text),
                HumanMessage(
                    content=(
                        "Continue exactly where the response stopped. Do not repeat "
                        "completed content. Finish every missing day and required "
                        "section, close any unfinished sentence or table, and end "
                        f"with {PLAN_COMPLETION_MARKER}."
                    )
                ),
            ]
        )
        text = f"{text.rstrip()}\n\n{str(continuation.content).lstrip()}"
        calls += 1

    return text.replace(PLAN_COMPLETION_MARKER, "").rstrip(), calls


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    query_for_llm = _prompt_context(query, 4_000)
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query_for_llm}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Selection guidance:
- Judge agents by whether their data materially improves the plan, not only by
  whether the user explicitly names that data type.
- Weather usually affects a multi-day, multi-city, outdoor, beach, or date-sensitive
  destination itinerary through timing, packing, comfort, and safety. In those
  cases, select weather_agent even when the user did not explicitly ask for weather.
- You may omit weather_agent when weather is genuinely irrelevant, such as a
  flight-status-only lookup.

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query_for_llm}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Select specialists based "
            "on practical value, not merely explicit keywords. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        reasoning = str(parsed.get("reasoning", "")).strip()

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent - direct AviationStack tool
# =========================
def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")
    query = state["user_query"]

    try:
        flight_data = search_flights(query)
    except Exception as exc:
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent - MCP Tavily tool
# =========================
def hotel_agent(state: TravelState):
    constraints = state.get("trip_constraints", {})
    destination = str(constraints.get("destination", "")).strip()
    budget = str(constraints.get("budget", "")).strip()
    style = str(constraints.get("travel_style", "")).strip()
    destination_text = destination or state["user_query"]
    query = (
        f"Recommend hotels in {destination_text}. Find named hotels with "
        "neighborhood, approximate nightly price, rating or notable amenity. "
        f"Prioritize {style or 'practical'} stays {('within ' + budget) if budget else 'for a sensible budget'}."
    )

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )
        hotel_results = _format_hotel_search_results(hotel_results)

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }


# =========================
# Weather Agent - original behavior kept
# =========================
def weather_agent(state: TravelState):
    city = extract_destination(
        state["user_query"]
    )

    try:
        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        weather_results = f"""
Live Weather MCP Result for {city}:

Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""

    except Exception as exc:
        print(
            f"WEATHER AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        weather_results = (
            f"Weather MCP data for {city} is temporarily unavailable. "
            "Any seasonal guidance must be labeled non-live, and the traveler "
            "should verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }


# =========================
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{_prompt_context(state['user_query'], 4_000)}

Trip Constraints:
{_prompt_context(state.get('trip_constraints', {}), 2_000)}

Flight Results:
{_prompt_context(state.get('flight_results', ''), 3_000)}

Hotel Results:
{_prompt_context(state.get('hotel_results', ''), 4_500)}

Weather Results:
{_prompt_context(state.get('weather_results', ''), 3_000)}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent - original behavior extended with selected results
# =========================
def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{_prompt_context(state['user_query'], 1_400)}

Trip Constraints:
{_prompt_context(state.get('trip_constraints', {}), 600)}

Flight Results:
{_prompt_context(state.get('flight_results', ''), 2_600)}

Hotel Results:
{_prompt_context(state.get('hotel_results', ''), 3_000)}

Weather Results:
{_prompt_context(state.get('weather_results', ''), 1_600)}

Budget Results:
{_prompt_context(state.get('budget_results', ''), 2_000)}

Completion rules:
- Determine the requested trip duration and include every day from Day 1 through
  the final requested day. Do not create a separate Day 0; arrival is part of Day 1.
- Complete all day sections before adding optional recommendations or notes.
- Use a short summary followed by `## Day 1` through `## Day N` headings.
- Give each day at most four concise, time-ordered bullet points. Do not use tables.
- Keep flights, hotel, weather, and budget notes brief so the complete itinerary
  fits in one response.

Create a practical, budget-aware draft ready for human review.
"""

    itinerary_text, plan_calls = _complete_plan_text(
        "You are an expert travel planner. Completion is mandatory: never stop "
        "mid-day, mid-section, mid-table, or before the requested trip ends.",
        prompt,
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": itinerary_text,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + plan_calls,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - original format kept, HITL feedback added
# =========================
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{_prompt_context(
    state.get('human_feedback', '') or 'Improve the draft before finalizing it.',
    4_000,
)}
"""

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{_prompt_context(state['user_query'], 1_400)}

Supervisor Constraints:
{_prompt_context(state.get('trip_constraints', {}), 600)}

Flights:
{_prompt_context(state.get('flight_results', ''), 2_600)}

Hotels:
{_prompt_context(state.get('hotel_results', ''), 3_000)}

Weather:
{_prompt_context(state.get('weather_results', ''), 1_600)}

Budget Analysis:
{_prompt_context(state.get('budget_results', ''), 1_800)}

Draft Itinerary:
{_prompt_context(state.get('itinerary', ''), 2_800)}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Treat the Weather section as live/current only when Weather contains a successful
  Weather MCP result. If weather data is absent or unavailable, clearly label any
  seasonal guidance as non-live instead of presenting model knowledge as live data.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
- In Flight Information, include the useful route, airline, schedule, and status
  details available in the flight results. Compare up to three returned flight
  options by airline, schedule, status, and a practical "best for" label.
  AviationStack does not provide fare data: never declare a flight cheaper unless
  a price is explicitly present in the source. Instead, clearly say that fares
  must be checked through a fare-search provider.
- In Hotel Suggestions, give exactly three recommended named hotels selected from
  the search evidence, each with neighborhood, approximate price only when the
  source provides one, and a one-line reason it suits the trip. Do not reproduce
  raw search results, source excerpts, or general travel articles.
- Preserve every completed day from the draft. If the draft has a day missing,
  add that day before any optional detail. Use compact headings and bullets, not tables.
- Keep weather, budget, and final-recommendation notes concise so the full
  day-by-day itinerary is never cut off.
"""

    final_response, plan_calls = _complete_plan_text(
        "You are a professional AI travel booking assistant. Completion is "
        "mandatory: never stop mid-day, mid-section, mid-table, mid-total, or "
        "before the final recommendations are finished.",
        final_prompt,
    )

    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
        "llm_calls": state.get("llm_calls", 0) + plan_calls,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = get_database_url()
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)
