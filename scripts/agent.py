from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from sklearn.metrics.pairwise import cosine_similarity
from pymilvus import MilvusClient
from dotenv import load_dotenv
from tavily import TavilyClient
from typing import Generator
import datetime
import psycopg2.extras
import psycopg2.pool
import numpy as np
import psycopg2
import json
import os

from configs.config import (
    LLM_MODEL, LLM_FAST_MODEL, EMBEDDING_MODEL,
    MILVUS_COLLECTION, PROFESSOR_COLLECTION, MILVUS_TOP_K,
    MILVUS_URI,
)

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY", "")

_tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "")) if os.getenv("TAVILY_API_KEY") else None

milvus_client = MilvusClient(uri=MILVUS_URI)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "scholarscout"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

_db_pool = psycopg2.pool.SimpleConnectionPool(1, 5, **DB_CONFIG)

llm        = ChatOpenAI(model=LLM_MODEL, temperature=0)
llm_fast   = ChatOpenAI(model=LLM_FAST_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

_SCRIPTS_DIR     = os.path.dirname(os.path.abspath(__file__))
_CATEGORIES_PATH = os.path.join(_SCRIPTS_DIR, "..", "data", "field_categories.json")
ANCHOR_CACHE_PATH = os.path.join(_SCRIPTS_DIR, "..", "data", "anchor_vectors.npy")
with open(_CATEGORIES_PATH) as _f:
    FIELD_CATEGORIES: list[str] = json.load(_f)

_CATEGORIES_LIST = "\n".join(f"  - {c}" for c in FIELD_CATEGORIES)

DB_SCHEMA = f"""
Table: universities
Columns:
  - id                INT     (primary key)
  - program_name      TEXT    (name of the scholarship/master program)
  - university_name   TEXT    (name of the university or consortium)
  - field_category    TEXT    (must be one of the allowed values listed below)
  - deadline_ir       DATE    (application deadline for international students)
  - deadline_non_ir   DATE    (application deadline for non-international students)
  - funding_type      TEXT    (e.g. "erasmus", "germany")

Allowed values for field_category:
{_CATEGORIES_LIST}

Table: finances  (JOIN to universities via finances.university_id = universities.id)
Columns:
  - university_id               INT     (foreign key → universities.id)
  - tuition_fee_eu_per_year     INT     (annual tuition for EU students, in EUR; NULL if unknown)
  - tuition_fee_non_eu_per_year INT     (annual tuition for non-EU students, in EUR; NULL if unknown)
  - scholarship_available       BOOLEAN (true if any scholarship is offered)
  - funding_category            JSONB   (array of funding types, e.g. ["fully_funded", "tuition_free", "self_funded"])
  - urls                        JSONB   (array of URLs for fees/scholarship pages)

JSONB funding_category values:
  - "fully_funded"   – full Erasmus Mundus scholarship (covers tuition + living allowance)
  - "tuition_free"   – tuition waived but no living allowance
  - "self_funded"    – student pays tuition themselves

JSONB query examples for funding_category:
  - funding_category @> '["fully_funded"]'::jsonb   → has fully-funded option
  - funding_category @> '["self_funded"]'::jsonb     → has self-funded option
"""

_ROUTE_ANCHORS = {
    "db": (
        "list programs, opportunities, show all scholarships, filter programs, find opportunities, "
        "programs with deadline, upcoming deadlines, passed deadlines, deadline before date, "
        "tuition fee, cost per year, how much does it cost, affordable programs, free tuition, "
        "programs in the field of, which universities offer, show me programs where, "
        "scholarship available, budget under, programs sorted by, programs by country"
    ),
    "semantic": (
        "what is this program about, program description, curriculum overview, coursework details, "
        "programs focused on, programs related to topic, programs about a subject area, "
        "digital transformation, sustainability, interdisciplinary, research area, specialization in, "
        "admission requirements, eligibility criteria, who can apply, language requirements, "
        "required documents, application process, prerequisites"
    ),
    "chat": (
        "general advice, greetings, visa process, how do I write, essay writing tips, "
        "explain the difference between, what does fully funded mean, definition of scholarship, "
        "what is the difference, compare two concepts, general knowledge about scholarships, "
        "understanding terms, what does X mean, how things work"
    ),
    "professor": (
        "professor research, PhD position, doctoral position, postdoc, "
        "research assistant, research internship, academic job, open position in lab, "
        "supervised research, professor looking for student, research group opening, "
        "research fellowship, join a lab, university research vacancy"
    ),
}

def _load_anchor_vectors() -> dict:
    if os.path.exists(ANCHOR_CACHE_PATH):
        data = np.load(ANCHOR_CACHE_PATH, allow_pickle=True).item()
        if set(data.keys()) == set(_ROUTE_ANCHORS.keys()):
            return data
    vectors = {
        route: np.array(embeddings.embed_query(desc)).reshape(1, -1)
        for route, desc in _ROUTE_ANCHORS.items()
    }
    np.save(ANCHOR_CACHE_PATH, vectors)
    return vectors

_anchor_vectors = _load_anchor_vectors()


class AgentState(TypedDict):
    query:             str
    query_vec:         list[float]  # cached embedding reused by vector_search_node
    messages:          list[BaseMessage]
    needs_db:          bool
    needs_vector:      bool
    needs_professor:   bool
    sql:               str
    db_results:        list[dict]
    vector_results:    list[dict]
    professor_results: list[dict]
    web_results:       list[dict]
    response:          str
    validation_passed: bool
    retry_count:       int
    user_profile:      dict        # {degree_level, field, specialization, current_education}
    awaiting_profile:  bool        # True when the agent paused to ask for profile info
    pending_query:     str         # original search query saved while waiting for profile info


_PROFILE_EXTRACT_SYSTEM = """Extract academic profile information from the user's message.
Return a JSON object with these exact keys (use null for anything not mentioned):
{
  "degree_level": "BS" | "MS" | "PhD" | null,
  "field": "<field of study, e.g. Computer Science, Mechanical Engineering, Finance>" | null,
  "specialization": "<specific area within the field, e.g. Machine Learning, Robotics>" | null,
  "current_education": "<current degree and/or institution, e.g. BSc CS at NUST>" | null
}
Return ONLY the JSON object — no explanation, no markdown fences."""


def _extract_profile(text: str) -> dict:
    """Parse profile fields from a single message. Returns only non-null fields."""
    try:
        raw = llm_fast.invoke([
            SystemMessage(content=_PROFILE_EXTRACT_SYSTEM),
            HumanMessage(content=text),
        ]).content.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        return {k: v for k, v in data.items() if v is not None}
    except Exception:
        return {}


def profile_extractor_node(state: AgentState) -> AgentState:
    """Passively merge any profile info found in the current message into user_profile."""
    extracted = _extract_profile(state["query"])
    merged    = {**state.get("user_profile", {}), **extracted}
    return {**state, "user_profile": merged}


def query_restorer_node(state: AgentState) -> AgentState:
    """If the previous turn was waiting for profile info and profile is now complete,
    restore the original search query so routing and retrieval use the right intent."""
    pending = state.get("pending_query", "")
    if pending and _profile_complete(state.get("user_profile", {})):
        return {**state, "query": pending, "pending_query": "", "query_vec": []}
    return state


def router_node(state: AgentState) -> AgentState:
    vec = embeddings.embed_query(state["query"])
    query_vec = np.array(vec).reshape(1, -1)
    scores = {
        route: cosine_similarity(query_vec, anchor)[0][0]
        for route, anchor in _anchor_vectors.items()
    }
    chat_score      = scores["chat"]
    needs_db        = bool(scores["db"]        > chat_score)
    needs_vector    = bool(scores["semantic"]  > chat_score)
    needs_professor = bool(scores["professor"] > chat_score and scores["professor"] >= scores["db"] and scores["professor"] >= scores["semantic"])
    # Professor route is exclusive — if professor wins, suppress the others
    if needs_professor:
        needs_db     = False
        needs_vector = False
    return {**state, "query_vec": vec, "needs_db": needs_db, "needs_vector": needs_vector, "needs_professor": needs_professor}


def route_decision(state: AgentState) -> str:
    if state["needs_db"] and state["needs_vector"]:
        return "hybrid"
    if state["needs_db"]:
        return "query_db"
    if state["needs_vector"]:
        return "vector_search"
    return "respond"


def _profile_complete(profile: dict) -> bool:
    return bool(profile.get("degree_level") and profile.get("field"))


def _build_profile_question(profile: dict) -> str:
    missing = []
    if not profile.get("degree_level"):
        missing.append("your degree level (BS, MS, or PhD)")
    if not profile.get("field"):
        missing.append("your field of study (e.g. Computer Science, Engineering, Business)")
    ask = f"{missing[0]} and {missing[1]}" if len(missing) == 2 else missing[0]
    return (
        f"I'd love to find the most relevant programs for you! "
        f"Could you quickly share {ask}?"
    )


def _enrich_query(query: str, profile: dict) -> str:
    """Append profile context to the raw query so SQL generation and vector search are more targeted."""
    parts = [query]
    if profile.get("degree_level"):
        parts.append(f"for a {profile['degree_level']} student")
    if profile.get("field"):
        parts.append(f"in {profile['field']}")
    if profile.get("specialization"):
        parts.append(f"specializing in {profile['specialization']}")
    return " ".join(parts)


def profile_gate_node(state: AgentState) -> AgentState:
    """Block search queries until degree_level and field are known; enrich query when they are."""
    is_search = state["needs_db"] or state["needs_vector"] or state.get("needs_professor", False)
    profile   = state.get("user_profile", {})

    if is_search and not _profile_complete(profile):
        question = _build_profile_question(profile)
        updated_messages = state.get("messages", []) + [AIMessage(content=question)]
        return {
            **state,
            "awaiting_profile": True,
            "pending_query":    state["query"],
            "response":         question,
            "messages":         updated_messages,
        }

    if is_search and profile:
        enriched = _enrich_query(state["query"], profile)
        # Clear query_vec so vector_search_node re-embeds the enriched query
        return {**state, "awaiting_profile": False, "query": enriched, "query_vec": []}

    return {**state, "awaiting_profile": False}


def profile_gate_decision(state: AgentState) -> str:
    if state.get("awaiting_profile"):
        return "end"
    if state.get("needs_professor"):
        return "professor_search"
    if state["needs_db"] and state["needs_vector"]:
        return "hybrid"
    if state["needs_db"]:
        return "query_db"
    if state["needs_vector"]:
        return "vector_search"
    return "respond"


def query_db_node(state: AgentState) -> AgentState:
    system = SystemMessage(content=f"""You are a SQL expert. Given the schema below, write a
    single PostgreSQL SELECT query that answers the user's question.
    {DB_SCHEMA}
    Rules:
    - Only use SELECT statements, no INSERT/UPDATE/DELETE.
    - When the query involves fees, budget, cost, or scholarships: JOIN universities with finances on finances.university_id = universities.id and select all columns from both tables (u.*, f.*).
    - When the query is only about programs/deadlines with no financial filter: query universities alone.
    - For field_category filters, always use exact equality (= or IN) with the allowed category values.
    - Use ILIKE for text matching on program_name and university_name.
    - For budget filters (e.g. "under €5000"): filter on tuition_fee_eu_per_year or tuition_fee_non_eu_per_year with <= and handle NULLs with IS NULL OR <=.
    - For fully-funded queries: filter with funding_category @> '["fully_funded"]'::jsonb
    - Always include a LIMIT 20 unless the user asks for all results.
    - Return ONLY the raw SQL query, no explanation, no markdown fences.
    """)
    human      = HumanMessage(content=state["query"])
    sql_result = llm_fast.invoke([system, human])
    sql        = sql_result.content.strip().strip("```sql").strip("```").strip()

    rows = []
    conn = None
    try:
        conn = _db_pool.getconn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
    except Exception as e:
        rows = [{"error": str(e)}]
    finally:
        if conn:
            _db_pool.putconn(conn)

    return {**state, "sql": sql, "db_results": rows}


def vector_search_node(state: AgentState) -> AgentState:
    query_vec = state.get("query_vec") or embeddings.embed_query(state["query"])
    hits = milvus_client.search(
        collection_name=MILVUS_COLLECTION,
        data=[query_vec],
        limit=MILVUS_TOP_K,
        output_fields=["university_name", "url", "description"],
    )
    results = [
        {
            "university_name": hit["entity"].get("university_name", ""),
            "url":             hit["entity"].get("url", ""),
            "description":     hit["entity"].get("description", ""),
            "score":           hit["distance"],
        }
        for hit in hits[0]
    ]
    return {**state, "vector_results": results}


def hybrid_node(state: AgentState) -> AgentState:
    state = query_db_node(state)
    state = vector_search_node(state)
    return state


def professor_search_node(state: AgentState) -> AgentState:
    query_vec = state.get("query_vec") or embeddings.embed_query(state["query"])
    hits = milvus_client.search(
        collection_name=PROFESSOR_COLLECTION,
        data=[query_vec],
        limit=MILVUS_TOP_K,
        output_fields=["professor_name", "university", "position_type", "title",
                       "country", "city", "deadline", "email", "url", "funding", "description"],
    )
    results = [
        {
            "professor_name": hit["entity"].get("professor_name", ""),
            "university":     hit["entity"].get("university", ""),
            "position_type":  hit["entity"].get("position_type", ""),
            "title":          hit["entity"].get("title", ""),
            "country":        hit["entity"].get("country", ""),
            "city":           hit["entity"].get("city", ""),
            "deadline":       hit["entity"].get("deadline", ""),
            "email":          hit["entity"].get("email", ""),
            "url":            hit["entity"].get("url", ""),
            "funding":        hit["entity"].get("funding", ""),
            "description":    hit["entity"].get("description", ""),
            "score":          hit["distance"],
        }
        for hit in hits[0]
    ]
    return {**state, "professor_results": results}


def tavily_search_node(state: AgentState) -> AgentState:
    """Tavily web search fallback — used when DB and vector store return nothing."""
    web_results = []
    if _tavily:
        try:
            resp = _tavily.search(
                query=state["query"],
                max_results=5,
                search_depth="basic",
                include_answer=True,
            )
            if resp.get("answer"):
                web_results.append({"title": "Quick Summary", "url": "", "content": resp["answer"]})
            for r in resp.get("results", []):
                web_results.append({"title": r["title"], "url": r["url"], "content": r["content"]})
        except Exception:
            pass

    return {**state, "web_results": web_results, "retry_count": state.get("retry_count", 0) + 1}


def _build_context(state: AgentState) -> str:
    """Assemble all available data sources into a labelled context string."""
    parts = []

    if state.get("db_results"):
        parts.append(
            "=== STRUCTURED DATA (deadlines, fees, funding) ===\n"
            + json.dumps(state["db_results"], indent=2, default=str)
        )
    if state.get("vector_results"):
        parts.append(
            "=== PROGRAM DESCRIPTIONS (admission, eligibility, curriculum) ===\n"
            + json.dumps(state["vector_results"], indent=2, default=str)
        )
    if state.get("professor_results"):
        parts.append(
            "=== PROFESSOR RESEARCH OPPORTUNITIES ===\n"
            + json.dumps(state["professor_results"], indent=2, default=str)
        )
    if state.get("web_results"):
        web_text = "\n\n".join(
            f"[Source: {r['title']}]\nURL: {r['url']}\n{r['content']}"
            for r in state["web_results"]
        )
        parts.append("=== WEB SEARCH RESULTS (Tavily) ===\n" + web_text)

    return "\n\n".join(parts)


def _format_profile_ctx(profile: dict) -> str:
    if not profile:
        return ""
    parts = []
    if profile.get("degree_level"):
        parts.append(f"degree level: {profile['degree_level']}")
    if profile.get("field"):
        parts.append(f"field: {profile['field']}")
    if profile.get("specialization"):
        parts.append(f"specialization: {profile['specialization']}")
    if profile.get("current_education"):
        parts.append(f"current education: {profile['current_education']}")
    return "User profile — " + ", ".join(parts) + ".\n"


def _build_system_message(state: AgentState, todays_date: str) -> SystemMessage:
    has_db        = bool(state.get("db_results"))
    has_vector    = bool(state.get("vector_results"))
    has_professor = bool(state.get("professor_results"))
    has_web       = bool(state.get("web_results"))
    has_any       = has_db or has_vector or has_professor or has_web
    profile_ctx = _format_profile_ctx(state.get("user_profile") or {})

    if not has_any:
        return SystemMessage(content=f"""You are a helpful scholarship assistant.
        {profile_ctx}Answer the user's question about scholarships, programs, and academic opportunities.
        If you don't have enough information, say so clearly.""")

    context = _build_context(state)

    # Choose format based on data mix
    if has_professor:
        program_format = """
        Use this EXACT format for every research opportunity — no deviations:

        ### [Position Title]
        - **Type**: [PhD / Postdoc / Research Assistant / Master Thesis]
        - **Supervisor**: [professor name]
        - **University**: [university name], [city], [country]
        - **Research Area**: [field / topics]
        - **Funding**: [funding details or "Not specified"]
        - **Deadline**: [date or "Open / Rolling"]
        - **Contact**: [email — only if present, otherwise omit]
        - **More info**: [url — only if present, otherwise omit]
        - **About**: [2-3 sentence summary of the research and what the student would work on]

        Separate opportunities with a line containing only ---

        Rules:
        - List opportunities with upcoming or open deadlines first; past-deadline ones last.
        - Do not fabricate contact details or URLs.
        - Omit any bullet line where data is genuinely unavailable.
        - Tailor the emphasis to the student's field and level from their profile."""
    elif has_db:
        program_format = """
        Use this EXACT format for every program — no deviations:

        ### [Program Name]
        - **University**: [university name]
        - **Field**: [field category]
        - **Deadline**: [date] ([upcoming] or [passed])
        - **Funding**: [funding type]
        - **Tuition (EU)**: [€X/year] or N/A
        - **Tuition (Non-EU)**: [€X/year] or N/A
        - **About**: [brief description — only if available from descriptions or web data]
        - **Admission**: [key requirements — only if available]
        - **More info**: [url — only if present, otherwise omit]

        Separate programs with a line containing only ---

        Rules:
        - List programs with upcoming deadlines first; passed-deadline programs last.
        - If all deadlines passed, open with: "All deadlines for this query have passed — here are the options for future reference."
        - If tuition is NULL, write N/A.
        - Omit any bullet line where data is genuinely unavailable.
        - Do not fabricate details. If web data contradicts structured data, prefer structured data."""
    else:
        program_format = """
        Use this format for each result:

        ### [University / Program Name]
        - **About**: [2-3 sentence summary]
        - **Admission**: [key eligibility and requirements]
        - **More info**: [url — only if available]

        Separate results with a line containing only ---

        Rules:
        - Only use information present in the data above.
        - Do not fabricate details."""

    return SystemMessage(content=f"""You are a helpful scholarship assistant who helps in identifying relevant academic
    opportunities.
    Today is {todays_date}.
        {profile_ctx}
        The following data was retrieved to answer the user's question:

        {context}

        {program_format}

        - You cover international scholarship and academic programs (Erasmus Mundus, German universities, and others) as well as professor-supervised research opportunities. For completely unrelated queries give brief general advice only.
        - Tailor your answer to the user's profile when available (degree level, field, specialization).
        - If no relevant programs are found in any source, say: "Sorry, I cannot find any relevant opportunity for your query. Try adding more information"
        """)


def respond_node(state: AgentState) -> AgentState:
    todays_date = datetime.datetime.now().strftime("%B %d, %Y")
    system      = _build_system_message(state, todays_date)
    history     = state.get("messages", [])
    has_data = bool(state.get("db_results") or state.get("vector_results") or state.get("professor_results") or state.get("web_results"))
    model    = llm if has_data else llm_fast
    result   = model.invoke([system] + history)
    updated_messages = history + [AIMessage(content=result.content)]
    return {**state, "response": result.content, "messages": updated_messages}


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("profile_extractor", profile_extractor_node)
    graph.add_node("query_restorer",    query_restorer_node)
    graph.add_node("router",            router_node)
    graph.add_node("profile_gate",      profile_gate_node)
    graph.add_node("query_db",          query_db_node)
    graph.add_node("vector_search",     vector_search_node)
    graph.add_node("hybrid",            hybrid_node)
    graph.add_node("professor_search",  professor_search_node)
    graph.add_node("respond",           respond_node)

    graph.set_entry_point("profile_extractor")

    graph.add_edge("profile_extractor", "query_restorer")
    graph.add_edge("query_restorer",    "router")
    graph.add_edge("router",            "profile_gate")

    graph.add_conditional_edges("profile_gate", profile_gate_decision, {
        "end":             END,
        "query_db":        "query_db",
        "vector_search":   "vector_search",
        "hybrid":          "hybrid",
        "professor_search":"professor_search",
        "respond":         "respond",
    })

    graph.add_edge("query_db",          "respond")
    graph.add_edge("vector_search",     "respond")
    graph.add_edge("hybrid",            "respond")
    graph.add_edge("professor_search",  "respond")

    graph.add_edge("respond", END)

    return graph.compile(checkpointer=MemorySaver())


agent = build_agent()


def ask_stream(query: str, thread_id: str = "1", web_search: bool = False) -> Generator[str, None, None]:
    """Streams the final LLM response. Runs Tavily only when the user explicitly enables web search."""
    config       = {"configurable": {"thread_id": thread_id}}
    prior        = agent.get_state(config)
    prior_values = prior.values if prior.values else {}
    prior_messages: list[BaseMessage] = prior_values.get("messages") or []
    prior_profile: dict               = prior_values.get("user_profile") or {}
    prior_pending: str                = prior_values.get("pending_query") or ""
    new_messages = prior_messages + [HumanMessage(content=query)]

    state: AgentState = {
        "query": query, "query_vec": [], "messages": new_messages,
        "needs_db": False, "needs_vector": False, "needs_professor": False,
        "sql": "", "db_results": [], "vector_results": [], "professor_results": [],
        "web_results": [], "response": "",
        "validation_passed": False, "retry_count": 0,
        "user_profile": prior_profile, "awaiting_profile": False,
        "pending_query": prior_pending,
    }

    # Step 1: extract profile, restore original search query if one was pending,
    # then route on the correct query
    state = profile_extractor_node(state)
    state = query_restorer_node(state)
    state = router_node(state)
    state = profile_gate_node(state)

    # Step 2: profile incomplete — stream the question and save state, then stop
    if state.get("awaiting_profile"):
        yield state["response"]
        agent.update_state(config, state)
        return

    # Step 3: fetch data (query already enriched with profile by profile_gate_node)
    if state.get("needs_professor"):
        state = professor_search_node(state)
    elif state["needs_db"] and state["needs_vector"]:
        state = hybrid_node(state)
    elif state["needs_db"]:
        state = query_db_node(state)
    elif state["needs_vector"]:
        state = vector_search_node(state)

    # Step 4: run Tavily only when the user toggled web search on
    if web_search:
        yield "\x00STATUS\x00Searching the web for live results…"
        state = tavily_search_node(state)

    # Step 5: stream the final response
    todays_date = datetime.datetime.now().strftime("%B %d, %Y")
    system  = _build_system_message(state, todays_date)
    history = state["messages"]

    has_data = bool(state.get("db_results") or state.get("vector_results") or state.get("professor_results") or state.get("web_results"))
    model    = llm if has_data else llm_fast

    full_response = ""
    for chunk in model.stream([system] + history):
        token = chunk.content
        if token:
            full_response += token
            yield token

    updated_messages = history + [AIMessage(content=full_response)]
    agent.update_state(
        config,
        {**state, "response": full_response, "messages": updated_messages},
    )
