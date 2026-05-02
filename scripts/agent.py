import os
import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
import datetime

load_dotenv()

# LangChain expects OPENAI_API_KEY; remap if using OPENAI_KEY
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY", "")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "scholarscout"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "haris12345"),
}

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

_CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "field_categories.json")
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
  - funding_type      TEXT    (e.g. "erasmus")

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

# Two anchor descriptions — query is routed to whichever is more similar
_ROUTE_ANCHORS = {
    "db": "find scholarships, programs, universities, deadlines, funding types, available opportunities, tuition fees, budget, cost, affordable, cheap, expensive, fully funded, self funded, scholarship money, financial aid, how much does it cost",
    "chat": "general conversation, advice, greetings, essay tips, visa questions, how things work",
}

_ANCHOR_CACHE = os.path.join(os.path.dirname(__file__), "anchor_vectors.npy")

def _load_anchor_vectors() -> dict:
    if os.path.exists(_ANCHOR_CACHE):
        data = np.load(_ANCHOR_CACHE, allow_pickle=True).item()
        if set(data.keys()) == set(_ROUTE_ANCHORS.keys()):
            return data
    vectors = {
        route: np.array(embeddings.embed_query(desc)).reshape(1, -1)
        for route, desc in _ROUTE_ANCHORS.items()
    }
    np.save(_ANCHOR_CACHE, vectors)
    return vectors

_anchor_vectors = _load_anchor_vectors()

class AgentState(TypedDict):
    query: str
    messages: list[BaseMessage]
    needs_db: bool
    sql: str
    db_results: list[dict]
    response: str

def router_node(state: AgentState) -> AgentState:
    query_vec = np.array(embeddings.embed_query(state["query"])).reshape(1, -1)
    scores = {
        route: cosine_similarity(query_vec, anchor)[0][0]
        for route, anchor in _anchor_vectors.items()
    }
    needs_db = False
    if scores["db"] > scores["chat"]:
        needs_db = True
    return {**state, "needs_db": needs_db}


def route_decision(state: AgentState) -> str:
    return "query_db" if state["needs_db"] else "respond"


def query_db_node(state: AgentState) -> AgentState:
    # Resolve field categories before SQL generation so the LLM uses exact values

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

    human = HumanMessage(content=state["query"])
    sql_result = llm.invoke([system, human])
    sql = sql_result.content.strip().strip("```sql").strip("```").strip()

    rows = []
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
    except Exception as e:
        rows = [{"error": str(e)}]

    return {**state, "sql": sql, "db_results": rows}


def respond_node(state: AgentState) -> AgentState:
    todays_date = datetime.datetime.now().strftime("%B %d, %Y")
    if state.get("db_results"):
        db_context = json.dumps(state["db_results"], indent=2, default=str)
        system = SystemMessage(content=f"""You are a helpful scholarship assistant.
        The user asked a question and the following data was retrieved from the database:
        {db_context}
        If the returned data is empty simply say:
        'Sorry I cannot find any relevant opportunity.'
        Otherwise answer clearly and concisely. Follow these rules:
        - Deadline analysis: today is {todays_date}. List programmes with upcoming deadlines first, then mention any with passed deadlines.
        - Financial details: if tuition fee or funding data is present, include it in your answer (e.g. EU fee, non-EU fee, whether fully funded or self-funded is available).
        - If the user asked about budget/affordability, highlight the cheapest options and flag any fully-funded programmes.
        - Keep the response structured and easy to read.
        - Currently you cover only erasmus universities. If questions asked about others
        You must provide general info and no further reference.
        """)
    else:
        system = SystemMessage(content="""You are a helpful scholarship assistant.
        Answer the user's question about scholarships, programs, and academic opportunities.
        If you don't have enough information, say so clearly.""")

    history = state.get("messages", [])
    result = llm.invoke([system] + history)
    updated_messages = history + [AIMessage(content=result.content)]
    return {**state, "response": result.content, "messages": updated_messages}


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("query_db", query_db_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "query_db": "query_db",
        "respond": "respond",
    })
    graph.add_edge("query_db", "respond")
    graph.add_edge("respond", END)

    checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


agent = build_agent()


def ask(query: str, thread_id: str = "1") -> str:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    # Load prior state from checkpointer; only supply new inputs
    prior = agent.get_state(config)
    prior_messages: list[BaseMessage] = (prior.values.get("messages") or []) if prior.values else []
    new_messages = prior_messages + [HumanMessage(content=query)]
    result = agent.invoke(
        {"query": query, "messages": new_messages, "needs_db": False, "sql": "", "db_results": [], "response": ""},
        config,
    )
    return result["response"]

if __name__ == "__main__":
    print("ScholarScout Agent (type 'exit' to quit)\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        print(f"Agent: {ask(query)}\n")
