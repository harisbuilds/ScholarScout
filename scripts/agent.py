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
# llm = ChatOllama(
#     base_url="https://ollama.com",
#     model="kimi-k2.5:cloud",   # or whatever cloud model you want
#     headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"},
#     temperature=0,
# )

_CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "field_categories.json")
with open(_CATEGORIES_PATH) as _f:
    FIELD_CATEGORIES: list[str] = json.load(_f)

_CATEGORIES_LIST = "\n".join(f"  - {c}" for c in FIELD_CATEGORIES)

DB_SCHEMA = f"""
Table: universities
Columns:
  - program_name      TEXT    (name of the scholarship/master program)
  - university_name   TEXT    (name of the university or consortium)
  - field_category    TEXT    (must be one of the allowed values listed below)
  - deadline_ir       DATE    (application deadline for international students)
  - deadline_non_ir   DATE    (application deadline for non-international students)
  - funding_type      TEXT    (e.g. "erasmus")

Allowed values for field_category:
{_CATEGORIES_LIST}
"""

class AgentState(TypedDict):
    query: str
    sql: str
    db_results: list[dict]
    response: str

def query_db_node(state: AgentState) -> AgentState:
    # Resolve field categories before SQL generation so the LLM uses exact values

    system = SystemMessage(content=f"""You are a SQL expert. Given the schema below, write a
    single PostgreSQL SELECT query that answers the user's question.
    {DB_SCHEMA}
    Rules:
    - Only use SELECT statements, no INSERT/UPDATE/DELETE. Return all columns with *
    - For field_category filters, always use exact equality (= or IN) with the allowed category values — provided along with SCHEMA.
    - Use ILIKE for text matching on program_name and university_name asked in query.
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
        If: returned schema is empty simply say:
        'Sorry I cannot find any relevant opportunity.'
        else:
        Answer the user's question clearly and concisely based on this context. Before answer make analysis
        of deadlines. If any of the deadline is less than current date of execution i.e ({todays_date})
        then mention the ones which are still available first and then the ones which has passed the deadline.
        """)
    else:
        system = SystemMessage(content="""You are a helpful scholarship assistant.
        Answer the user's question about scholarships, programs, and academic opportunities.
        If you don't have enough information, say so clearly.""")

    human = HumanMessage(content=state["query"])
    result = llm.invoke([system, human])
    return {**state, "response": result.content}


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("query_db", query_db_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("query_db")
    graph.add_edge("query_db", "respond")
    graph.add_edge("respond", END)

    checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)


agent = build_agent()


def ask(query: str) -> str:
    initial_state: AgentState = {
        "query": query,
        "sql": "",
        "db_results": [],
        "response": "",
    }
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    result = agent.invoke(initial_state, config)
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
