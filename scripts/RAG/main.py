"""
main.py – Demonstration of the Scholarship RAG system.

Run
───
  # Quick start (uses Milvus Lite – no server needed)
  python main.py

  # With OpenAI for LLM answer generation
  OPENAI_API_KEY=sk-... python main.py

Sections
────────
  1. System setup (schema creation + data ingestion)
  2. Single-query RAG with the recommended Hybrid (RRF) retriever
  3. Manual filter override (without auto-parsing)
  4. Side-by-side algorithm comparison
  5. Benchmark across multiple test cases
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))   # ensure local imports work

from rich.console import Console

from compare_algorithms import benchmark, compare_retrievers, print_comparison
from config import FINAL_K
from scripts.RAG.rag_pipeline import ScholarshipRAG, setup_system

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Setup
# ─────────────────────────────────────────────────────────────────────────────

console.rule("[bold blue]Scholarship RAG System – Demo[/bold blue]")
client, corpus, retrievers = setup_system(drop_existing=True)

# Optional: wire up OpenAI for actual LLM generation
llm_client = None
if os.getenv("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        llm_client = OpenAI()
        console.print("[green]OpenAI LLM client loaded.[/green]")
    except ImportError:
        console.print("[yellow]openai package not installed – using fallback formatter.[/yellow]")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single-query RAG   (the flagship use-case)
# ─────────────────────────────────────────────────────────────────────────────

console.rule("[bold]2. RAG – Hybrid Retriever (Recommended)[/bold]")

rag = ScholarshipRAG(
    retriever=retrievers["hybrid_rrf"],
    llm_client=llm_client,
)

# This is the example from the brief:
#   "Germany scholarships for 2026 specifically for Data Science"
#   → auto-extracts filters: country=Germany, year=2026
#   → cleaned query ("Data Science") drives dense+BM25 search
response = rag.answer(
    "Germany scholarships for 2026 specifically for Data Science"
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. More example queries
# ─────────────────────────────────────────────────────────────────────────────

console.rule("[bold]3. More Example Queries[/bold]")

example_queries = [
    # Semantic: no explicit keywords, relies on dense retrieval
    "I want to study machine learning in Europe",

    # Pure keyword filter: only year matters
    "Show me all 2026 scholarships",

    # Multi-filter: country + year, semantic core = robotics
    "South Korea robotics PhD scholarships 2026",

    # Deadline awareness
    "Scholarships in UK applying before November 2025",
]

for q in example_queries:
    console.rule(f"[cyan]Query: {q}[/cyan]")
    rag.answer(q, top_k=3)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Manual filter override (bypass auto-parsing)
# ─────────────────────────────────────────────────────────────────────────────

console.rule("[bold]4. Manual Filter Override[/bold]")

# Useful when the calling application already knows the filter values
# (e.g. from a structured UI form with country/year dropdowns)
manual_rag = ScholarshipRAG(retriever=retrievers["dense"])
manual_rag.answer(
    query="environmental sustainability programmes",
    auto_parse=False,
    manual_filters={"country": "Germany", "year": 2026},
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Side-by-side algorithm comparison
# ─────────────────────────────────────────────────────────────────────────────

console.rule("[bold]5. Side-by-Side Algorithm Comparison[/bold]")

comparison = compare_retrievers(
    query="Germany scholarships for 2026 specifically for Data Science",
    retrievers=retrievers,
    top_k=5,
)
print_comparison(comparison, top_k=5)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Benchmark with ground-truth labels
# ─────────────────────────────────────────────────────────────────────────────

console.rule("[bold]6. Benchmark (Precision / Recall / MRR / NDCG)[/bold]")

# Define test cases with ground-truth relevant titles.
# In the sample data the IDs assigned by Milvus are auto-generated INT64 values,
# so we identify "relevant" documents by title substring match on the TF-IDF
# corpus (which has sequential indices 0..N-1) for demonstration purposes.

def get_idx_by_title_keywords(*keywords: str) -> set:
    """Return corpus indices whose title contains all keywords (case-insensitive)."""
    result = set()
    for i, s in enumerate(corpus):
        if all(kw.lower() in s["title"].lower() for kw in keywords):
            result.add(i)
    return result


# For Milvus retrievers the IDs are Milvus-assigned INT64; for TF-IDF they are
# corpus indices.  For the benchmark we use TF-IDF corpus indices as a proxy,
# meaning the quantitative metrics only account for TF-IDF results accurately.
# In production, align IDs across all retrievers via a shared external ID field.
test_cases = [
    {
        "query": "Data Science scholarships Germany 2026",
        "relevant_ids": get_idx_by_title_keywords("DAAD") | get_idx_by_title_keywords("ETH"),
    },
    {
        "query": "Machine learning artificial intelligence Europe 2026",
        "relevant_ids": get_idx_by_title_keywords("Erasmus") | get_idx_by_title_keywords("DAAD", "AI"),
    },
    {
        "query": "environmental sustainability scholarship",
        "relevant_ids": get_idx_by_title_keywords("Heinrich") | get_idx_by_title_keywords("Swedish"),
    },
]

# Run benchmark – quantitative metrics are most meaningful for TF-IDF here
# (Milvus IDs won't match corpus indices, so Milvus metric scores will be 0)
benchmark(test_cases, {"tfidf": retrievers["tfidf"]}, top_k=FINAL_K)

console.rule("[bold green]Demo Complete[/bold green]")
console.print(
    "\n[bold]Summary of retrieval strategies:[/bold]\n"
    "  • [cyan]Dense (Semantic ANN)[/cyan]      – best for meaning/intent-based queries\n"
    "  • [cyan]BM25 (Milvus built-in)[/cyan]   – best for exact keyword/country/field matches\n"
    "  • [cyan]Hybrid RRF[/cyan]  [green]← RECOMMENDED[/green]  – combines both; robust across query types\n"
    "  • [cyan]Hybrid Weighted[/cyan]           – tune dense vs BM25 weight per use-case\n"
    "  • [cyan]TF-IDF (sklearn)[/cyan]          – baseline comparison; no embedding needed\n"
)
