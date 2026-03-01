"""
rag_pipeline.py – End-to-end RAG pipeline.

Flow
────
  User query
    │
    ├─► query_parser.parse_query()
    │     → extracted filters (country, year, …)
    │     → cleaned semantic query
    │
    ├─► ScholarshipRAG.retrieve()
    │     → chosen retriever (default: HybridRetriever with RRF)
    │     → Milvus scalar pre-filter applied server-side
    │     → top-K scholarships returned
    │
    └─► ScholarshipRAG.generate()
          → context assembled from top-K results
          → LLM generates a grounded answer
          → (LLM is optional – falls back to a formatted summary if unavailable)
"""

from __future__ import annotations

import os
from typing import Any

from pymilvus import MilvusClient
from rich.console import Console

from configs.config import FINAL_K, MILVUS_URI, COLLECTION_NAME
from scripts.RAG.data_ingestion import SAMPLE_SCHOLARSHIPS, build_searchable_text, ingest_data, SEARCHABLE_TEXT_FIELD
from scripts.RAG.query_parser import parse_query
from retrievers import (
    BaseRetriever,
    HybridRetriever,
    get_all_retrievers,
)
from schema_setup import create_collection, get_client

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────────────────────────────────────

def build_context(results: list[dict]) -> str:
    """Format retrieved scholarships as a numbered context block for the LLM."""
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] **{r['title']}**\n"
            f"    Country      : {r['country']}\n"
            f"    University   : {r['university']}\n"
            f"    Field        : {r['field_of_study']}\n"
            f"    Year         : {r['year']}\n"
            f"    Deadline     : {r['last_date']}\n"
            f"    Link         : {r['link']}\n"
            f"    Description  : {r['description']}\n"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# RAG pipeline
# ──────────────────────────────────────────────────────────────────────────────

class ScholarshipRAG:
    """
    Scholarship Retrieval-Augmented Generation system.

    Parameters
    ----------
    retriever   : one of the retriever instances from retrievers.py
    llm_client  : optional OpenAI-compatible client for answer generation
                  (pass None to use the built-in fallback formatter)
    llm_model   : model name to use for generation
    """

    SYSTEM_PROMPT = (
        "You are a helpful scholarship advisor. "
        "Answer the user's question using ONLY the scholarship information "
        "provided in the context below. "
        "If no scholarships match, say so clearly. "
        "Be concise, specific, and always cite the scholarship name."
    )

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_client=None,
        llm_model: str = "gpt-4o-mini",
    ):
        self.retriever  = retriever
        self.llm_client = llm_client
        self.llm_model  = llm_model

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_K,
        auto_parse: bool = True,
        manual_filters: dict | None = None,
    ) -> tuple[list[dict], dict]:
        """
        Parse the query, apply filters, and retrieve top-K scholarships.

        Returns
        -------
        (results, parsed_info)
        """
        parsed = parse_query(query) if auto_parse else {
            "raw_query": query, "cleaned_query": query,
            "filters": manual_filters or {}, "extracted": {},
        }

        filters = {**parsed["filters"], **(manual_filters or {})}
        search_q = parsed["cleaned_query"] or query

        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
        console.print(f"[dim]Cleaned:[/dim] {search_q}")
        console.print(f"[dim]Filters:[/dim] {filters}")
        console.print(f"[dim]Retriever:[/dim] {self.retriever.name}\n")

        results = self.retriever.retrieve(search_q, filters=filters, top_k=top_k)
        return results, parsed

    # ── Generate ──────────────────────────────────────────────────────────────

    def generate(self, query: str, results: list[dict]) -> str:
        """Generate an answer grounded in the retrieved scholarships."""
        context = build_context(results)

        if self.llm_client is None:
            # ── Fallback: structured formatter (no LLM) ─────────────────────
            return self._fallback_answer(query, results, context)

        # ── LLM generation ──────────────────────────────────────────────────
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Query: {query}\n\nContext:\n{context}"},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as exc:
            console.print(f"[yellow]LLM call failed ({exc}), using fallback.[/yellow]")
            return self._fallback_answer(query, results, context)

    def _fallback_answer(self, query: str, results: list[dict], context: str) -> str:
        if not results:
            return (
                f"No scholarships were found matching your query: '{query}'. "
                "Try broadening your search (e.g. remove year or country filters)."
            )
        top = results[0]
        answer = (
            f"Based on your query '{query}', here are the top matching scholarships:\n\n"
            f"{context}\n"
            f"─────────────────────────────────────────\n"
            f"Top recommendation: **{top['title']}** in {top['country']} "
            f"({top['field_of_study']}, {top['year']}).\n"
            f"Apply before {top['last_date']} at {top['link']}."
        )
        return answer

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        top_k: int = FINAL_K,
        auto_parse: bool = True,
        manual_filters: dict | None = None,
    ) -> dict[str, Any]:
        """
        End-to-end: parse → retrieve → generate.

        Returns
        -------
        {
            "query":     str,
            "parsed":    dict,
            "results":   list[dict],
            "answer":    str,
        }
        """
        results, parsed = self.retrieve(query, top_k, auto_parse, manual_filters)
        answer_text = self.generate(query, results)

        console.rule("[bold green]Generated Answer[/bold green]")
        console.print(answer_text)
        console.print()

        return {
            "query":   query,
            "parsed":  parsed,
            "results": results,
            "answer":  answer_text,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Setup helper
# ──────────────────────────────────────────────────────────────────────────────

def setup_system(
    drop_existing: bool = False,
) -> tuple[MilvusClient, list[dict], dict[str, BaseRetriever]]:
    """
    One-shot setup: connect → create collection → ingest data → build retrievers.

    Returns
    -------
    (client, corpus, retrievers_dict)
    """
    client = get_client()
    create_collection(client, drop_if_exists=drop_existing)

    # Ingest only if collection is empty
    stats = client.get_collection_stats(COLLECTION_NAME)
    row_count = int(stats.get("row_count", 0))
    if row_count == 0:
        corpus = ingest_data(client, SAMPLE_SCHOLARSHIPS)
    else:
        console.print(f"[schema] Collection already has {row_count} records – skipping ingest.")
        # Reconstruct corpus with searchable_text for TF-IDF
        corpus = SAMPLE_SCHOLARSHIPS.copy()
        for s in corpus:
            s[SEARCHABLE_TEXT_FIELD] = build_searchable_text(s)

    retrievers = get_all_retrievers(client, corpus)
    return client, corpus, retrievers
