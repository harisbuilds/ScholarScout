"""
compare_algorithms.py – Side-by-side comparison of retrieval algorithms.

Comparison metrics implemented
──────────────────────────────
• Precision@K          – fraction of top-K results that are "relevant"
• Recall@K             – fraction of all relevant docs found in top-K
• MRR (Mean Reciprocal Rank) – rank of the first relevant result
• NDCG@K               – normalised discounted cumulative gain
• Overlap@K            – how many results appear in ALL retriever top-K lists
• Latency              – wall-clock time per retrieval call

When no ground-truth relevance labels are provided the comparison prints
ranked lists only (qualitative comparison).  Pass `relevant_ids` to enable
quantitative metrics.
"""

from __future__ import annotations

import math
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from config import FINAL_K
from query_parser import parse_query
from retrievers import BaseRetriever

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(results: list[dict], relevant_ids: set, k: int) -> float:
    top = results[:k]
    hits = sum(1 for r in top if r["id"] in relevant_ids)
    return hits / k if k else 0.0


def recall_at_k(results: list[dict], relevant_ids: set, k: int) -> float:
    top = results[:k]
    hits = sum(1 for r in top if r["id"] in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def mrr(results: list[dict], relevant_ids: set) -> float:
    for rank, r in enumerate(results, start=1):
        if r["id"] in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[dict], relevant_ids: set, k: int) -> float:
    def dcg(hits: list[int]) -> float:
        return sum(h / math.log2(i + 2) for i, h in enumerate(hits))

    top_k = results[:k]
    hits = [1 if r["id"] in relevant_ids else 0 for r in top_k]
    ideal = sorted(hits, reverse=True)
    d, id_ = dcg(hits), dcg(ideal)
    return d / id_ if id_ else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Single query comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare_retrievers(
    query: str,
    retrievers: dict[str, BaseRetriever],
    top_k: int = FINAL_K,
    relevant_ids: set | None = None,
    auto_parse: bool = True,
) -> dict[str, Any]:
    """
    Run all retrievers on `query`, measure latency, and compute metrics.

    Parameters
    ----------
    query         : natural-language query
    retrievers    : dict of {name: retriever_instance}
    top_k         : number of results per retriever
    relevant_ids  : set of "ground-truth relevant" document IDs (optional)
    auto_parse    : if True, extract filters and cleaned query from `query`

    Returns
    -------
    dict with keys: parsed, results_per_retriever, metrics_per_retriever
    """
    parsed = parse_query(query) if auto_parse else {
        "raw_query": query,
        "cleaned_query": query,
        "filters": {},
        "extracted": {},
    }

    search_query = parsed["cleaned_query"] or parsed["raw_query"]
    filters      = parsed["filters"]

    all_results: dict[str, list[dict]] = {}
    latencies:   dict[str, float]      = {}

    for name, retriever in retrievers.items():
        t0 = time.perf_counter()
        try:
            results = retriever.retrieve(search_query, filters=filters, top_k=top_k)
        except Exception as exc:
            console.print(f"[red][{name}] Error: {exc}[/red]")
            results = []
        latencies[name]   = round((time.perf_counter() - t0) * 1000, 1)  # ms
        all_results[name] = results

    # Compute overlap: IDs present in every retriever's results
    id_sets = [set(r["id"] for r in res) for res in all_results.values() if res]
    overlap = id_sets[0].intersection(*id_sets[1:]) if id_sets else set()

    # Compute quantitative metrics if ground truth provided
    metrics: dict[str, dict] = {}
    if relevant_ids:
        for name, results in all_results.items():
            metrics[name] = {
                f"P@{top_k}":    round(precision_at_k(results, relevant_ids, top_k), 4),
                f"R@{top_k}":    round(recall_at_k(results, relevant_ids, top_k), 4),
                "MRR":           round(mrr(results, relevant_ids), 4),
                f"NDCG@{top_k}": round(ndcg_at_k(results, relevant_ids, top_k), 4),
                "latency_ms":    latencies[name],
            }

    return {
        "parsed":                  parsed,
        "results_per_retriever":   all_results,
        "metrics_per_retriever":   metrics,
        "latencies_ms":            latencies,
        "overlap_ids":             overlap,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison(comparison: dict[str, Any], top_k: int = FINAL_K) -> None:
    """Render comparison output to the terminal using Rich tables."""
    parsed = comparison["parsed"]

    console.rule("[bold cyan]Scholarship RAG – Retrieval Comparison[/bold cyan]")
    console.print(f"[bold]Original query :[/bold] {parsed['raw_query']}")
    console.print(f"[bold]Cleaned query  :[/bold] {parsed['cleaned_query']}")
    console.print(f"[bold]Extracted filters:[/bold] {parsed['filters']}")
    console.print(f"[bold]Extracted entities:[/bold] {parsed['extracted']}")
    console.print()

    # ── Per-retriever results table ───────────────────────────────────────────
    for retriever_name, results in comparison["results_per_retriever"].items():
        latency = comparison["latencies_ms"].get(retriever_name, "?")
        console.rule(f"[yellow]{retriever_name}[/yellow]  "
                     f"[dim]({latency} ms)[/dim]")

        if not results:
            console.print("[red]  No results returned.[/red]")
            continue

        tbl = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
        tbl.add_column("Rank", style="dim", width=4)
        tbl.add_column("Score",     width=8)
        tbl.add_column("Title",     width=35)
        tbl.add_column("Country",   width=12)
        tbl.add_column("Field",     width=18)
        tbl.add_column("Year",      width=6)
        tbl.add_column("Deadline",  width=12)

        for rank, r in enumerate(results, start=1):
            tbl.add_row(
                str(rank),
                str(r["score"]),
                r["title"],
                r["country"],
                r["field_of_study"],
                str(r["year"]),
                r["last_date"],
            )
        console.print(tbl)

    # ── Overlap ───────────────────────────────────────────────────────────────
    console.print()
    if comparison["overlap_ids"]:
        console.print(f"[green]IDs appearing in ALL retrievers' top-{top_k}:[/green] "
                      f"{comparison['overlap_ids']}")
    else:
        console.print(f"[yellow]No common IDs across all retrievers' top-{top_k}.[/yellow]")

    # ── Quantitative metrics (if ground truth was provided) ───────────────────
    if comparison["metrics_per_retriever"]:
        console.print()
        console.rule("[bold]Quantitative Metrics (ground truth provided)[/bold]")
        metric_keys = list(next(iter(comparison["metrics_per_retriever"].values())).keys())
        mtbl = Table(show_header=True, header_style="bold blue", padding=(0, 1))
        mtbl.add_column("Retriever", width=35)
        for k in metric_keys:
            mtbl.add_column(k, width=12)
        for name, m in comparison["metrics_per_retriever"].items():
            mtbl.add_row(name, *[str(m[k]) for k in metric_keys])
        console.print(mtbl)


# ──────────────────────────────────────────────────────────────────────────────
# Batch benchmark
# ──────────────────────────────────────────────────────────────────────────────

def benchmark(
    test_cases: list[dict[str, Any]],
    retrievers: dict[str, BaseRetriever],
    top_k: int = FINAL_K,
) -> dict[str, dict]:
    """
    Run multiple test cases and aggregate metrics per retriever.

    Each test_case dict:
      {
        "query":        str,
        "relevant_ids": set[int]   # ground-truth relevant IDs
      }
    """
    aggregated: dict[str, dict[str, list]] = {
        name: {"P": [], "R": [], "MRR": [], "NDCG": [], "lat": []}
        for name in retrievers
    }

    for tc in test_cases:
        result = compare_retrievers(
            query=tc["query"],
            retrievers=retrievers,
            top_k=top_k,
            relevant_ids=tc.get("relevant_ids"),
        )
        for name, m in result["metrics_per_retriever"].items():
            aggregated[name]["P"].append(m.get(f"P@{top_k}", 0))
            aggregated[name]["R"].append(m.get(f"R@{top_k}", 0))
            aggregated[name]["MRR"].append(m.get("MRR", 0))
            aggregated[name]["NDCG"].append(m.get(f"NDCG@{top_k}", 0))
            aggregated[name]["lat"].append(result["latencies_ms"].get(name, 0))

    def mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    summary = {}
    for name, vals in aggregated.items():
        summary[name] = {
            f"Mean P@{top_k}":    mean(vals["P"]),
            f"Mean R@{top_k}":    mean(vals["R"]),
            "Mean MRR":           mean(vals["MRR"]),
            f"Mean NDCG@{top_k}": mean(vals["NDCG"]),
            "Mean Latency (ms)":  mean(vals["lat"]),
        }

    # Print summary table
    console.rule("[bold]Benchmark Summary[/bold]")
    keys = list(next(iter(summary.values())).keys())
    tbl = Table(show_header=True, header_style="bold green", padding=(0, 1))
    tbl.add_column("Retriever", width=35)
    for k in keys:
        tbl.add_column(k, width=18)
    for name, m in summary.items():
        tbl.add_row(name, *[str(m[k]) for k in keys])
    console.print(tbl)

    return summary
