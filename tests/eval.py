"""
Evaluate ScholarScout routing accuracy and response quality.

Routing test (default):
    Runs only router_node on every labeled test case. Fast and cheap —
    requires only the embedding API, no DB or Milvus.

Full pipeline test (--full):
    Runs route → fetch → respond for every test case, then scores each
    response with an LLM judge (quality 1-3) and a faithfulness check
    (are all claims grounded in the retrieved data?).
    Requires DB and Milvus to be running.

Usage:
    python tests/eval.py
    python tests/eval.py --full
    python tests/eval.py --full --output results.json
"""
import argparse
import json
import os
import re
import sys
import time

# Project root and scripts/ must both be on the path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import agent as ag
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

TEST_QUERIES_FILE = "tests/test_queries.txt"

MOCK_PROFILE = {"degree_level": "MS", "field": "Computer Science"}


def load_test_cases(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"\[(\w+)\]\s+(.+)", line)
            if m:
                cases.append({"expected": m.group(1), "query": m.group(2)})
    return cases


def _blank_state(query: str) -> ag.AgentState:
    return {
        "query": query, "query_vec": [],
        "messages": [HumanMessage(content=query)],
        "needs_db": False, "needs_vector": False, "needs_professor": False,
        "sql": "", "db_results": [], "vector_results": [],
        "professor_results": [], "web_results": [], "response": "",
        "validation_passed": False, "retry_count": 0,
        "user_profile": MOCK_PROFILE, "awaiting_profile": False,
    }


def _predicted_route(state: ag.AgentState) -> str:
    if state.get("needs_professor"):
        return "professor_search"
    if state["needs_db"] and state["needs_vector"]:
        return "hybrid"
    if state["needs_db"]:
        return "query_db"
    if state["needs_vector"]:
        return "vector_search"
    return "respond"


def _faithfulness_judge(context: str, response: str) -> tuple[bool, str]:
    """Check whether the response is grounded in the retrieved context.
    Returns (is_faithful, one-sentence reason)."""
    system = SystemMessage(content="""You are auditing a scholarship assistant for hallucinations.
You will be given the RETRIEVED DATA that was available to the assistant, and the RESPONSE it produced.

Decide whether every specific factual claim in the response (program names, universities, deadlines,
fees, URLs, admission requirements, funding details) is supported by the retrieved data.

Reply on two lines:
Line 1: FAITHFUL  or  UNFAITHFUL
Line 2: One sentence explaining your verdict (quote the unsupported claim if UNFAITHFUL).""")
    human = HumanMessage(content=f"RETRIEVED DATA:\n{context[:2000]}\n\nRESPONSE:\n{response[:800]}")
    try:
        raw   = ag.llm_fast.invoke([system, human]).content.strip()
        lines = raw.splitlines()
        verdict = lines[0].strip().upper()
        reason  = lines[1].strip() if len(lines) > 1 else ""
        return verdict.startswith("FAITHFUL"), reason
    except Exception:
        return True, "judge failed"


def _llm_judge(query: str, response: str) -> int:
    """Score a response 1–3. Returns -1 on failure."""
    system = SystemMessage(content="""You are evaluating a scholarship assistant's response.
Score it from 1 to 3:
  3 — Directly answers the query with specific, useful information
  2 — Partially answers the query but is vague or missing key details
  1 — Does not answer the query, is off-topic, or only says it has no information

Reply with ONLY a single digit: 1, 2, or 3.""")
    human = HumanMessage(content=f"Query: {query}\n\nResponse:\n{response[:800]}")
    try:
        verdict = ag.llm_fast.invoke([system, human]).content.strip()
        return int(verdict[0]) if verdict and verdict[0] in "123" else -1
    except Exception:
        return -1


def run_routing(cases: list[dict]) -> dict:
    results = []
    console.print(f"\n[bold]Running routing tests[/bold] ({len(cases)} queries)…\n")

    for case in cases:
        state   = _blank_state(case["query"])
        t0      = time.perf_counter()
        state   = ag.router_node(state)
        elapsed = (time.perf_counter() - t0) * 1000

        predicted = _predicted_route(state)
        correct   = predicted == case["expected"]
        results.append({
            "query":      case["query"],
            "expected":   case["expected"],
            "predicted":  predicted,
            "correct":    correct,
            "latency_ms": round(elapsed, 1),
        })

    total   = len(results)
    n_ok    = sum(r["correct"] for r in results)
    routes  = sorted({r["expected"] for r in results})

    per_route = {}
    for route in routes:
        subset = [r for r in results if r["expected"] == route]
        n      = len(subset)
        ok     = sum(r["correct"] for r in subset)
        per_route[route] = {"correct": ok, "total": n, "accuracy": ok / n}

    mean_lat = sum(r["latency_ms"] for r in results) / total

    return {
        "accuracy":         n_ok / total,
        "correct":          n_ok,
        "total":            total,
        "mean_latency_ms":  round(mean_lat, 1),
        "per_route":        per_route,
        "details":          results,
    }


def run_full(cases: list[dict]) -> dict:
    results = []
    console.print(f"\n[bold]Running full pipeline tests[/bold] ({len(cases)} queries)…\n")

    for i, case in enumerate(cases, 1):
        console.print(f"  [{i}/{len(cases)}] {case['query'][:70]}…")

        state   = _blank_state(case["query"])
        timings = {}

        t0 = time.perf_counter()
        state = ag.router_node(state)
        timings["routing_ms"] = (time.perf_counter() - t0) * 1000

        state = ag.profile_gate_node(state)
        predicted = _predicted_route(state)

        t0 = time.perf_counter()
        if state.get("needs_professor"):
            state = ag.professor_search_node(state)
        elif state["needs_db"] and state["needs_vector"]:
            state = ag.hybrid_node(state)
        elif state["needs_db"]:
            state = ag.query_db_node(state)
        elif state["needs_vector"]:
            state = ag.vector_search_node(state)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        t0    = time.perf_counter()
        state = ag.respond_node(state)
        timings["llm_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = sum(timings.values())

        db_error  = any("error" in r for r in state.get("db_results", []))
        db_hits   = len(state.get("db_results", []))
        vec_hits  = len(state.get("vector_results", []))
        prof_hits = len(state.get("professor_results", []))
        has_data  = bool(db_hits or vec_hits or prof_hits)

        judge = _llm_judge(case["query"], state["response"])

        faithful, faithful_reason = None, None
        if has_data:
            context = ag._build_context(state)
            faithful, faithful_reason = _faithfulness_judge(context, state["response"])

        results.append({
            "query":            case["query"],
            "expected":         case["expected"],
            "predicted":        predicted,
            "route_correct":    predicted == case["expected"],
            "db_error":         db_error,
            "db_hits":          db_hits,
            "vec_hits":         vec_hits,
            "prof_hits":        prof_hits,
            "has_data":         has_data,
            "judge_score":      judge,
            "faithful":         faithful,
            "faithful_reason":  faithful_reason,
            "timings":          {k: round(v, 1) for k, v in timings.items()},
            "response":         state["response"],
        })

    total          = len(results)
    db_cases       = [r for r in results if r["expected"] in ("query_db", "hybrid")]
    vec_cases      = [r for r in results if r["expected"] in ("vector_search", "hybrid")]
    scored         = [r for r in results if r["judge_score"] > 0]
    faith_cases    = [r for r in results if r["faithful"] is not None]

    def _mean(vals): return round(sum(vals) / len(vals), 3) if vals else None
    def _pct(vals):  return round(sum(vals) / len(vals) * 100, 1) if vals else None

    return {
        "routing_accuracy":  _pct([r["route_correct"] for r in results]),
        "db_error_rate":     _pct([r["db_error"]       for r in db_cases]),
        "db_hit_rate":       _pct([r["db_hits"] > 0    for r in db_cases]),
        "vec_hit_rate":      _pct([r["vec_hits"] > 0   for r in vec_cases]),
        "mean_judge_score":  _mean([r["judge_score"]   for r in scored]),
        "faithfulness_rate": _pct([r["faithful"]       for r in faith_cases]),
        "n_faithfulness":    len(faith_cases),
        "mean_total_ms":     _mean([r["timings"]["total_ms"]    for r in results]),
        "mean_routing_ms":   _mean([r["timings"]["routing_ms"]  for r in results]),
        "mean_retrieval_ms": _mean([r["timings"]["retrieval_ms"] for r in results]),
        "mean_llm_ms":       _mean([r["timings"]["llm_ms"]      for r in results]),
        "n_scored":          len(scored),
        "total":             total,
        "details":           results,
    }


def print_routing_report(m: dict):
    console.rule("[bold cyan]Routing Accuracy")

    t = Table(box=box.SIMPLE_HEAD, show_footer=False)
    t.add_column("Route",    style="cyan")
    t.add_column("Correct",  justify="right")
    t.add_column("Total",    justify="right")
    t.add_column("Accuracy", justify="right")
    for route, s in sorted(m["per_route"].items()):
        color = "green" if s["accuracy"] == 1.0 else ("yellow" if s["accuracy"] >= 0.7 else "red")
        t.add_row(route, str(s["correct"]), str(s["total"]),
                  f"[{color}]{s['accuracy']*100:.0f}%[/{color}]")
    console.print(t)

    overall_color = "green" if m["accuracy"] >= 0.8 else ("yellow" if m["accuracy"] >= 0.6 else "red")
    console.print(f"  Overall accuracy : [{overall_color}]{m['accuracy']*100:.1f}%[/{overall_color}]  ({m['correct']}/{m['total']})")
    console.print(f"  Mean latency     : {m['mean_latency_ms']} ms / query\n")

    misses = [r for r in m["details"] if not r["correct"]]
    if misses:
        console.print("[bold yellow]Misclassified queries:[/bold yellow]")
        for r in misses:
            console.print(f"  [red]✗[/red] expected=[cyan]{r['expected']}[/cyan]  got=[yellow]{r['predicted']}[/yellow]  {r['query'][:80]}")
        console.print()


def print_full_report(m: dict):
    console.rule("[bold cyan]Full Pipeline Results")

    summary = Table(box=box.SIMPLE_HEAD)
    summary.add_column("Metric",  style="cyan")
    summary.add_column("Value",   justify="right")

    def row(label, val, suffix=""):
        if val is None:
            summary.add_row(label, "[dim]n/a[/dim]")
        else:
            color = "green" if (isinstance(val, float) and val >= 80) else "white"
            summary.add_row(label, f"[{color}]{val}{suffix}[/{color}]")

    row("Routing accuracy",   m["routing_accuracy"],  "%")
    row("DB hit rate",        m["db_hit_rate"],        "%")
    row("DB error rate",      m["db_error_rate"],      "%")
    row("Vector hit rate",    m["vec_hit_rate"],       "%")
    row("Mean judge score",   m["mean_judge_score"],   f" / 3  (n={m['n_scored']})")
    row("Faithfulness rate",  m["faithfulness_rate"],  f"%  (n={m['n_faithfulness']})")
    summary.add_row("", "")
    row("Mean total latency",    m["mean_total_ms"],     " ms")
    row("  → routing",          m["mean_routing_ms"],   " ms")
    row("  → retrieval",        m["mean_retrieval_ms"], " ms")
    row("  → LLM",              m["mean_llm_ms"],       " ms")
    console.print(summary)

    console.print("\n[bold]Per-query breakdown:[/bold]")
    detail = Table(box=box.SIMPLE, show_lines=False)
    detail.add_column("#",        width=3,  justify="right")
    detail.add_column("Route",    width=14, style="cyan")
    detail.add_column("DB",       width=5,  justify="right")
    detail.add_column("Vec",      width=5,  justify="right")
    detail.add_column("Score",    width=7,  justify="right")
    detail.add_column("Faith",    width=7,  justify="right")
    detail.add_column("Total ms", width=9,  justify="right")
    detail.add_column("Query",    no_wrap=False)

    for i, r in enumerate(m["details"], 1):
        route_str = r["predicted"]
        if not r["route_correct"]:
            route_str = f"[red]{r['predicted']}[/red] ([dim]{r['expected']}[/dim])"
        score = str(r["judge_score"]) if r["judge_score"] > 0 else "[dim]-[/dim]"
        db_err = "[red]ERR[/red]" if r["db_error"] else str(r["db_hits"])
        if r["faithful"] is None:
            faith_str = "[dim]n/a[/dim]"
        elif r["faithful"]:
            faith_str = "[green]✓[/green]"
        else:
            faith_str = "[red]✗[/red]"
        detail.add_row(
            str(i), route_str, db_err, str(r["vec_hits"]),
            score, faith_str, str(r["timings"]["total_ms"]), r["query"][:70],
        )
    console.print(detail)

    unfaithful = [r for r in m["details"] if r["faithful"] is False]
    if unfaithful:
        console.print("\n[bold yellow]Hallucinations detected:[/bold yellow]")
        for r in unfaithful:
            console.print(f"  [red]✗[/red] {r['query'][:70]}")
            console.print(f"    [dim]{r['faithful_reason']}[/dim]")
        console.print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ScholarScout agent.")
    parser.add_argument("--full",   action="store_true", help="Run full pipeline + LLM judge")
    parser.add_argument("--output", metavar="FILE",      help="Save results to JSON file")
    args = parser.parse_args()

    cases = load_test_cases(TEST_QUERIES_FILE)
    if not cases:
        console.print("[red]No test cases found in test_queries.txt[/red]")
        sys.exit(1)

    console.print(f"[bold]ScholarScout Eval[/bold] — {len(cases)} test cases loaded")

    if args.full:
        metrics = run_full(cases)
        print_full_report(metrics)
    else:
        metrics = run_routing(cases)
        print_routing_report(metrics)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        console.print(f"Results saved to [cyan]{args.output}[/cyan]")


if __name__ == "__main__":
    main()
