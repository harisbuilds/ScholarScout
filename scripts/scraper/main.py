"""
Erasmus Mundus Scraper Orchestrator
Usage:
    python main.py               # run Phase 1 then Phase 2 (all programmes)
    python main.py --phase 1     # Phase 1 only  (builds programs.csv)
    python main.py --phase 2     # Phase 2 only  (reads programs.csv → JSON/CSV)
    python main.py --limit 10    # Phase 2 with first 10 programmes only (testing)

"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_CSV  = Path("programs.csv")
OUTPUT_JSON = Path("programs_detailed.json")
OUTPUT_CSV  = Path("programs_detailed.csv")


def run_phase1():
    """Scrape the EACEA catalogue → programs.csv"""
    log.info("═══════════════════════════════════════")
    log.info("  PHASE 1  –  Catalogue Scraper")
    log.info("═══════════════════════════════════════")
    from catalogue_scraper import scrape_catalogue
    programmes = scrape_catalogue(output_path=INPUT_CSV)
    log.info("Phase 1 complete: %d programmes saved to %s", len(programmes), INPUT_CSV)
    return programmes


def run_phase2(limit: int | None = None):
    """Deep-scrape each programme's official site → programs_detailed.json + .csv"""
    if not INPUT_CSV.exists():
        log.error(
            "programs.csv not found. Run Phase 1 first:  python main.py --phase 1"
        )
        sys.exit(1)

    log.info("═══════════════════════════════════════")
    log.info("  PHASE 2  –  Programme Detail Scraper")
    if limit:
        log.info("  (limited to first %d programmes)", limit)
    log.info("═══════════════════════════════════════")

    from program_scraper import scrape_all_programmes
    results = scrape_all_programmes(
        input_path=INPUT_CSV,
        output_json=OUTPUT_JSON,
        output_csv=OUTPUT_CSV,
        limit=limit,
    )

    total = len(results)
    with_details = sum(1 for r in results if r.get("has_details"))
    log.info(
        "Phase 2 complete: %d scraped | %d with details (%.0f%%)",
        total, with_details, (with_details / total * 100) if total else 0,
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Erasmus Mundus Catalyst Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only Phase 1 (catalogue) or Phase 2 (details). Default: both.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="(Phase 2) Limit scraping to the first N programmes. Useful for testing.",
    )
    args = parser.parse_args()

    if args.phase == 1:
        run_phase1()
    elif args.phase == 2:
        run_phase2(limit=args.limit)
    else:
        # Both phases
        run_phase1()
        run_phase2(limit=args.limit)


if __name__ == "__main__":
    main()
