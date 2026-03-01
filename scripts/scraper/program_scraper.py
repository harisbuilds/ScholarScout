"""
program_scraper.py
Scrape each programme's official website for scholarship tracking data.
"""

import csv
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 20
SELENIUM_TIMEOUT = 20
DELAY = 1.5            # seconds between requests
MAX_DESC_CHARS = 1000

INPUT_CSV = Path("programs.csv")
OUTPUT_JSON = Path("programs_detailed.json")
OUTPUT_CSV = Path("programs_detailed.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
)

DATE_PATTERNS = [
    # 15 January 2025  /  15 Jan 2025
    re.compile(rf"\b(\d{{1,2}}\s+{_MONTHS}\s+\d{{4}})\b", re.IGNORECASE),
    # January 15, 2025
    re.compile(rf"\b({_MONTHS}\s+\d{{1,2}},?\s+\d{{4}})\b", re.IGNORECASE),
    # 2025-01-15  or  2025/01/15
    re.compile(r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b"),
    # 15/01/2025  or  15-01-2025
    re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b"),
]

DEADLINE_KEYWORDS = re.compile(
    r"(?:application\s+deadline|apply\s+by|deadline\s+for\s+application|"
    r"closing\s+date|submission\s+deadline|applications?\s+close)",
    re.IGNORECASE,
)
START_KEYWORDS = re.compile(
    r"(?:programme?\s+start|course\s+start|intake\s+date|start\s+date|commenc)",
    re.IGNORECASE,
)
END_KEYWORDS = re.compile(
    r"(?:programme?\s+end|course\s+end|end\s+date|graduation\s+date|last\s+day)",
    re.IGNORECASE,
)

#city names for location extraction
LOCATION_HINTS = re.compile(
    r"\b(France|Germany|Spain|Italy|Portugal|Netherlands|Belgium|Sweden|Finland|"
    r"Denmark|Norway|Austria|Switzerland|Czech Republic|Poland|Hungary|Romania|"
    r"Greece|Croatia|Slovenia|Slovakia|Estonia|Latvia|Lithuania|Luxembourg|Malta|"
    r"Ireland|UK|United Kingdom|Scotland|Paris|Berlin|Madrid|Rome|Lisboa|Lisbon|"
    r"Amsterdam|Brussels|Stockholm|Vienna|Prague|Warsaw|Budapest|Bucharest|Athens|"
    r"Barcelona|Valencia|Munich|Hamburg|Turin|Naples|Ghent|Leuven|Utrecht|Groningen|"
    r"Toulouse|Lyon|Bordeaux|Marseille|Gdansk|Wroclaw|Krakow|Poznan|"
    r"Gothenburg|Malmo|Helsinki|Copenhagen|Oslo|Zurich|Geneva|Basel)\b",
    re.IGNORECASE,
)

def _fetch_requests(url: str) -> Optional[BeautifulSoup]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        log.debug("requests failed for %s: %s", url, exc)
        return None


_selenium_driver = None  # reuse a single driver instance across calls


def _get_driver():
    global _selenium_driver
    if _selenium_driver is None:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1280,900")
        opts.add_argument(f"user-agent={HEADERS['User-Agent']}")
        _selenium_driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts,
        )
    return _selenium_driver


def _fetch_selenium(url: str) -> Optional[BeautifulSoup]:
    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = _get_driver()
        driver.get(url)
        WebDriverWait(driver, SELENIUM_TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, "//body"))
        )
        time.sleep(2)  # allow JS to settle
        return BeautifulSoup(driver.page_source, "lxml")
    except Exception as exc:
        log.warning("Selenium failed for %s: %s", url, exc)
        return None


def quit_driver():
    global _selenium_driver
    if _selenium_driver:
        try:
            _selenium_driver.quit()
        except Exception:
            pass
        _selenium_driver = None

def _extract_dates_from_text(text: str) -> dict:
    """
    Scan text for date strings near scholarship-relevant keywords.
    Returns dict with keys: start_date, end_date, application_deadline.
    Each value is a string (first match found) or None.
    """
    result = {"start_date": None, "end_date": None, "application_deadline": None}

    def first_date_near(keyword_re, text, window=200):
        """Find the first date pattern within `window` chars of a keyword match."""
        for m in keyword_re.finditer(text):
            block = text[max(0, m.start() - window): m.end() + window]
            for dp in DATE_PATTERNS:
                dm = dp.search(block)
                if dm:
                    return dm.group(1)
        return None

    result["application_deadline"] = first_date_near(DEADLINE_KEYWORDS, text)
    result["start_date"] = first_date_near(START_KEYWORDS, text)
    result["end_date"] = first_date_near(END_KEYWORDS, text)

    if not any(result.values()):
        all_dates = []
        for dp in DATE_PATTERNS:
            all_dates.extend(dp.findall(text))
        if all_dates:
            result["start_date"] = all_dates[0]  # best-effort

    return result


def _extract_locations(text: str) -> list[str]:
    """Return deduplicated location mentions found in text."""
    matches = LOCATION_HINTS.findall(text)
    seen = set()
    locs = []
    for m in matches:
        key = m.strip().lower()
        if key not in seen:
            seen.add(key)
            locs.append(m.strip())
    return locs


def _extract_description(soup: BeautifulSoup) -> str:
    """Extract a human-readable description from the page."""
    # Priority: meta description → first meaningful <p> block → <h1>
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content", "").strip():
        return meta["content"].strip()

    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content", "").strip():
        return og_desc["content"].strip()

    # Try <main> or <article> content area first, then fall back to <body>
    container = soup.find("main") or soup.find("article") or soup.body
    if container:
        paragraphs = container.find_all("p")
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 60:  # skip trivially short paragraphs
                return text[:MAX_DESC_CHARS]

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return ""


def _extract_reference_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """
    Collect reference links (external + same domain) that look meaningful
    (not nav, footer, social media, or anchor-only links).
    """
    IGNORE_DOMAINS = {
        "facebook.com", "twitter.com", "x.com", "instagram.com",
        "linkedin.com", "youtube.com", "google.com",
    }
    base_domain = urlparse(base_url).netloc

    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)
        if parsed.scheme not in ("http", "https"):
            continue
        domain = parsed.netloc
        if any(ig in domain for ig in IGNORE_DOMAINS):
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        links.append(abs_url)

    return links[:30]  # cap at 30 links per page


# ──────────────────────────────────────────────
# Main scraping logic per programme
# ──────────────────────────────────────────────

def scrape_programme(row: dict, use_selenium_fallback: bool = True) -> dict:
    """
    Fetch and parse a single programme's official website.
    Returns an enriched dict ready for JSON/CSV output.
    """
    url = row.get("official_url", "").strip()
    record = {
        "name": row.get("name", ""),
        "acronym": row.get("acronym", ""),
        "official_url": url,
        "project_overview_url": row.get("project_overview_url", ""),
        "description": "",
        "start_date": None,
        "end_date": None,
        "application_deadline": None,
        "locations": [],
        "reference_links": [],
        "has_details": False,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }

    if not url:
        record["error"] = "no_url"
        return record

    log.info("  Scraping programme page: %s", url)

    soup = _fetch_requests(url)
    if soup is None and use_selenium_fallback:
        log.info("    → falling back to Selenium")
        soup = _fetch_selenium(url)

    if soup is None:
        record["error"] = "fetch_failed"
        return record

    # Remove noisy tags
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "iframe"]):
        tag.decompose()

    full_text = soup.get_text(" ", strip=True)

    record["description"] = _extract_description(soup)
    dates = _extract_dates_from_text(full_text)
    record["start_date"] = dates["start_date"]
    record["end_date"] = dates["end_date"]
    record["application_deadline"] = dates["application_deadline"]
    record["locations"] = _extract_locations(full_text)
    record["reference_links"] = _extract_reference_links(soup, url)

    record["has_details"] = any(
        [record["start_date"], record["end_date"], record["application_deadline"]]
    )

    return record


def scrape_all_programmes(
    input_path: Path = INPUT_CSV,
    output_json: Path = OUTPUT_JSON,
    output_csv: Path = OUTPUT_CSV,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Read programs.csv from Phase 1, scrape each programme website,
    and write programs_detailed.json + programs_detailed.csv.
    """
    # Read input CSV
    rows = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if limit:
        rows = rows[:limit]

    log.info("Scraping details for %d programmes…", len(rows))

    results = []
    for i, row in enumerate(rows, 1):
        log.info("[%d/%d] %s", i, len(rows), row.get("name", "?"))
        try:
            record = scrape_programme(row)
        except Exception as exc:
            log.error("Unhandled error for %s: %s", row.get("official_url"), exc)
            record = {**row, "error": str(exc), "has_details": False}
        results.append(record)

        if i < len(rows):
            time.sleep(DELAY)

    quit_driver()

    # ── Write JSON ──
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("Saved JSON → %s", output_json)

    # ── Write CSV (flatten list fields) ──
    csv_fields = [
        "name", "acronym", "official_url", "project_overview_url",
        "description", "start_date", "end_date", "application_deadline",
        "locations", "reference_links", "has_details", "scraped_at", "error",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rec in results:
            flat = dict(rec)
            flat["locations"] = "; ".join(rec.get("locations") or [])
            flat["reference_links"] = "; ".join(rec.get("reference_links") or [])
            writer.writerow(flat)
    log.info("Saved CSV → %s", output_csv)

    # ── Summary stats ──
    total = len(results)
    with_details = sum(1 for r in results if r.get("has_details"))
    errors = sum(1 for r in results if r.get("error"))
    log.info(
        "Done.  Total: %d  |  has_details=True: %d  |  Errors: %d",
        total, with_details, errors,
    )

    return results


if __name__ == "__main__":
    scrape_all_programmes()
