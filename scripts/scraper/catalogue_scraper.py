import csv
import logging
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.eacea.ec.europa.eu/scholarships/erasmus-mundus-catalogue_en"
TOTAL_PAGES = 11
DELAY = 1.5
OUTPUT_CSV = Path("programs.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_with_requests(url: str) -> BeautifulSoup | None:
    """GET a page and return a BeautifulSoup tree, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        log.warning("requests failed for %s: %s", url, exc)
        return None


def _fetch_with_selenium(url: str) -> BeautifulSoup | None:
    """Headless Chrome fallback using Selenium."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument(f"user-agent={HEADERS['User-Agent']}")

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts,
        )
        driver.get(url)
        # Wait until at least one article/listing card appears
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//article | //div[contains(@class,'views-row')]"))
        )
        soup = BeautifulSoup(driver.page_source, "lxml")
        driver.quit()
        return soup
    except Exception as exc:
        log.error("Selenium fallback also failed for %s: %s", url, exc)
        return None


def _parse_programmes(soup: BeautifulSoup) -> list[dict]:
    """
    Extract programme entries from a catalogue page soup.

    The EACEA site renders each programme inside an <article> element.
    Each article contains:
      - An <a> whose text is the programme name and href is the official site.
      - A sibling text node / <span> with the acronym.
      - A second <a> labelled "Project overview" pointing to erasmus-plus.ec.europa.eu.
    """
    results = []

    articles = soup.find_all("article")

    if not articles:
        articles = soup.select("div.views-row")

    if not articles:
        articles = soup.select("div.view-content li")

    for article in articles:
        links = article.find_all("a", href=True)
        if not links:
            continue

        # The first link in the card is the official programme URL
        official_link = None
        overview_link = None
        name = ""
        acronym = ""

        for a in links:
            href = a["href"].strip()
            text = a.get_text(strip=True)

            if "erasmus-plus.ec.europa.eu/projects" in href:
                overview_link = href
            elif href.startswith("http") and not official_link:
                official_link = href
                name = text

        # Acronym: usually a short word immediately following the name link
        raw_text = article.get_text(" ", strip=True)
        # Try to find short capitalised token between name and "Project overview"
        import re
        acronym_match = re.search(
            r"(?:^|\s)([A-Z][A-Z0-9\-]{1,20})\s*[-–]\s*Project overview",
            raw_text,
        )
        if acronym_match:
            acronym = acronym_match.group(1).strip()

        if name and official_link:
            results.append(
                {
                    "name": name,
                    "acronym": acronym,
                    "official_url": official_link,
                    "project_overview_url": overview_link or "",
                }
            )

    return results


def scrape_catalogue(output_path: Path = OUTPUT_CSV) -> list[dict]:
    """Scrape all catalogue pages and write programs.csv. Returns list of records."""
    all_programmes: list[dict] = []

    for page_num in range(TOTAL_PAGES):
        url = BASE_URL if page_num == 0 else f"{BASE_URL}?page={page_num}"
        log.info("Fetching catalogue page %d/%d  →  %s", page_num + 1, TOTAL_PAGES, url)

        soup = _fetch_with_requests(url)
        if soup is None:
            log.warning("Falling back to Selenium for page %d", page_num)
            soup = _fetch_with_selenium(url)

        if soup is None:
            log.error("Could not fetch page %d – skipping.", page_num)
            continue

        page_results = _parse_programmes(soup)
        log.info("  Found %d programmes on page %d", len(page_results), page_num + 1)
        all_programmes.extend(page_results)

        if page_num < TOTAL_PAGES - 1:
            time.sleep(DELAY)

    # De-duplicate (same official_url can appear on mirror Drupal paths)
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for prog in all_programmes:
        if prog["official_url"] not in seen_urls:
            seen_urls.add(prog["official_url"])
            unique.append(prog)

    log.info("Total unique programmes found: %d", len(unique))

    # Write CSV
    fieldnames = ["name", "acronym", "official_url", "project_overview_url"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique)

    log.info("Saved %d programmes to %s", len(unique), output_path)
    return unique


if __name__ == "__main__":
    scrape_catalogue()
