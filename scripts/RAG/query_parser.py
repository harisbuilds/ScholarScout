"""
query_parser.py – Extract structured filters from a natural-language query.

This is a lightweight rule-based parser (regex + lookup tables).
For production, replace with an LLM-based entity extractor.

Example
───────
  parse_query("Germany scholarships for 2026 specifically for Data Science")
  →  {
       "cleaned_query": "scholarships specifically for Data Science",
       "filters":       {"country": "Germany", "year": 2026},
       "extracted":     {"countries": ["Germany"], "years": [2026], "fields": ["Data Science"]}
     }
"""

from __future__ import annotations

import re
from typing import Any


# ── Known entity lookup tables ─────────────────────────────────────────────────

KNOWN_COUNTRIES = [
    "Germany", "United Kingdom", "UK", "Britain", "England",
    "USA", "United States", "America", "US",
    "Switzerland", "Sweden", "Netherlands", "Holland",
    "Japan", "South Korea", "Korea",
    "Canada", "Australia",
    "France", "Spain", "Italy", "India", "China", "Brazil",
]

# Normalise common aliases to canonical country name
COUNTRY_ALIASES: dict[str, str] = {
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "us": "USA",
    "america": "USA",
    "united states": "USA",
    "holland": "Netherlands",
    "korea": "South Korea",
}

KNOWN_FIELDS = [
    "Data Science", "Machine Learning", "Artificial Intelligence", "AI",
    "Computer Science", "Engineering", "Physics", "Robotics",
    "Neuroscience", "Biomedical Sciences", "Environmental Studies",
    "Sustainability", "Agriculture", "Public Policy", "Economics",
    "International Relations",
]

FIELD_ALIASES: dict[str, str] = {
    "ai": "Artificial Intelligence",
    "ml": "Machine Learning",
}


def _extract_years(text: str) -> list[int]:
    """Extract 4-digit years in range 2000–2099."""
    return [int(m) for m in re.findall(r"\b(20[0-9]{2})\b", text)]


def _extract_countries(text: str) -> list[str]:
    """Case-insensitive match against known country list."""
    found = []
    lower = text.lower()
    for country in KNOWN_COUNTRIES:
        if re.search(r"\b" + re.escape(country.lower()) + r"\b", lower):
            canonical = COUNTRY_ALIASES.get(country.lower(), country)
            if canonical not in found:
                found.append(canonical)
    return found


def _extract_fields(text: str) -> list[str]:
    """Case-insensitive match against known field-of-study list."""
    found = []
    lower = text.lower()
    for field in KNOWN_FIELDS:
        if re.search(r"\b" + re.escape(field.lower()) + r"\b", lower):
            canonical = FIELD_ALIASES.get(field.lower(), field)
            if canonical not in found:
                found.append(canonical)
    return found


def _strip_entity_words(text: str, entities: list[str]) -> str:
    """Remove recognised entity words from the query to get the semantic core."""
    result = text
    for entity in entities:
        result = re.sub(r"\b" + re.escape(entity) + r"\b", "", result, flags=re.IGNORECASE)
    # Clean up extra whitespace / connector words left behind
    result = re.sub(r"\b(for|in|at|from|about|the|a|an|and|or|of|on)\b", " ", result, flags=re.IGNORECASE)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


def parse_query(
    query: str,
    extract_field_as_filter: bool = False,
) -> dict[str, Any]:
    """
    Parse a natural-language scholarship query.

    Parameters
    ----------
    query                  : raw user query
    extract_field_as_filter: if True, also add field_of_study to `filters`
                             so Milvus pre-filters on it.  Set False (default)
                             to let the dense/BM25 search handle field matching
                             semantically – usually more flexible.

    Returns
    -------
    {
        "raw_query":     str   – original query unchanged
        "cleaned_query": str   – query with entity words stripped (for vector search)
        "filters":       dict  – scalar filters to pass to Milvus/TF-IDF
        "extracted":     dict  – all extracted entities for transparency
    }
    """
    years     = _extract_years(query)
    countries = _extract_countries(query)
    fields    = _extract_fields(query)

    filters: dict[str, Any] = {}

    # Year filter: single year → equality; multiple → IN list
    if len(years) == 1:
        filters["year"] = years[0]
    elif len(years) > 1:
        filters["year"] = years

    # Country filter: single country → equality; multiple → IN list
    if len(countries) == 1:
        filters["country"] = countries[0]
    elif len(countries) > 1:
        filters["country"] = countries

    # Optionally add field_of_study filter (usually better left to retrieval)
    if extract_field_as_filter and len(fields) == 1:
        filters["field_of_study"] = fields[0]

    # Build cleaned query for dense/BM25 search
    all_entities = [str(y) for y in years] + countries + fields
    cleaned = _strip_entity_words(query, all_entities)
    if not cleaned:
        # Fallback: use field descriptions if everything was stripped
        cleaned = " ".join(fields) if fields else query

    return {
        "raw_query":     query,
        "cleaned_query": cleaned,
        "filters":       filters,
        "extracted": {
            "countries": countries,
            "years":     years,
            "fields":    fields,
        },
    }
