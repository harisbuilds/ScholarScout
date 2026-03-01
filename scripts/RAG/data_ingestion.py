"""
data_ingestion.py – Load scholarship data into Milvus.

The `build_searchable_text` helper creates a rich concatenated string used by
both Milvus BM25 and the local TF-IDF retriever so every retriever works on
the same textual representation.
"""

from __future__ import annotations

import json
from typing import Any

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from tqdm import tqdm

from configs import (
    COLLECTION_NAME,
    DENSE_MODEL_NAME,
    SEARCHABLE_TEXT_FIELD,
)


# ── Sample scholarship data ────────────────────────────────────────────────────
SAMPLE_SCHOLARSHIPS: list[dict[str, Any]] = [
    {
        "title": "DAAD Research Grant",
        "country": "Germany",
        "university": "Technical University of Munich",
        "field_of_study": "Data Science",
        "year": 2026,
        "last_date": "2025-11-30",
        "link": "https://www.daad.de/research-grant",
        "description": (
            "Fully funded research grant for international students pursuing "
            "Data Science, AI, and Machine Learning at top German universities. "
            "Includes monthly stipend, health insurance, and travel allowance."
        ),
    },
    {
        "title": "Humboldt Research Fellowship",
        "country": "Germany",
        "university": "Humboldt University Berlin",
        "field_of_study": "Computer Science",
        "year": 2026,
        "last_date": "2026-01-15",
        "link": "https://www.humboldt-foundation.de/fellowship",
        "description": (
            "Prestigious fellowship for postdoctoral researchers in Computer Science "
            "and related fields. Promotes long-term academic cooperation between "
            "Germany and international researchers."
        ),
    },
    {
        "title": "Chevening Scholarship",
        "country": "United Kingdom",
        "university": "University of Oxford",
        "field_of_study": "Public Policy",
        "year": 2026,
        "last_date": "2025-11-05",
        "link": "https://www.chevening.org/scholarship",
        "description": (
            "UK government's global scholarship program offering full funding for "
            "a one-year master's degree in Public Policy, International Relations, "
            "or Economics at leading UK universities."
        ),
    },
    {
        "title": "Fulbright Foreign Student Program",
        "country": "USA",
        "university": "MIT",
        "field_of_study": "Engineering",
        "year": 2026,
        "last_date": "2025-10-01",
        "link": "https://foreign.fulbrightonline.org",
        "description": (
            "Flagship US government exchange program enabling graduate students "
            "and young professionals to undertake graduate study, advanced research, "
            "or university teaching in Engineering and Technology."
        ),
    },
    {
        "title": "ETH Zurich Excellence Scholarship",
        "country": "Switzerland",
        "university": "ETH Zurich",
        "field_of_study": "Data Science",
        "year": 2026,
        "last_date": "2025-12-15",
        "link": "https://ethz.ch/excellence-scholarship",
        "description": (
            "Merit-based scholarship for outstanding master's students in Data Science, "
            "Mathematics, and Physics. Covers full tuition and living expenses "
            "at one of Europe's leading technical universities."
        ),
    },
    {
        "title": "Gates Cambridge Scholarship",
        "country": "United Kingdom",
        "university": "University of Cambridge",
        "field_of_study": "Biomedical Sciences",
        "year": 2026,
        "last_date": "2025-10-12",
        "link": "https://www.gatescambridge.org",
        "description": (
            "Full-cost award for outstanding applicants from outside the UK to pursue "
            "a full-time postgraduate degree in Biomedical Sciences, Life Sciences, "
            "or Medicine at the University of Cambridge."
        ),
    },
    {
        "title": "Erasmus Mundus Joint Master",
        "country": "Germany",
        "university": "Heidelberg University",
        "field_of_study": "Machine Learning",
        "year": 2026,
        "last_date": "2026-02-28",
        "link": "https://www.erasmus-mundus.eu/ml-scholarship",
        "description": (
            "European Commission funded joint master's program in Machine Learning "
            "and Artificial Intelligence across multiple partner universities including "
            "Heidelberg, Amsterdam, and Edinburgh. Full tuition waiver plus stipend."
        ),
    },
    {
        "title": "Swedish Institute Scholarship",
        "country": "Sweden",
        "university": "KTH Royal Institute of Technology",
        "field_of_study": "Sustainability",
        "year": 2026,
        "last_date": "2026-02-10",
        "link": "https://si.se/scholarship",
        "description": (
            "Full scholarship for global professionals in Sustainability, Environment, "
            "and Renewable Energy. Covers tuition, living expenses, travel, and insurance "
            "for master's programs at Swedish universities."
        ),
    },
    {
        "title": "KAIST International Scholarship",
        "country": "South Korea",
        "university": "KAIST",
        "field_of_study": "Robotics",
        "year": 2026,
        "last_date": "2025-09-30",
        "link": "https://admission.kaist.ac.kr/scholarship",
        "description": (
            "Full tuition waiver and stipend for international students in Robotics, "
            "Electrical Engineering, and AI at KAIST. Research assistantships also "
            "available for PhD students."
        ),
    },
    {
        "title": "Heinrich Böll Foundation Scholarship",
        "country": "Germany",
        "university": "Free University of Berlin",
        "field_of_study": "Environmental Studies",
        "year": 2026,
        "last_date": "2026-01-01",
        "link": "https://www.boell.de/scholarship",
        "description": (
            "Scholarship by the Green political foundation supporting students in "
            "Environmental Studies, Political Science, and Social Sciences at German "
            "universities. Emphasises civic engagement and democratic values."
        ),
    },
    {
        "title": "Japan MEXT Scholarship",
        "country": "Japan",
        "university": "University of Tokyo",
        "field_of_study": "Physics",
        "year": 2026,
        "last_date": "2025-08-31",
        "link": "https://www.mext.go.jp/scholarship",
        "description": (
            "Japanese government scholarship for international undergraduate and "
            "graduate students in Physics, Engineering, and Natural Sciences at "
            "leading Japanese universities including UTokyo, Kyoto, and Osaka."
        ),
    },
    {
        "title": "Australia Awards Scholarship",
        "country": "Australia",
        "university": "University of Melbourne",
        "field_of_study": "Agriculture",
        "year": 2026,
        "last_date": "2025-04-30",
        "link": "https://www.australiaawards.gov.au",
        "description": (
            "Australian government scholarship targeting students from developing nations "
            "for full-time bachelor or master's degrees in Agriculture, Food Security, "
            "and International Development at Australian universities."
        ),
    },
    {
        "title": "DAAD AI & Machine Learning Grant",
        "country": "Germany",
        "university": "RWTH Aachen University",
        "field_of_study": "Artificial Intelligence",
        "year": 2026,
        "last_date": "2025-12-01",
        "link": "https://www.daad.de/ai-grant-2026",
        "description": (
            "Specialized DAAD grant for students focusing on Artificial Intelligence, "
            "Deep Learning, and Neural Networks at RWTH Aachen and partner institutions. "
            "Includes access to HPC clusters and industry mentorship."
        ),
    },
    {
        "title": "Canada Vanier Graduate Scholarship",
        "country": "Canada",
        "university": "University of Toronto",
        "field_of_study": "Neuroscience",
        "year": 2026,
        "last_date": "2025-11-01",
        "link": "https://vanier.gc.ca",
        "description": (
            "Canada's most prestigious doctoral scholarship in Neuroscience, Health Sciences, "
            "and Natural Sciences at Canadian universities. Valued at CAD 50,000 per year "
            "for three years."
        ),
    },
    {
        "title": "Netherlands Fellowship Programme",
        "country": "Netherlands",
        "university": "Delft University of Technology",
        "field_of_study": "Data Science",
        "year": 2025,
        "last_date": "2024-04-01",
        "link": "https://www.nuffic.nl/fellowship",
        "description": (
            "Dutch government fellowship for professionals from developing countries "
            "to pursue master's degrees in Data Science, Water Management, and "
            "Urban Planning at Dutch universities."
        ),
    },
]


def build_searchable_text(scholarship: dict[str, Any]) -> str:
    """
    Create a rich text blob used by BM25 and TF-IDF.
    Repeating key fields (country, field_of_study) boosts their term weight.
    """
    return (
        f"Scholarship: {scholarship['title']}. "
        f"Country: {scholarship['country']}. "
        f"Country: {scholarship['country']}. "          # intentional repeat for TF weight
        f"University: {scholarship['university']}. "
        f"Field of Study: {scholarship['field_of_study']}. "
        f"Field of Study: {scholarship['field_of_study']}. "
        f"Year: {scholarship['year']}. "
        f"Deadline: {scholarship['last_date']}. "
        f"Description: {scholarship['description']}"
    )


def ingest_data(
    client: MilvusClient,
    scholarships: list[dict[str, Any]] | None = None,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """
    Embed and insert scholarship records into Milvus.

    Returns the list of scholarship dicts with `searchable_text` added
    so they can also be used by the local TF-IDF retriever.
    """
    if scholarships is None:
        scholarships = SAMPLE_SCHOLARSHIPS

    print(f"[ingest] Loading embedding model '{DENSE_MODEL_NAME}' …")
    model = SentenceTransformer(DENSE_MODEL_NAME)

    # Build searchable_text for every record
    for s in scholarships:
        s[SEARCHABLE_TEXT_FIELD] = build_searchable_text(s)

    # Batch embedding
    print(f"[ingest] Embedding {len(scholarships)} scholarships …")
    texts = [s[SEARCHABLE_TEXT_FIELD] for s in scholarships]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # Build insertion rows
    rows = []
    for s, emb in zip(scholarships, embeddings):
        rows.append({
            "title":            s["title"],
            "country":          s["country"],
            "university":       s["university"],
            "field_of_study":   s["field_of_study"],
            "year":             s["year"],
            "last_date":        s["last_date"],
            "link":             s["link"],
            "description":      s["description"],
            SEARCHABLE_TEXT_FIELD: s[SEARCHABLE_TEXT_FIELD],
            "dense_vector":     emb.tolist(),
            # sparse_vector is auto-generated by Milvus BM25 Function from searchable_text
        })

    # Insert in batches
    for i in tqdm(range(0, len(rows), batch_size), desc="Inserting batches"):
        batch = rows[i : i + batch_size]
        client.insert(collection_name=COLLECTION_NAME, data=batch)

    print(f"[ingest] Inserted {len(rows)} records into '{COLLECTION_NAME}'.")
    return scholarships
