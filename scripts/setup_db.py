"""
setup_db.py — Create tables and load data into the ScholarScout PostgreSQL database.

Usage:
    python scripts/setup_db.py
"""
import csv
import json
import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

UNIVERSITIES_CSV = os.path.join(_DATA_DIR, "universities.csv")
FINANCES_CSV     = os.path.join(_DATA_DIR, "finances.csv")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "scholarscout"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

_CREATE_UNIVERSITIES = """
CREATE TABLE IF NOT EXISTS universities (
    id              SERIAL PRIMARY KEY,
    program_name    TEXT NOT NULL,
    university_name TEXT NOT NULL,
    field_category  TEXT,
    deadline_ir     DATE,
    deadline_non_ir DATE,
    funding_type    TEXT
);
"""

_CREATE_FINANCES = """
CREATE TABLE IF NOT EXISTS finances (
    university_id               INT PRIMARY KEY,
    tuition_fee_eu_per_year     INT,
    tuition_fee_non_eu_per_year INT,
    scholarship_available       BOOLEAN,
    funding_category            JSONB,
    urls                        JSONB,
    CONSTRAINT finances_university_id_fkey
        FOREIGN KEY (university_id) REFERENCES universities(id)
);
"""

_INSERT_UNIVERSITY = """
    INSERT INTO universities
        (id, program_name, university_name, field_category,
         deadline_ir, deadline_non_ir, funding_type)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING
"""

_INSERT_FINANCE = """
    INSERT INTO finances
        (university_id, tuition_fee_eu_per_year, tuition_fee_non_eu_per_year,
         scholarship_available, funding_category, urls)
    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
    ON CONFLICT (university_id) DO NOTHING
"""


def _str(v):  return v.strip() or None
def _int(v):  v = v.strip(); return int(v) if v else None
def _date(v): v = v.strip(); return v if v and v.upper() != "NULL" else None
def _bool(v):
    v = v.strip().lower()
    return True if v == "true" else (False if v == "false" else None)
def _json(v):
    v = v.strip()
    if not v:
        return None
    try:
        return json.dumps(json.loads(v))
    except json.JSONDecodeError:
        return None


def create_tables(cur):
    cur.execute(_CREATE_UNIVERSITIES)
    cur.execute(_CREATE_FINANCES)
    print("Tables ready.")


def load_universities(cur):
    with open(UNIVERSITIES_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        cur.execute(_INSERT_UNIVERSITY, (
            _int(row["id"]),
            _str(row["program_name"]),
            _str(row["university_name"]),
            _str(row["field_category"]),
            _date(row["deadline_ir"]),
            _date(row["deadline_non_ir"]),
            _str(row["funding_type"]),
        ))
    # Advance the serial past the explicitly inserted IDs so future inserts don't collide
    cur.execute("SELECT setval('universities_id_seq', (SELECT MAX(id) FROM universities))")
    print(f"Loaded {len(rows)} universities.")


def load_finances(cur):
    with open(FINANCES_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        cur.execute(_INSERT_FINANCE, (
            _int(row["university_id"]),
            _int(row["tuition_fee_eu_per_year"]),
            _int(row["tuition_fee_non_eu_per_year"]),
            _bool(row["scholarship_available"]),
            _json(row["funding_category"]),
            _json(row["urls"]),
        ))
    print(f"Loaded {len(rows)} finance records.")


def main():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as e:
        print(f"Could not connect to database: {e}", file=sys.stderr)
        print("Make sure the database exists first — run scripts/scholarscout_schema.sql against your PostgreSQL instance.", file=sys.stderr)
        sys.exit(1)

    with conn:
        cur = conn.cursor()
        create_tables(cur)
        load_universities(cur)
        load_finances(cur)
        cur.close()

    conn.close()
    print("Database setup complete.")


if __name__ == "__main__":
    main()
