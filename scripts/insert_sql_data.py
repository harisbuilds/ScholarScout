import csv
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "scholarscout"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "haris12345"),
}

CSV_FILE = os.path.join(os.path.dirname(__file__), "universities.csv")

INSERT_SQL = """
    INSERT INTO universities
        (program_name, university_name, field_category,
         deadline_ir, deadline_non_ir, funding_type)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def parse_str(value: str) -> str | None:
    return value.strip() if value.strip() else None


def parse_bool(value: str) -> bool | None:
    if value.strip().lower() == "true":
        return True
    if value.strip().lower() == "false":
        return False
    return None


def parse_date(value: str) -> str | None:
    v = value.strip()
    if not v or v.upper() == "NULL":
        return None
    return v


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    inserted = 0
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute(INSERT_SQL, (
                parse_str(row["program_name"]),
                parse_str(row["university_name"]),
                parse_str(row["field_category"]),
                parse_date(row["deadline_ir"]),
                parse_date(row["deadline_non_ir"]),
                parse_str(row["funding_type"]),
            ))
            inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {inserted} rows into universities table.")


if __name__ == "__main__":
    main()
