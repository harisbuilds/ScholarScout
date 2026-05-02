"""
Index professor_opportunities.json into a separate Milvus collection.
Run from any directory:  python scripts/index_professors.py
"""
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient, DataType
from langchain_text_splitters import RecursiveCharacterTextSplitter

from configs.config import MILVUS_URI, PROFESSOR_COLLECTION, EMBEDDING_MODEL, EMBED_DIM, CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()

_DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
JSON_PATH     = os.path.join(_DATA_DIR, "professor_opportunities.json")
PROGRESS_FILE = os.path.join(_DATA_DIR, "professor_index_progress.json")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

client_oai    = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY", ""))
client_milvus = MilvusClient(uri=MILVUS_URI)


def record_key(r: dict) -> str:
    return f"{r.get('university', '')}|{r.get('title', '')}|{r.get('professor_name', '')}"


def load_progress() -> set:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_progress(done: set) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f, indent=2)


def ensure_collection():
    if client_milvus.has_collection(PROFESSOR_COLLECTION):
        print(f"Collection '{PROFESSOR_COLLECTION}' already exists — appending to it")
        return

    schema = client_milvus.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",              DataType.INT64,        is_primary=True)
    schema.add_field("professor_name",  DataType.VARCHAR,      max_length=256)
    schema.add_field("university",      DataType.VARCHAR,      max_length=256)
    schema.add_field("position_type",   DataType.VARCHAR,      max_length=64)
    schema.add_field("title",           DataType.VARCHAR,      max_length=512)
    schema.add_field("country",         DataType.VARCHAR,      max_length=128)
    schema.add_field("city",            DataType.VARCHAR,      max_length=128)
    schema.add_field("deadline",        DataType.VARCHAR,      max_length=32)
    schema.add_field("email",           DataType.VARCHAR,      max_length=256)
    schema.add_field("url",             DataType.VARCHAR,      max_length=1024)
    schema.add_field("funding",         DataType.VARCHAR,      max_length=512)
    schema.add_field("description",     DataType.VARCHAR,      max_length=8192)
    schema.add_field("embedding",       DataType.FLOAT_VECTOR, dim=EMBED_DIM)

    index_params = client_milvus.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="IP")

    client_milvus.create_collection(PROFESSOR_COLLECTION, schema=schema, index_params=index_params)
    print(f"Created collection '{PROFESSOR_COLLECTION}'")


def build_full_text(r: dict) -> str:
    """Build rich searchable text that covers all dimensions a student might search by."""
    areas = ", ".join(r.get("research_areas") or [])
    parts = [
        r.get("title", ""),
        f"{r.get('position_type', '')} position at {r.get('university', '')} — {r.get('city', '')}, {r.get('country', '')}",
        f"Supervisor: {r.get('professor_name', '')}",
        f"Department: {r.get('department', '')}",
        f"Research areas: {areas}",
        r.get("description", ""),
        f"Requirements: {r.get('requirements', '')}",
        f"Funding: {r.get('funding', '')}",
        f"Deadline: {r.get('deadline', '')}",
    ]
    return "\n\n".join(filter(None, parts))


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client_oai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def main():
    with open(JSON_PATH, encoding="utf-8") as f:
        all_records: list[dict] = json.load(f)

    done    = load_progress()
    pending = [r for r in all_records if record_key(r) not in done]
    print(f"Total: {len(all_records)} | Already indexed: {len(done)} | To index: {len(pending)}")

    if not pending:
        print("Nothing left to index.")
        return

    ensure_collection()

    total_chunks = 0
    failed       = 0

    for record in tqdm(pending, desc="Indexing professors"):
        key       = record_key(record)
        full_text = build_full_text(record)
        chunks    = splitter.split_text(full_text)
        if not chunks:
            done.add(key)
            save_progress(done)
            continue

        try:
            embeddings = embed_texts(chunks)

            rows = [
                {
                    "professor_name": (record.get("professor_name") or "")[:256],
                    "university":     (record.get("university")     or "")[:256],
                    "position_type":  (record.get("position_type")  or "")[:64],
                    "title":          (record.get("title")          or "")[:512],
                    "country":        (record.get("country")        or "")[:128],
                    "city":           (record.get("city")           or "")[:128],
                    "deadline":       (record.get("deadline")       or "")[:32],
                    "email":          (record.get("email")          or "")[:256],
                    "url":            (record.get("url")            or "")[:1024],
                    "funding":        (record.get("funding")        or "")[:512],
                    "description":    chunk[:8192],
                    "embedding":      embedding,
                }
                for chunk, embedding in zip(chunks, embeddings)
            ]

            client_milvus.insert(PROFESSOR_COLLECTION, rows)
            client_milvus.flush(PROFESSOR_COLLECTION)

            done.add(key)
            save_progress(done)
            total_chunks += len(rows)

        except Exception as e:
            tqdm.write(f"  FAILED: {record.get('title', '')!r} — {e}")
            failed += 1
            continue

    print(f"\nDone — {total_chunks} chunks from {len(pending) - failed} records into '{PROFESSOR_COLLECTION}'")
    if failed:
        print(f"Failed: {failed} (re-run to retry)")


if __name__ == "__main__":
    main()
