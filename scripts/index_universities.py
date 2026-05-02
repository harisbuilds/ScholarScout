import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient, DataType
from langchain_text_splitters import RecursiveCharacterTextSplitter

from configs.config import MILVUS_URI, MILVUS_COLLECTION, EMBEDDING_MODEL, EMBED_DIM, CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()

_DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
JSON_PATH     = os.path.join(_DATA_DIR, "universities_semantic_data.json")
PROGRESS_FILE = os.path.join(_DATA_DIR, "index_progress.json")

ALREADY_INDEXED = {"scraped_universities"}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

client_oai    = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY", ""))
client_milvus = MilvusClient(uri=MILVUS_URI)


def record_key(record: dict) -> str:
    """Stable unique key for a record — used to track what's been indexed."""
    return f"{record.get('_source', '')}|{record.get('official_url', '')}|{record.get('program_name', '')}"


def load_progress() -> set:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_progress(done: set) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f, indent=2)


def ensure_collection():
    if client_milvus.has_collection(MILVUS_COLLECTION):
        print(f"Collection '{MILVUS_COLLECTION}' already exists — appending to it")
        return

    schema = client_milvus.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",              DataType.INT64,        is_primary=True)
    schema.add_field("university_name", DataType.VARCHAR,      max_length=512)
    schema.add_field("url",             DataType.VARCHAR,      max_length=1024)
    schema.add_field("description",     DataType.VARCHAR,      max_length=8192)
    schema.add_field("embedding",       DataType.FLOAT_VECTOR, dim=EMBED_DIM)

    index_params = client_milvus.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="IP")

    client_milvus.create_collection(MILVUS_COLLECTION, schema=schema, index_params=index_params)
    print(f"Created collection '{MILVUS_COLLECTION}'")


def build_display_name(record: dict) -> str:
    program = (record.get("program_name") or "").strip()
    uni     = (record.get("university_name") or "").strip()
    if program and uni:
        return f"{program} | {uni}"
    return program or uni


def build_full_text(record: dict) -> str:
    name      = build_display_name(record)
    desc      = (record.get("description") or "").strip()
    sections  = record.get("sections") or {}
    admission = (sections.get("admission") or {}).get("content", "").strip()

    # scraped_* records were already indexed without fees — keep the same text
    # format so their embeddings stay comparable to the 58 already in Milvus.
    if (record.get("_source") or "").startswith("scraped_"):
        return "\n\n".join(filter(None, [name, desc, admission]))

    degree   = (record.get("degree_type") or "").strip()
    city     = (record.get("city") or "").strip()
    subject  = (record.get("subject") or "").strip()
    duration = (record.get("duration") or "").strip()
    langs    = ", ".join(record.get("languages") or [])
    fees     = (sections.get("fees") or {}).get("content", "").strip()
    meta     = ", ".join(filter(None, [degree, city, subject, duration, langs]))
    return "\n\n".join(filter(None, [name, meta, desc, admission, fees]))


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client_oai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def main():
    with open(JSON_PATH, encoding="utf-8") as f:
        all_records: list[dict] = json.load(f)

    records = [r for r in all_records if r.get("_source") not in ALREADY_INDEXED]
    skipped_source = len(all_records) - len(records)

    done = load_progress()
    pending  = [r for r in records if record_key(r) not in done]
    skipped_progress = len(records) - len(pending)

    print(f"Total:              {len(all_records)}")
    print(f"Skipped (source):   {skipped_source}  ({', '.join(ALREADY_INDEXED)})")
    print(f"Skipped (progress): {skipped_progress} already indexed this session")
    print(f"To index:           {len(pending)}")

    if not pending:
        print("Nothing left to index.")
        return

    ensure_collection()

    total_chunks = 0
    failed       = 0

    for record in tqdm(pending, desc="Indexing"):
        name      = build_display_name(record)
        url       = (record.get("official_url") or "").strip()
        full_text = build_full_text(record)
        chunks    = splitter.split_text(full_text)
        if not chunks:
            done.add(record_key(record))
            save_progress(done)
            continue

        try:
            embeddings = embed_texts(chunks)

            rows = [
                {
                    "university_name": name[:512],
                    "url":             url[:1024],
                    "description":     chunk[:8192],
                    "embedding":       embedding,
                }
                for chunk, embedding in zip(chunks, embeddings)
            ]

            client_milvus.insert(MILVUS_COLLECTION, rows)
            client_milvus.flush(MILVUS_COLLECTION)

            done.add(record_key(record))
            save_progress(done)
            total_chunks += len(rows)

        except Exception as e:
            tqdm.write(f"  FAILED: {name!r} — {e}")
            failed += 1
            continue

    print(f"\nDone — indexed {total_chunks} chunks from {len(pending) - failed} records")
    if failed:
        print(f"Failed: {failed} records (re-run the script to retry them)")


if __name__ == "__main__":
    main()
