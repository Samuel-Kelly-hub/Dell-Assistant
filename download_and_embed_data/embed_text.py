import uuid

import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

MODEL_NAME = "BAAI/bge-large-en-v1.5"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "chunks"

BATCH_SIZE = 64
CHUNK_OVERLAP_TOKENS = 50

model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer
max_tokens = model.max_seq_length
vector_size = model.get_sentence_embedding_dimension()


def split_by_tokens(text):
    """Split text into chunks based on token limit with overlap."""
    ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    if len(ids) <= max_tokens:
        return [text]

    step = max_tokens - CHUNK_OVERLAP_TOKENS

    chunks = []
    for start in range(0, len(ids), step):
        chunk_ids = ids[start : start + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if start + max_tokens >= len(ids):
            break
    return chunks


def ensure_collection(client):
    """Create or validate Qdrant collection with correct vector configuration."""
    print("ensure_collection: building vectors_config", flush=True)

    vectors_config = qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
    print("ensure_collection: vectors_config built", flush=True)
    print("ensure_collection: get_collection", flush=True)

    try:
        info = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("ensure_collection: create_collection", flush=True)
        client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vectors_config)
        info = client.get_collection(COLLECTION_NAME)

    existing_size = info.config.params.vectors.size
    if existing_size != vector_size:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' exists with vector size {existing_size}, "
            f"but the embedding model produces {vector_size}. "
            "Use a different collection name."
        )
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="product",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise


def iter_records(df_to_embed):
    """Yield (id, text, payload) tuples for each row/chunk in the DataFrame."""
    for row_i, (idx, row) in enumerate(df_to_embed.iterrows(), start=1):

        if row_i % 100 == 0:
            print(f"Prepared rows={row_i}/{len(df_to_embed)}")

        text = row["text"]
        title = row["title"]

        title = "" if pd.isna(title) else title

        embed_text = f"{title}\n\n{text}" if title else text
        chunks = split_by_tokens(embed_text)

        if len(chunks) == 1:
            payload = {"text": chunks[0], "url": row["url"], "product": row["product"], "title": title}

            try:
                int_id = int(idx)
                use_int_id = 0 <= int_id <= 2**63 - 1
            except (TypeError, ValueError):
                use_int_id = False

            yield (int_id if use_int_id else str(uuid.uuid4())), chunks[0], payload
            continue

        for chunk_text in chunks:
            yield str(uuid.uuid4()), chunk_text, {"text": chunk_text, "url": row["url"], "product": row["product"], "title": title}


def filter_already_embedded_urls(client, df_to_embed):
    """Filter out URLs that have already been embedded in Qdrant."""
    df_to_embed = df_to_embed.copy()

    urls_in_qdrant = set()
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5000,
            with_payload=["url"],
            with_vectors=False,
            offset=next_offset,
        )

        for p in points:
            payload = p.payload or {}
            url = payload.get("url")
            if url:
                urls_in_qdrant.add(url)

        if next_offset is None:
            break

    print(f"Found urls_in_qdrant={len(urls_in_qdrant)}", flush=True)

    df_to_embed = df_to_embed[~df_to_embed["url"].isin(urls_in_qdrant)].copy()

    print(f"Remaining rows after url-skip={len(df_to_embed)}", flush=True)

    return df_to_embed


def upsert_dataframe(df_to_embed):
    """Embed and upsert DataFrame rows into Qdrant."""
    print("starting")
    client = QdrantClient(url=QDRANT_URL)
    print("done qdrant")
    ensure_collection(client)
    df_to_embed = filter_already_embedded_urls(client, df_to_embed)
    print(f"Start upsert rows={len(df_to_embed)} batch_size={BATCH_SIZE}")
    total_points = 0
    batch_i = 0

    print(f"Upserting into collection='{COLLECTION_NAME}' batch_size={BATCH_SIZE}")
    total_points = 0

    buffer_ids, buffer_texts, buffer_payloads = [], [], []

    def flush():
        nonlocal total_points, batch_i
        batch_i += 1
        n = len(buffer_texts)
        print(f"Batch {batch_i}: encoding n={n} total_points={total_points}")

        if not buffer_texts:
            return

        embeddings = model.encode(
            buffer_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
        )

        points = [
            qmodels.PointStruct(id=pid, vector=vec.tolist(), payload=payload)
            for pid, vec, payload in zip(buffer_ids, embeddings, buffer_payloads)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)

        total_points += n
        print(f"Batch {batch_i}: upserted n={n} total_points={total_points}")

        buffer_ids.clear()
        buffer_texts.clear()
        buffer_payloads.clear()

    for pid, text, payload in iter_records(df_to_embed):
        buffer_ids.append(pid)
        buffer_texts.append(text)
        buffer_payloads.append(payload)
        if len(buffer_texts) >= BATCH_SIZE:
            flush()

    flush()
    print(f"Done. total_points={total_points}")


def embed_text(df_to_embed):
    """Entry point for embedding a DataFrame directly.

    Args:
        df_to_embed: DataFrame with columns: url, title, text, product
    """
    upsert_dataframe(df_to_embed)


if __name__ == "__main__":
    df_to_embed = pd.read_csv(DATA_DIR / "text_to_embed.csv")
    embed_text(df_to_embed)
