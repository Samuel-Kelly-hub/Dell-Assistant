import re

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from sentence_transformers import SentenceTransformer

from chatbot_backend.config import (
    EMBEDDING_MODEL_NAME,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    RAG_TOP_K,
)

_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_client = QdrantClient(url=QDRANT_URL)


def _normalise_product_name(name: str) -> str:
    """Lowercase and hyphenate a product name to match Qdrant payload slugs."""
    normalised = name.lower().replace(" ", "-")
    normalised = re.sub(r"[^a-z0-9-]", "", normalised)
    normalised = re.sub(r"-+", "-", normalised).strip("-")
    return normalised


def _format_results(results: list[dict]) -> str:
    """Turn a list of result dicts into a numbered, human-readable string."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] {r['title']}\n"
            f"Source: {r['url']}\n\n"
            f"{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


def rag_search(product_name: str, query: str) -> tuple[str, list[str]]:
    """Search the Dell technical support knowledge base.

    Encodes the query with a sentence-transformer model, queries the Qdrant
    vector database, and returns the top-K matching chunks as a formatted
    string alongside a list of source URLs.

    Args:
        product_name: The Dell product name to search for.
        query: The search query describing the technical issue.

    Returns:
        A tuple of (formatted context string, list of source URLs).
        Returns ("", []) if no results found.
    """
    query_vec = _model.encode([query], normalize_embeddings=True)[0].tolist()

    query_filter = None
    if product_name:
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="product",
                    match=qmodels.MatchValue(
                        value=_normalise_product_name(product_name)
                    ),
                )
            ]
        )

    res = _client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vec,
        limit=RAG_TOP_K,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )

    results = [
        {
            "score": p.score,
            "text": (p.payload or {}).get("text", ""),
            "url": (p.payload or {}).get("url", ""),
            "title": (p.payload or {}).get("title", ""),
            "product": (p.payload or {}).get("product", ""),
        }
        for p in res.points
    ]

    if not results:
        return "", []

    urls = [r["url"] for r in results]
    return _format_results(results), urls
