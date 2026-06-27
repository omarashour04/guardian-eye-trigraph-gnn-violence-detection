from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "reference_corpus"
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "faiss_reference.index"
METADATA_PATH = DATA_DIR / "reference_metadata.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

Document = Dict[str, Any]

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model once for the reference store."""

    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def load_documents() -> List[Document]:
    """Load reference corpus documents from text files."""

    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(f"Reference corpus directory not found: {REFERENCE_DIR}")

    documents: List[Document] = []
    for text_file in sorted(REFERENCE_DIR.glob("*.txt")):
        text = text_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            {
                "title": text_file.stem,
                "text": text,
                "embedding": None,
            }
        )

    if not documents:
        raise FileNotFoundError(f"No reference documents found in {REFERENCE_DIR}")

    return documents


def create_embeddings(documents: List[Document]) -> List[Document]:
    """Create embeddings for reference documents."""

    model = _get_model()
    texts = [document["text"] for document in documents]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    for document, embedding in zip(documents, embeddings):
        document["embedding"] = embedding

    return documents


def build_faiss_index(documents: List[Document]) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from document embeddings."""

    embeddings = np.array(
        [document["embedding"] for document in documents],
        dtype="float32",
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, documents: List[Document]) -> None:
    """Save the FAISS index and reference metadata to disk."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    metadata = [
        {
            "title": document["title"],
            "text": document["text"],
        }
        for document in documents
    ]
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_index() -> Tuple[faiss.IndexFlatIP, List[Document]]:
    """Load the FAISS index and reference metadata from disk."""

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        documents = create_embeddings(load_documents())
        index = build_faiss_index(documents)
        save_index(index, documents)
        return index, documents

    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    documents = [
        {
            "title": item["title"],
            "text": item["text"],
            "embedding": None,
        }
        for item in metadata
    ]
    return index, documents


def retrieve_reference(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve the most relevant Guardian Eye reference snippets for a query."""

    if not query.strip():
        return []

    index, documents = load_index()
    model = _get_model()
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results: List[Dict[str, Any]] = []
    for score, document_index in zip(scores[0], indices[0]):
        if document_index < 0:
            continue

        document = documents[document_index]
        results.append(
            {
                "title": document["title"],
                "snippet": document["text"][:500],
                "score": float(score),
            }
        )

    return results


def main() -> None:
    """Run a basic retrieval smoke test for Guardian Eye RAG Store #1."""

    retrieved = retrieve_reference(
        "fight involving a bottle"
    )

    for result in retrieved:
        print(f"{result['title']}: {result['score']:.4f}")


if __name__ == "__main__":
    main()
