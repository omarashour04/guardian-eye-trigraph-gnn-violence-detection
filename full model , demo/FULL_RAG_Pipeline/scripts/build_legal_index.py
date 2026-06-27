"""Build the Legal Consequences RAG FAISS index and SQLite metadata DB.

Real build:
    python scripts/build_legal_index.py

Offline smoke test:
    python scripts/build_legal_index.py --dry-run --fixture

Outputs for real builds:
    data/legal_faiss.index
    data/legal_metadata.db

This script does not load or download any LLM. It only extracts legal text,
filters/chunks it, embeds chunks with a sentence-transformers embedding model,
and saves FAISS + SQLite retrieval artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_service.legal_chunker import LegalChunk, chunk_cleaned_sections
from rag_service.legal_cleaner import clean_legal_documents
from rag_service.legal_index import build_legal_index
from rag_service.legal_scraper import RawLegalDocument, scrape_legal_sources
from rag_service.legal_sources import get_all_legal_sources


DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "legal_faiss.index"
DEFAULT_METADATA_DB_PATH = REPO_ROOT / "data" / "legal_metadata.db"
DEFAULT_RAW_CACHE_PATH = REPO_ROOT / "data" / "legal_raw_documents.json"
DEFAULT_CHUNKS_CACHE_PATH = REPO_ROOT / "data" / "legal_chunks.json"


class LocalSentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str, *, offline: bool) -> None:
        self.model_name = model_name
        self.offline = offline
        self._model: Any | None = None

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required to build the real legal index. "
                "Install FULL_RAG_Pipeline requirements first."
            ) from exc

        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=self.offline,
                )
            except TypeError:
                if self.offline:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                self._model = SentenceTransformer(self.model_name)

        passages = [f"passage: {text}" for text in texts]
        return self._model.encode(
            passages,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


class MockEmbeddingProvider:
    dimension = 768
    vocabulary = (
        "assault",
        "weapon",
        "bottle",
        "threat",
        "violence",
        "harm",
        "attack",
        "public order",
    )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vector = np.zeros(self.dimension, dtype=np.float32)
            for index, term in enumerate(self.vocabulary):
                vector[index] = float(lowered.count(term))
            for token in lowered.replace("\n", " ").split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vector[int.from_bytes(digest[:2], "big") % self.dimension] += 0.1
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Legal RAG FAISS and SQLite metadata artifacts."
    )
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--metadata-db-path", default=str(DEFAULT_METADATA_DB_PATH))
    parser.add_argument("--raw-cache-path", default=str(DEFAULT_RAW_CACHE_PATH))
    parser.add_argument("--chunks-cache-path", default=str(DEFAULT_CHUNKS_CACHE_PATH))
    parser.add_argument(
        "--from-raw-json",
        help="Use previously scraped raw legal documents JSON instead of fetching sources.",
    )
    parser.add_argument(
        "--from-chunks-json",
        help="Use previously chunked LegalChunk JSON instead of scraping/cleaning/chunking.",
    )
    parser.add_argument(
        "--country",
        action="append",
        help="Limit build to one country. Can be passed multiple times.",
    )
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("LEGAL_RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download embedding model files; use only local Hugging Face cache.",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Use small built-in fixture documents instead of network scraping.",
    )
    parser.add_argument(
        "--no-demo-fallback-fixtures",
        action="store_true",
        help=(
            "Do not add built-in demo legal-source fixtures when selected countries "
            "produce zero chunks from scraped/cached sources."
        ),
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic test embeddings. Intended only for dry runs/tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build into a temporary directory and remove outputs afterwards.",
    )
    parser.add_argument(
        "--write-intermediate",
        action="store_true",
        help="Write raw document and chunk JSON caches during real builds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        chunks = load_or_build_chunks(args)
        if not chunks:
            raise RuntimeError(
                "No legal chunks were produced. Check source fetch/extraction, legal keywords, "
                "or provide --from-chunks-json."
            )

        provider = (
            MockEmbeddingProvider()
            if args.mock_embeddings
            else LocalSentenceTransformerEmbeddingProvider(args.embedding_model, offline=args.offline)
        )

        legal_index = build_legal_index(chunks, embedding_provider=provider)
        index_path, metadata_path = output_paths(args)
        legal_index.save(index_path, metadata_path)

        print(f"Built Legal RAG index with {legal_index.index_count()} chunks.")
        print(f"FAISS index: {index_path}")
        print(f"Metadata DB: {metadata_path}")
        print_country_counts(
            "Chunks written to metadata DB",
            Counter(row.country for row in legal_index.metadata_rows),
            selected_country_names(args.country),
        )

        if args.dry_run:
            print("Dry run completed; outputs were written to a temporary directory.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def load_or_build_chunks(args: argparse.Namespace) -> list[LegalChunk | dict[str, Any]]:
    if args.from_chunks_json:
        chunks = read_json(Path(args.from_chunks_json))
        print(f"Loaded {len(chunks)} chunks from {args.from_chunks_json}.")
        print_country_counts(
            "Chunks loaded from JSON",
            Counter(to_dict(chunk).get("country", "unknown") for chunk in chunks),
            selected_country_names(args.country),
        )
        return chunks

    expected_countries = selected_country_names(args.country)
    print_country_counts(
        "Configured legal sources",
        source_counts_by_country(args.country),
        expected_countries,
    )

    if args.fixture:
        raw_documents = fixture_raw_documents(args.country)
        print(f"Loaded {len(raw_documents)} fixture raw legal documents.")
    elif args.from_raw_json:
        raw_documents = read_json(Path(args.from_raw_json))
        print(f"Loaded {len(raw_documents)} raw legal documents from {args.from_raw_json}.")
    else:
        sources = selected_sources(args.country)
        if not sources:
            raise RuntimeError("No legal sources selected from the registry.")
        print(f"Scraping {len(sources)} legal source(s).")
        raw_documents = scrape_legal_sources(
            sources,
            timeout=args.timeout,
            retries=args.retries,
        )

    print_country_counts(
        "Scraped raw documents",
        Counter(to_dict(document).get("country", "unknown") for document in raw_documents),
        expected_countries,
    )
    print_country_counts(
        "Successfully extracted raw documents",
        Counter(
            to_dict(document).get("country", "unknown")
            for document in raw_documents
            if to_dict(document).get("extraction_status") == "success"
        ),
        expected_countries,
    )

    successful_raw = [
        document
        for document in raw_documents
        if to_dict(document).get("extraction_status") == "success"
    ]
    failed_raw = len(raw_documents) - len(successful_raw)
    if failed_raw:
        print(f"Warning: {failed_raw} source(s) failed extraction.")
    if not successful_raw and (args.no_demo_fallback_fixtures or args.fixture):
        raise RuntimeError(
            "No legal source text was extracted. Use --from-raw-json, --fixture, "
            "or rerun with network access."
        )

    cleaned = clean_legal_documents(successful_raw)
    successful_cleaned = [
        section
        for section in cleaned
        if to_dict(section).get("cleaning_status") == "success"
    ]
    print_country_counts(
        "Cleaned legal sections",
        Counter(to_dict(section).get("country", "unknown") for section in successful_cleaned),
        expected_countries,
    )
    if not successful_cleaned and (args.no_demo_fallback_fixtures or args.fixture):
        raise RuntimeError(
            "No violence-related legal sections survived cleaning. Check legal keywords "
            "or source text quality."
        )

    chunks = chunk_cleaned_sections(successful_cleaned)
    print_country_counts(
        "Generated legal chunks before demo fallback",
        Counter(to_dict(chunk).get("country", "unknown") for chunk in chunks),
        expected_countries,
    )

    if not args.no_demo_fallback_fixtures and not args.fixture:
        missing_countries = countries_without_chunks(chunks, expected_countries)
        if missing_countries:
            print(
                "Adding demo legal-source fallback fixture(s) for country/countries with "
                f"zero generated chunks: {', '.join(missing_countries)}"
            )
            fixture_documents = demo_fallback_raw_documents(missing_countries)
            fixture_cleaned = [
                section
                for section in clean_legal_documents(fixture_documents)
                if to_dict(section).get("cleaning_status") == "success"
            ]
            fixture_chunks = chunk_cleaned_sections(fixture_cleaned)
            chunks.extend(fixture_chunks)
            print_country_counts(
                "Generated legal chunks after demo fallback",
                Counter(to_dict(chunk).get("country", "unknown") for chunk in chunks),
                expected_countries,
            )

    print(
        f"Pipeline counts: raw={len(raw_documents)}, cleaned={len(successful_cleaned)}, "
        f"chunks={len(chunks)}"
    )

    if args.write_intermediate and not args.dry_run:
        write_json(Path(args.raw_cache_path), [to_dict(document) for document in raw_documents])
        write_json(Path(args.chunks_cache_path), [to_dict(chunk) for chunk in chunks])

    return chunks


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.dry_run:
        return Path(args.index_path), Path(args.metadata_db_path)

    temp_dir = Path(tempfile.mkdtemp(prefix="legal-rag-index-"))
    return temp_dir / "legal_faiss.index", temp_dir / "legal_metadata.db"


def selected_sources(countries: list[str] | None) -> list[dict[str, Any]]:
    registry = get_all_legal_sources()
    if not countries:
        return [source for sources in registry.values() for source in sources]

    selected: list[dict[str, Any]] = []
    for country in countries:
        selected.extend(registry.get(country, []))
    return selected


def selected_country_names(countries: list[str] | None) -> list[str]:
    registry = get_all_legal_sources()
    if not countries:
        return list(registry.keys())
    return [country for country in countries if country in registry]


def source_counts_by_country(countries: list[str] | None) -> Counter[str]:
    registry = get_all_legal_sources()
    selected = selected_country_names(countries)
    return Counter({country: len(registry.get(country, [])) for country in selected})


def print_country_counts(
    label: str,
    counts: Counter[str],
    countries: list[str],
) -> None:
    print(f"{label} by country:")
    for country in countries:
        print(f"  - {country}: {counts.get(country, 0)}")


def countries_without_chunks(
    chunks: list[LegalChunk | dict[str, Any]],
    countries: list[str],
) -> list[str]:
    chunk_counts = Counter(to_dict(chunk).get("country", "unknown") for chunk in chunks)
    return [country for country in countries if chunk_counts.get(country, 0) == 0]


def fixture_raw_documents(countries: list[str] | None) -> list[RawLegalDocument]:
    selected = countries or ["UAE", "UK", "Canada"]
    documents: list[RawLegalDocument] = []
    for country in selected:
        documents.append(
            RawLegalDocument(
                country=country,
                source_url=f"https://example.test/{country.lower().replace(' ', '-')}",
                law_title=f"{country} Fixture Violence Law",
                source_language="en",
                source_type="html",
                official_source=False,
                raw_text=(
                    "Article 1 Assault\n"
                    "A person who commits assault or causes physical harm may be "
                    "subject to possible legal consequences depending on judicial "
                    "determination.\n\n"
                    "Article 2 Weapon\n"
                    "Using a weapon, bottle, or dangerous object during violent conduct "
                    "may be treated as an aggravating context."
                ),
                extraction_method="fixture",
                extraction_status="success",
                error_message=None,
            )
        )
    return documents


def demo_fallback_raw_documents(countries: list[str]) -> list[RawLegalDocument]:
    """Return explicit demo legal-source fixtures for unavailable source text.

    These fixtures are not official legal text. They are narrow demo corpus
    entries used only when the configured sources for a selected country produce
    no retrievable chunks, so the demo can still exercise jurisdiction filtering.
    """

    fixture_texts = {
        "UAE": (
            "Article 1 Demo legal-source fixture for UAE Crimes and Penalties Law\n"
            "This demo fixture is not official legal advice or a substitute for "
            "the official UAE legal text. It is included only when official UAE "
            "sources cannot be extracted for the local demo index.\n\n"
            "Article 2 Assault and physical harm\n"
            "Possible legal consequences may arise when an incident involves "
            "assault, attack, bodily harm, injury, threat, or violent conduct. "
            "Any outcome depends on competent authorities, court determination, "
            "and the facts established through lawful procedures.\n\n"
            "Article 3 Weapon or dangerous object context\n"
            "Using a weapon, bottle, knife, or other dangerous object during an "
            "alleged violent incident may be relevant to legal assessment and "
            "possible penalties under the selected jurisdiction."
        ),
        "KSA": (
            "Article 1 Demo legal-source fixture for KSA Protection from Abuse Law\n"
            "This demo fixture is not official legal advice or a substitute for "
            "the official Saudi legal text. It is included only when official KSA "
            "sources cannot be extracted for the local demo index.\n\n"
            "Article 2 Abuse, assault, and physical harm\n"
            "Possible legal consequences may arise when an incident involves "
            "abuse, assault, attack, physical harm, injury, threat, intimidation, "
            "or violent conduct. Any outcome depends on competent authorities, "
            "court determination, and the facts established through lawful "
            "procedures.\n\n"
            "Article 3 Protective and aggravating context\n"
            "Use of a weapon, bottle, knife, or dangerous object, or violence in "
            "a family or protection context, may be relevant to legal assessment "
            "and possible protective measures or penalties."
        ),
    }

    documents: list[RawLegalDocument] = []
    for country in countries:
        text = fixture_texts.get(country)
        if not text:
            continue
        documents.append(
            RawLegalDocument(
                country=country,
                source_url=f"demo-fixture://legal-source/{country.lower()}",
                law_title=f"{country} Demo Legal Source Fixture",
                source_language="en",
                source_type="html",
                official_source=False,
                raw_text=text,
                extraction_method="demo_fixture",
                extraction_status="success",
                error_message=None,
            )
        )
    return documents


def to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Cannot serialize value of type {type(value)!r}")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise RuntimeError(f"Input JSON does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
