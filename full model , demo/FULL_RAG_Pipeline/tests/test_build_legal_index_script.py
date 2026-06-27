from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_legal_index.py"


def _load_builder_module():
    spec = importlib.util.spec_from_file_location("build_legal_index", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(raw_path: Path, *, no_demo_fallback_fixtures: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        from_chunks_json=None,
        fixture=False,
        from_raw_json=str(raw_path),
        country=["UAE"],
        timeout=1,
        retries=0,
        write_intermediate=False,
        dry_run=True,
        raw_cache_path=str(raw_path),
        chunks_cache_path=str(raw_path.with_name("chunks.json")),
        no_demo_fallback_fixtures=no_demo_fallback_fixtures,
    )


def _write_raw_documents(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "country": "UAE",
                    "source_url": "https://example.test/uae",
                    "law_title": "UAE Unusable Source",
                    "source_language": "en",
                    "source_type": "html",
                    "official_source": True,
                    "raw_text": "Navigation footer archive contact",
                    "extraction_method": "test",
                    "extraction_status": "success",
                    "error_message": None,
                }
            ]
        ),
        encoding="utf-8",
    )


def test_builder_adds_demo_fallback_chunks_for_selected_country_with_zero_chunks(tmp_path):
    builder = _load_builder_module()
    raw_path = tmp_path / "raw.json"
    _write_raw_documents(raw_path)

    chunks = builder.load_or_build_chunks(_args(raw_path))

    assert chunks
    assert {chunk.country for chunk in chunks} == {"UAE"}
    assert all(chunk.source_url == "demo-fixture://legal-source/uae" for chunk in chunks)


def test_builder_can_fail_instead_of_using_demo_fallback(tmp_path):
    builder = _load_builder_module()
    raw_path = tmp_path / "raw.json"
    _write_raw_documents(raw_path)

    try:
        builder.load_or_build_chunks(_args(raw_path, no_demo_fallback_fixtures=True))
    except RuntimeError as exc:
        assert "No violence-related legal sections survived cleaning" in str(exc)
    else:
        raise AssertionError("Expected build to fail without demo fallback fixtures.")
