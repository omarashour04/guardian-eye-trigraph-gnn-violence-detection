from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import List, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def check_import(module_name: str) -> Tuple[str, bool]:
    """Return whether a module imports successfully."""

    try:
        importlib.import_module(module_name)
    except Exception:
        return f"Import {module_name}", False
    return f"Import {module_name}", True


def check_file(relative_path: str) -> Tuple[str, bool]:
    """Return whether a required project file exists."""

    return f"Exists {relative_path}", (BASE_DIR / relative_path).exists()


def print_result(label: str, passed: bool) -> None:
    """Print one setup check result."""

    status = "PASS" if passed else "FAIL"
    print(f"{status}: {label}")


def main() -> None:
    """Run Day 1 setup checks for the Guardian Eye RAG project."""

    checks: List[Tuple[str, bool]] = [
        check_import("rag.schemas"),
        check_import("g3_incident_db"),
        check_import("g2_reference_store"),
        check_file("reference_corpus/violence_taxonomy.txt"),
        check_file("reference_corpus/non_violence_activities.txt"),
        check_file("reference_corpus/evaluation_caveats.txt"),
        check_file("reference_corpus/example_explanations.txt"),
    ]

    for label, passed in checks:
        print_result(label, passed)

    if all(passed for _, passed in checks):
        print("DAY 1 SETUP COMPLETE")


if __name__ == "__main__":
    main()
