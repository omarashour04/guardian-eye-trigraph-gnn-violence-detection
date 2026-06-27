from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def inspect_npz_vit(npz_path: str | Path) -> dict[str, Any]:
    path = Path(npz_path)
    result: dict[str, Any] = {
        "npz_path": str(path),
        "exists": path.exists(),
        "vit_embedding_exists": False,
        "shape": None,
        "mean": None,
        "std": None,
        "all_zero": None,
        "active": False,
    }
    if not path.exists():
        result["reason"] = "npz_missing"
        return result

    with np.load(str(path), allow_pickle=False) as npz:
        if "vit_embedding" not in npz.files:
            result["reason"] = "vit_embedding_missing"
            return result
        emb = np.asarray(npz["vit_embedding"], dtype=np.float32)

    all_zero = bool(emb.size == 0 or np.allclose(emb, 0.0))
    result.update(
        {
            "vit_embedding_exists": True,
            "shape": list(emb.shape),
            "mean": float(emb.mean()) if emb.size else 0.0,
            "std": float(emb.std()) if emb.size else 0.0,
            "all_zero": all_zero,
            "active": not all_zero,
            "reason": "active_nonzero_embedding" if not all_zero else "zero_filled_embedding",
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Guardian Eye NPZ vit_embedding activity.",
    )
    parser.add_argument("npz", help="Path to a cached .npz file")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    args = parser.parse_args()
    result = inspect_npz_vit(args.npz)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result.get("exists") else 2


if __name__ == "__main__":
    raise SystemExit(main())
