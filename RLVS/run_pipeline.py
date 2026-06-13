"""
run_pipeline.py
Guardian Eye — V9  |  RLVS end-to-end orchestrator

Runs the three phases in order, stopping immediately if any phase fails:
  1. trigraph_v9_rlvs_preprocess.py   (Phase 1 — preprocessing)
  2. trigraph_v9_rlvs_videomae.py     (Phase 2 — VideoMAE fine-tune + embeddings)
  3. trigraph_v9_rlvs_train.py        (Phase 3 — graph training + ablation)

Each phase runs as its own subprocess, so GPU memory is fully released between
phases (important — VideoMAE holds an 86M-param model you don't want lingering
when graph training starts). All three phases are individually resumable, so if
the pipeline is interrupted at any point, just re-run this script: finished
phases are skipped quickly and the interrupted phase continues where it stopped.

Usage
─────
  python run_pipeline.py
  python run_pipeline.py --from videomae        # start at phase 2 (skip preprocess)
  python run_pipeline.py --from train           # start at phase 3 only
  python run_pipeline.py --skip-preprocess      # alias for --from videomae

Notes
─────
  - Uses the SAME Python interpreter that runs this script (sys.executable),
    so activate your venv/conda env first, then run this.
  - Press Ctrl+C once to stop cleanly after the current phase's subprocess
    receives the interrupt; re-run later to resume.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (phase key, script filename, human label)
PHASES = [
    ("preprocess", "trigraph_v9_rlvs_preprocess.py", "Phase 1 — Preprocessing"),
    ("videomae",   "trigraph_v9_rlvs_videomae.py",   "Phase 2 — VideoMAE"),
    ("train",      "trigraph_v9_rlvs_train.py",       "Phase 3 — Training"),
]


def run_phase(script: str, label: str) -> int:
    """Run one phase as a subprocess. Returns its exit code."""
    script_path = HERE / script
    if not script_path.exists():
        print(f"\n[ERROR] Script not found: {script_path}")
        return 1

    print(f"\n{'#'*70}")
    print(f"# {label}")
    print(f"# {sys.executable} {script}")
    print(f"{'#'*70}\n", flush=True)

    t0 = time.time()
    # No capture: child stdout/stderr stream straight to this console so you see
    # tqdm bars and logs live. Same interpreter as the orchestrator.
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(HERE))
    dt = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[{label}] {status}  —  {dt/60:.1f} min", flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="RLVS V9 pipeline orchestrator")
    parser.add_argument(
        "--from", dest="start_from", choices=[p[0] for p in PHASES],
        default="preprocess",
        help="Phase to start from (default: preprocess). Earlier phases skipped.")
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Alias for --from videomae")
    args = parser.parse_args()

    start_key = "videomae" if args.skip_preprocess else args.start_from
    start_idx = next(i for i, p in enumerate(PHASES) if p[0] == start_key)

    print(f"{'='*70}")
    print("Guardian Eye V9 — RLVS pipeline")
    print(f"  Interpreter : {sys.executable}")
    print(f"  Starting at : {PHASES[start_idx][2]}")
    print(f"  Phases      : {' -> '.join(p[0] for p in PHASES[start_idx:])}")
    print(f"{'='*70}", flush=True)

    pipeline_t0 = time.time()
    for _, script, label in PHASES[start_idx:]:
        code = run_phase(script, label)
        if code != 0:
            print(f"\n{'='*70}")
            print(f"Pipeline STOPPED — {label} failed.")
            print("Fix the issue, then re-run this script: finished phases are "
                  "skipped and the interrupted phase resumes.")
            print(f"{'='*70}")
            return code

    print(f"\n{'='*70}")
    print(f"Pipeline COMPLETE — all phases finished in "
          f"{(time.time() - pipeline_t0)/60:.1f} min.")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
