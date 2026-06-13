"""
run_pipeline.py
Guardian Eye - V9  |  Hockey Fights end-to-end orchestrator

Runs the phases in order, stopping immediately if any phase fails:
  1. trigraph_v9_hf_preprocess.py    (Phase 1 - folder scan + preprocessing)
  2. trigraph_v9_hf_videomae.py      (Phase 2 - VideoMAE fine-tune + embeddings)
  3. trigraph_v9_hf_train.py         (Phase 3 - graph training + ablation)
  4. trigraph_v9_hf_diagnostics.py   (Phase 4 - post-hoc diagnostics) [OPT-IN]

Each phase runs as its own subprocess, so GPU memory is fully released between
phases (important - VideoMAE holds an 86M-param model you don't want lingering
when graph training starts). All phases are individually resumable, so if the
pipeline is interrupted at any point, just re-run this script: finished phases
are skipped quickly and the interrupted phase continues where it stopped.

Phase 4 (diagnostics) is NOT run by default - it only makes sense after the graph
experiments have produced checkpoints, and you usually want to inspect Phase 3
results first. Add --with-diagnostics to include it, or run it standalone:
  python trigraph_v9_hf_diagnostics.py

Usage
-----
  python run_pipeline.py                         # phases 1 -> 2 -> 3
  python run_pipeline.py --from videomae         # start at phase 2 (skip preprocess)
  python run_pipeline.py --from train            # start at phase 3 only
  python run_pipeline.py --skip-preprocess       # alias for --from videomae
  python run_pipeline.py --with-diagnostics      # also run phase 4 after training
  python run_pipeline.py --from diagnostics      # run phase 4 only

Notes
-----
  - Uses the SAME Python interpreter that runs this script (sys.executable),
    so activate your venv/conda env first, then run this.
  - Press Ctrl+C once to stop cleanly; re-run later to resume.
  - Hockey Fights is a clip-level dataset (no windowing), so there is no
    --plan-only step like NTU has. Phase 1 scans fights/ and nofights/ directly.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (phase key, script filename, human label)
PHASES = [
    ("preprocess",  "trigraph_v9_hf_preprocess.py",  "Phase 1 - Preprocessing"),
    ("videomae",    "trigraph_v9_hf_videomae.py",    "Phase 2 - VideoMAE"),
    ("train",       "trigraph_v9_hf_train.py",        "Phase 3 - Training"),
    ("diagnostics", "trigraph_v9_hf_diagnostics.py",  "Phase 4 - Diagnostics"),
]


def run_phase(script: str, label: str, extra_args=None) -> int:
    """Run one phase as a subprocess. Returns its exit code."""
    script_path = HERE / script
    if not script_path.exists():
        print(f"\n[ERROR] Script not found: {script_path}")
        return 1

    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    print(f"\n{'#'*70}")
    print(f"# {label}")
    print(f"# {' '.join(cmd)}")
    print(f"{'#'*70}\n", flush=True)

    t0 = time.time()
    # No capture: child stdout/stderr stream straight to this console so you see
    # tqdm bars and logs live. Same interpreter as the orchestrator.
    result = subprocess.run(cmd, cwd=str(HERE))
    dt = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[{label}] {status}  -  {dt/60:.1f} min", flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Hockey Fights V9 pipeline orchestrator")
    parser.add_argument(
        "--from", dest="start_from", choices=[p[0] for p in PHASES],
        default="preprocess",
        help="Phase to start from (default: preprocess). Earlier phases skipped.")
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Alias for --from videomae")
    parser.add_argument(
        "--with-diagnostics", action="store_true",
        help="Also run Phase 4 (diagnostics) after training")
    args = parser.parse_args()

    start_key = "videomae" if args.skip_preprocess else args.start_from
    start_idx = next(i for i, p in enumerate(PHASES) if p[0] == start_key)

    # Build the phase list. Diagnostics (index 3) is included only if explicitly
    # requested (--with-diagnostics) OR if the user starts from it (--from diagnostics).
    selected = list(PHASES[start_idx:])
    if not args.with_diagnostics and start_key != "diagnostics":
        selected = [p for p in selected if p[0] != "diagnostics"]

    print(f"{'='*70}")
    print("Guardian Eye V9 - Hockey Fights pipeline")
    print(f"  Interpreter : {sys.executable}")
    print(f"  Starting at : {selected[0][2] if selected else '(nothing to run)'}")
    print(f"  Phases      : {' -> '.join(p[0] for p in selected)}")
    print(f"{'='*70}", flush=True)

    pipeline_t0 = time.time()
    for _, script, label in selected:
        code = run_phase(script, label)
        if code != 0:
            print(f"\n{'='*70}")
            print(f"Pipeline STOPPED - {label} failed.")
            print("Fix the issue, then re-run this script: finished phases are "
                  "skipped and the interrupted phase resumes.")
            print(f"{'='*70}")
            return code

    print(f"\n{'='*70}")
    print(f"Pipeline COMPLETE - all phases finished in "
          f"{(time.time() - pipeline_t0)/60:.1f} min.")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
