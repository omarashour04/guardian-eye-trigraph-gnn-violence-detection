"""
smoke_videomae.py — isolate the VideoMAE first-forward hang in ONE command.

Runs a sequence of single-forward probes on VideoMAE-base with a dummy batch,
each in its OWN subprocess (so a hang in one doesn't block the next) with a
hard timeout. Whichever probe hangs/passes pinpoints the cause automatically.

Run on the WS (single command):
  python smoke_videomae.py

It runs these probes in order and prints a verdict:
  1. CPU forward+backward        -> is the MODEL itself ok?
  2. GPU, checkpointing OFF      -> is gradient checkpointing the deadlock?
  3. GPU, eval forward only      -> is it the train/autograd path?
  4. GPU, checkpointing ON       -> the real pipeline path

Each probe has a 180s timeout. A probe that times out = that configuration hangs.

(Internal: re-invokes itself with --probe to run one probe in a child process.)
"""
import argparse, subprocess, sys, time
import torch


def log(msg):
    print(f"  [smoke] {msg}", flush=True)


def run_one_probe(mode: str, bs: int):
    """Child process: run a single probe. mode in {cpu,no_ckpt,eval,ckpt}."""
    dev = torch.device("cpu" if mode == "cpu" else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"PROBE mode={mode} bs={bs}  torch={torch.__version__} device={dev}")
    if dev.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)} "
            f"cap={torch.cuda.get_device_capability(0)} "
            f"cudnn={torch.backends.cudnn.version()}")

    from transformers import VideoMAEModel
    log("loading videomae-base ...")
    base = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

    if mode == "ckpt":
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            log("checkpointing ON (non-reentrant)")
        except TypeError:
            base.gradient_checkpointing_enable()
            log("checkpointing ON (reentrant)")
    else:
        log("checkpointing OFF")

    base = base.to(dev)
    is_eval = (mode == "eval")
    base.train(not is_eval)
    log(f"mode={'eval' if is_eval else 'train'}  building dummy batch ...")

    x = torch.randn(bs, 16, 3, 224, 224, device=dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    log("FORWARD ...")
    t0 = time.time()
    ctx = torch.no_grad() if is_eval else torch.enable_grad()
    with ctx:
        out = base(pixel_values=x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    log(f"FORWARD ok {time.time()-t0:.1f}s  hidden={tuple(out.last_hidden_state.shape)}")

    if not is_eval:
        log("BACKWARD ...")
        t0 = time.time()
        out.last_hidden_state.mean().backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        log(f"BACKWARD ok {time.time()-t0:.1f}s")
    log("PROBE PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=["cpu", "no_ckpt", "eval", "ckpt"])
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    # Child mode: run exactly one probe (this is what the timeout wraps).
    if args.probe:
        run_one_probe(args.probe, args.bs)
        return

    # Parent mode: run each probe in its own subprocess with a timeout.
    PROBES = [
        ("cpu",     2,   "1. CPU forward+backward (is the model ok?)"),
        ("no_ckpt", 8,   "2. GPU, checkpointing OFF (is checkpointing the hang?)"),
        ("eval",    8,   "3. GPU, eval forward only (train-path vs CUDA?)"),
        ("ckpt",    8,   "4. GPU, checkpointing ON (real pipeline path)"),
    ]
    TIMEOUT = 180
    results = {}
    print("=" * 70)
    print("VideoMAE smoke test — running 4 probes, 180s timeout each")
    print("=" * 70, flush=True)

    for mode, bs, label in PROBES:
        print(f"\n{'-'*70}\n{label}\n{'-'*70}", flush=True)
        cmd = [sys.executable, __file__, "--probe", mode, "--bs", str(bs)]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, timeout=TIMEOUT)
            ok = (r.returncode == 0)
            results[mode] = "PASS" if ok else f"FAIL(exit {r.returncode})"
        except subprocess.TimeoutExpired:
            results[mode] = "HANG(timeout)"
            print(f"  [smoke] >>> {mode} HUNG (>{TIMEOUT}s) — killed.", flush=True)
        print(f"  [smoke] {mode}: {results[mode]}  ({time.time()-t0:.0f}s)", flush=True)

    # Verdict
    print(f"\n{'='*70}\nVERDICT\n{'='*70}")
    for mode, _, label in PROBES:
        print(f"  {results.get(mode,'?'):<14} {label}")
    print()
    cpu, no_ckpt, ev, ck = (results.get(k) for k in ("cpu", "no_ckpt", "eval", "ckpt"))
    if cpu != "PASS":
        print("  -> Hangs on CPU too: the MODEL/transformers/torch is the problem,")
        print("     not the GPU. Likely a bad transformers/torch version. Reinstall.")
    elif no_ckpt == "PASS" and ck != "PASS":
        print("  -> GPU works WITHOUT checkpointing but HANGS with it:")
        print("     gradient checkpointing is the deadlock. Set grad_checkpointing=False")
        print("     in trigraph_v9_ntu_videomae.py (use batch_size=8 to fit VRAM).")
    elif no_ckpt != "PASS":
        print("  -> GPU forward hangs even without checkpointing:")
        print("     CUDA/WDDM contention with the other GPU job, or a driver/CUDA issue.")
        print("     Fix: run NTU on the GPU EXCLUSIVELY (let the other job finish first),")
        print("     or update the GPU driver / reinstall a torch build matching the driver.")
    elif ev == "PASS" and ck != "PASS":
        print("  -> eval forward works, train/checkpoint path hangs: checkpointing path.")
        print("     Set grad_checkpointing=False and retry.")
    else:
        print("  -> All GPU probes passed. The hang may be input-specific")
        print("     (mixup/cutmix or a bad NPZ batch). Re-run the real script;")
        print("     if it still hangs, we instrument mixup next.")
    print("=" * 70)


if __name__ == "__main__":
    main()
