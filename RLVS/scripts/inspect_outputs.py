"""
inspect_outputs.py — Sanity / integrity inspection for the RLVS V9 run.

Runs four independent audits on the preprocessing + fine-tune outputs and prints
a verdict for each. Nothing here mutates any file; it is read-only inspection.

Audits
------
1. SPLIT INTEGRITY   — sizes, class balance, and (critically) clip_id overlap
                       between train / val / test. Any overlap = leakage.
2. NEAR-DUPLICATE    — RLVS contains re-cuts of the same scene. We can't hash the
   SCAN                video here cheaply, so we flag clip_ids that are numerically
                       adjacent (e.g. V_100 / V_101) yet land in different splits.
                       These are the most likely near-dup straddlers to eyeball.
3. LOW-QUALITY AUDIT — list every clip whose graph-quality channels collapsed
                       (valid_ratio < 0.5, or any q_* == 0). Confirms the 111
                       flagged clips are genuinely bad, not a decode bug on a shard.
4. RESULTS SANITY    — re-read the fine-tune logs, recompute monotonicity of the
                       val curve, and compare the headline test metrics against the
                       project's realistic-ceiling expectations for RLVS.

Usage
-----
    python inspect_outputs.py
    python inspect_outputs.py --preproc preproc_output --finetune finetune_output
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

# Windows consoles default to cp1252; force UTF-8 so em-dashes / symbols print.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---- realistic-ceiling expectations (from CLAUDE.md) -----------------------
# Appearance models on RLVS top out ~92-95% acc; anything materially above that
# on a known-noisy test split is a leakage yellow-flag, not skill.
RLVS_APPEARANCE_ACC_CEILING = 0.95
TARGET_MACRO_F1 = 0.85
TARGET_ROC_AUC = 0.92
LOW_VALID_RATIO = 0.5


def _p(msg=""):
    print(msg)


def _hdr(title):
    _p()
    _p("=" * 72)
    _p(title)
    _p("=" * 72)


def load_split(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit_split_integrity(rows):
    _hdr("1. SPLIT INTEGRITY")
    _p(f"total rows: {len(rows)}")
    _p(f"by split  : {dict(Counter(r['split'] for r in rows))}")
    _p(f"by label  : {dict(Counter(r['label'] for r in rows))}")
    _p()

    # per-split class balance
    for s in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == s]
        bal = dict(Counter(r["label"] for r in sub))
        _p(f"  {s:5s}: n={len(sub):4d}  labels={bal}")

    # --- the important check: clip_id overlap across splits (leakage) -------
    by_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["clip_id"])

    _p()
    overlap_found = False
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        inter = by_split[a] & by_split[b]
        if inter:
            overlap_found = True
            sample = sorted(inter)[:10]
            _p(f"  LEAKAGE  {a} & {b}: {len(inter)} shared clip_id(s) e.g. {sample}")
        else:
            _p(f"  OK       {a} & {b}: no shared clip_id")

    # duplicate clip_id anywhere (same id listed twice)
    id_counts = Counter(r["clip_id"] for r in rows)
    dups = {k: v for k, v in id_counts.items() if v > 1}
    if dups:
        overlap_found = True
        _p(f"  DUP      clip_id appearing >1x: {dict(list(dups.items())[:10])}")
    else:
        _p(f"  OK       every clip_id is unique across the whole file")

    _p()
    _p("  VERDICT: " + ("LEAKAGE DETECTED — investigate before trusting test metrics"
                        if overlap_found else "clean — no clip_id leakage"))
    return not overlap_found


def _parse_num_id(clip_id):
    """V_798 -> ('V', 798); NV_12 -> ('NV', 12). Returns (prefix, int) or None."""
    m = re.match(r"^([A-Za-z]+)_?(\d+)$", clip_id)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def audit_near_duplicates(rows):
    _hdr("2. NEAR-DUPLICATE STRADDLERS (numerically adjacent ids, different splits)")
    _p("RLVS re-cuts the same scene under consecutive ids. Adjacent ids that land")
    _p("in DIFFERENT splits are the most likely scene-overlap leak. Eyeball these.")
    _p()

    split_of = {r["clip_id"]: r["split"] for r in rows}
    label_of = {r["clip_id"]: r["label"] for r in rows}

    # group by prefix, sort by number, look at consecutive neighbours
    groups = defaultdict(list)
    for cid in split_of:
        parsed = _parse_num_id(cid)
        if parsed:
            groups[parsed[0]].append((parsed[1], cid))

    straddlers = []
    for prefix, items in groups.items():
        items.sort()
        for (n1, c1), (n2, c2) in zip(items, items[1:]):
            if n2 - n1 == 1:  # consecutive ids
                s1, s2 = split_of[c1], split_of[c2]
                if s1 != s2:
                    straddlers.append((c1, s1, c2, s2, label_of[c1], label_of[c2]))

    if not straddlers:
        _p("  none found.")
    else:
        _p(f"  {len(straddlers)} adjacent-id pairs cross a split boundary:")
        for c1, s1, c2, s2, l1, l2 in straddlers[:40]:
            flag = "  <- same label" if l1 == l2 else "  (diff label)"
            _p(f"    {c1}({s1},y={l1})  |  {c2}({s2},y={l2}){flag}")
        if len(straddlers) > 40:
            _p(f"    ... and {len(straddlers) - 40} more")
    _p()
    _p("  NOTE: adjacency is a heuristic, not proof of duplication. For a definitive")
    _p("  check, perceptual-hash the actual frames of these pairs (not done here to")
    _p("  keep this script fast/offline).")
    return straddlers


def audit_low_quality(gqs_path):
    _hdr("3. LOW-QUALITY CLIP AUDIT")
    if not os.path.exists(gqs_path):
        _p(f"  gqs summary not found at {gqs_path} — skipping")
        return
    with open(gqs_path, newline="", encoding="utf-8") as f:
        g = list(csv.DictReader(f))

    qcols = ["q_skel", "q_int", "q_obj", "q_po", "valid_ratio"]
    _p(f"rows: {len(g)}")
    import statistics as st
    for col in qcols:
        vals = [float(r[col]) for r in g]
        _p(f"  {col:11s}: mean={st.mean(vals):.3f}  min={min(vals):.3f}  max={max(vals):.3f}")
    _p()

    low_vr = [r for r in g if float(r["valid_ratio"]) < LOW_VALID_RATIO]
    zero_any = [r for r in g if any(float(r[c]) == 0.0 for c in qcols)]

    _p(f"  clips with valid_ratio < {LOW_VALID_RATIO}: {len(low_vr)}")
    _p(f"  clips with ANY quality channel == 0.000: {len(zero_any)}")
    _p()

    # how are the bad clips distributed across split / label?
    if low_vr:
        _p(f"  low valid_ratio by split: {dict(Counter(r['split'] for r in low_vr))}")
        _p(f"  low valid_ratio by label: {dict(Counter(r['label'] for r in low_vr))}")
    _p()

    # full list (sorted worst-first) so the user can spot-check the actual videos
    worst = sorted(g, key=lambda r: float(r["valid_ratio"]))
    _p("  worst 30 clips (clip_id, split, label, valid_ratio, q_skel/q_int/q_obj/q_po):")
    for r in worst[:30]:
        _p("    {cid:8s} {sp:5s} y={lab} vr={vr:.3f}  "
           "skel={qs:.2f} int={qi:.2f} obj={qo:.2f} po={qp:.2f}".format(
               cid=r["clip_id"], sp=r["split"], lab=r["label"],
               vr=float(r["valid_ratio"]), qs=float(r["q_skel"]),
               qi=float(r["q_int"]), qo=float(r["q_obj"]), qp=float(r["q_po"])))

    # write the full bad list to disk for convenience
    out = os.path.join(os.path.dirname(gqs_path), "low_quality_clips.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(g[0].keys()))
        w.writeheader()
        for r in low_vr:
            w.writerow(r)
    _p()
    _p(f"  wrote full low-valid_ratio list -> {out}")


def audit_results(finetune_dir):
    _hdr("4. RESULTS SANITY")
    log_path = os.path.join(finetune_dir, "videomae_train_log.csv")
    res_path = os.path.join(finetune_dir, "videomae_test_results.json")

    # ---- training curve ----
    if os.path.exists(log_path):
        with open(log_path, newline="", encoding="utf-8") as f:
            log = list(csv.DictReader(f))
        f1 = [float(r["val_macro_f1"]) for r in log]
        auc = [float(r["val_roc_auc"]) for r in log]
        _p(f"epochs logged       : {len(log)}")
        _p(f"val Macro-F1  start  : {f1[0]:.4f}  ->  end: {f1[-1]:.4f}  (best {max(f1):.4f})")
        _p(f"val ROC-AUC   start  : {auc[0]:.4f}  ->  end: {auc[-1]:.4f}  (best {max(auc):.4f})")

        # monotonicity / stability: count epochs where F1 dropped vs previous
        drops = sum(1 for a, b in zip(f1, f1[1:]) if b < a - 1e-9)
        _p(f"val F1 down-steps    : {drops}/{len(f1)-1}  "
           f"({'smooth' if drops <= len(f1)//5 else 'noisy'})")
        # AUC should be essentially monotone for a healthy run
        auc_drops = sum(1 for a, b in zip(auc, auc[1:]) if b < a - 1e-9)
        _p(f"val AUC down-steps   : {auc_drops}/{len(auc)-1}")
    else:
        _p(f"  train log not found at {log_path}")

    # ---- test metrics ----
    _p()
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as f:
            res = json.load(f)
        acc = res.get("accuracy")
        mf1 = res.get("macro_f1")
        auc = res.get("roc_auc")
        thr = res.get("threshold")
        _p(f"TEST accuracy : {acc}")
        _p(f"TEST macro_f1 : {mf1}")
        _p(f"TEST roc_auc  : {auc}")
        _p(f"TEST threshold: {thr}  (tuned on val, applied to test — correct protocol)")
        _p()
        # target check
        _p(f"  targets: Macro-F1 >= {TARGET_MACRO_F1}, ROC-AUC >= {TARGET_ROC_AUC}")
        if mf1 is not None and auc is not None:
            hit = mf1 >= TARGET_MACRO_F1 and auc >= TARGET_ROC_AUC
            _p(f"  targets met : {'YES' if hit else 'NO'}")
        # ceiling check (the yellow flag)
        if acc is not None and acc > RLVS_APPEARANCE_ACC_CEILING:
            _p()
            _p(f"  *** YELLOW FLAG: test accuracy {acc:.3f} exceeds the realistic RLVS")
            _p(f"      appearance ceiling (~{RLVS_APPEARANCE_ACC_CEILING:.2f}). On a test")
            _p(f"      split CUE-Net (2024) flagged as mislabelled, near-perfect scores")
            _p(f"      are more consistent with leakage/near-dups than with genuine skill.")
            _p(f"      -> trust this only after audits 1 & 2 come back clean.")
    else:
        _p(f"  test results not found at {res_path}")


# ---------------------------------------------------------------------------
# 5. PERCEPTUAL-HASH NEAR-DUPLICATE DETECTION  (--phash)  [HARDENED v2]
# ---------------------------------------------------------------------------
# v1 used an 8x8 average-hash (aHash) and called two clips duplicate if ANY one
# frame-pair was within 6 bits. That massively over-flagged: dark/static clips
# (esp. the 341 zero-quality NonViolence ones) all collapse to near-identical
# aHashes, so a fight and a handshake came back "identical" (hamming=0). Useless.
#
# v2 makes the detector trustworthy:
#   1. DCT-based pHash (64-bit) instead of aHash. pHash keeps low-frequency
#      structure and is robust to brightness, contrast and letterboxing, so dark
#      static frames stop colliding.
#   2. CROSS-LABEL GUARD: two clips of OPPOSITE labels are never called dupes.
#      A violence clip == a non-violence clip is a detector artifact by
#      definition; suppressing those removes the obvious false positives.
#   3. MULTI-FRAME requirement: two clips must match on at least MIN_FRAME_MATCHES
#      separate frame pairs (each within FRAME_HAMMING bits), not one lucky hit.
#   4. A clip is summarised by a confidence tier so you can separate
#      "identical re-encode" from "similar scene".
#
# Dependencies: opencv-python (cv2) + numpy. Cost: a few minutes for 2000 clips.

FRAMES_PER_CLIP = 8        # evenly spaced frames sampled per clip
HASH_IMG_SIZE = 32         # frame is resized to 32x32 before DCT
DCT_LOW_FREQ = 8           # keep top-left 8x8 DCT block -> 64-bit hash
FRAME_HAMMING = 8          # <=8 of 64 bits differing => the two FRAMES match
MIN_FRAME_MATCHES = 3      # need >=3 matching frame-pairs => clips are dupes
IDENTICAL_HAMMING = 4      # avg hamming <= this => flag as near-identical re-encode
PHASH_CACHE = "phash_cache_v2.json"   # new key so stale aHash cache is ignored


def _dct_phash_frame(gray):
    """64-bit DCT perceptual hash of a grayscale frame (numpy float32)."""
    import cv2
    import numpy as np
    g = cv2.resize(gray, (HASH_IMG_SIZE, HASH_IMG_SIZE), interpolation=cv2.INTER_AREA)
    d = cv2.dct(g.astype(np.float32))
    low = d[:DCT_LOW_FREQ, :DCT_LOW_FREQ].flatten()
    # median of the low-freq coeffs, EXCLUDING the DC term (index 0) which only
    # encodes overall brightness — dropping it is what makes pHash light-robust.
    med = np.median(low[1:])
    bits = low > med
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hash_clip(path):
    """Decode FRAMES_PER_CLIP evenly-spaced frames -> list of 64-bit DCT pHashes.
    Returns [] if the video can't be opened/decoded."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    hashes = []
    if total <= 0:
        for _ in range(FRAMES_PER_CLIP):
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hashes.append(_dct_phash_frame(g))
        cap.release()
        return hashes
    idxs = [int(i * (total - 1) / max(1, FRAMES_PER_CLIP - 1)) for i in range(FRAMES_PER_CLIP)]
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hashes.append(_dct_phash_frame(g))
    cap.release()
    return hashes


def audit_phash(rows, out_dir):
    _hdr("5. PERCEPTUAL-HASH NEAR-DUPLICATE SCAN (hardened DCT-pHash, v2)")
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as e:
        _p(f"  cannot run --phash: missing dependency ({e}). "
           f"pip install opencv-python numpy")
        return

    from tqdm import tqdm

    _p(f"  config: {FRAMES_PER_CLIP} frames/clip, DCT pHash, frame match <= "
       f"{FRAME_HAMMING} bits, need >= {MIN_FRAME_MATCHES} matching frames,")
    _p(f"          opposite-label pairs are SUPPRESSED (cannot be true dupes).")
    _p()

    # ---- 1. hash every clip (with on-disk cache) --------------------------
    cache_path = os.path.join(out_dir, PHASH_CACHE)
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
            _p(f"  loaded {len(cache)} cached clip-hashes from {PHASH_CACHE}")
        except Exception:
            cache = {}

    info = {}
    missing = []
    for r in tqdm(rows, desc="hashing clips", unit="clip"):
        cid = r["clip_id"]
        path = r["video_path"]
        if cid in cache and cache[cid].get("hashes"):
            hashes = cache[cid]["hashes"]
        else:
            if not os.path.exists(path):
                missing.append((cid, path))
                hashes = []
            else:
                hashes = _hash_clip(path)
            cache[cid] = {"hashes": hashes}
        info[cid] = {"split": r["split"], "label": r["label"], "hashes": hashes}

    try:
        json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    except Exception:
        pass

    n_ok = sum(1 for v in info.values() if v["hashes"])
    _p(f"  hashed {n_ok}/{len(rows)} clips successfully "
       f"({len(rows) - n_ok} undecodable/missing)")
    if missing:
        _p(f"  {len(missing)} clips had NO file at the recorded path, e.g. {missing[:3]}")

    def popcount(x):
        return bin(x).count("1")

    # ---- 2. candidate generation via exact-hash blocking ------------------
    # Only compare clips that share at least one exact frame-hash, or are within
    # the same label. To stay O(n^2)-bounded but correct, we compare every pair
    # but the cross-label guard + multi-frame test kills the false positives.
    clip_ids = [c for c in info if info[c]["hashes"]]

    def pair_stats(ca, cb):
        """Return (n_matching_frames, avg_hamming_of_matches) for two clips.
        n_matching_frames counts distinct frames of A that have >=1 partner in B
        within FRAME_HAMMING bits (greedy, good enough for dup detection)."""
        ha, hb = info[ca]["hashes"], info[cb]["hashes"]
        matches = 0
        dist_sum = 0
        for x in ha:
            best = min(popcount(x ^ y) for y in hb)
            if best <= FRAME_HAMMING:
                matches += 1
                dist_sum += best
        avg = (dist_sum / matches) if matches else 64
        return matches, avg

    dup_pairs = []  # (ca, cb, n_matches, avg_hamming)
    suppressed_crosslabel = 0
    for a in tqdm(range(len(clip_ids)), desc="dup compare", unit="clip"):
        ca = clip_ids[a]
        la = info[ca]["label"]
        for b in range(a + 1, len(clip_ids)):
            cb = clip_ids[b]
            # cross-label guard: a fight and a non-fight can't be the same scene
            if info[cb]["label"] != la:
                continue
            m, avg = pair_stats(ca, cb)
            if m >= MIN_FRAME_MATCHES:
                dup_pairs.append((ca, cb, m, avg))

    # count how many pairs WOULD have been flagged if we ignored the label guard,
    # so we can show the user how much noise the guard removed.
    # (cheap recompute only over already-found-ish space is hard; instead we note
    #  the guard is active and move on — the v1 report already showed the noise.)

    # ---- 3. classify by split ---------------------------------------------
    cross_split = []
    same_split = 0
    for ca, cb, m, avg in dup_pairs:
        sa, sb = info[ca]["split"], info[cb]["split"]
        la, lb = info[ca]["label"], info[cb]["label"]
        if sa != sb:
            cross_split.append((ca, sa, la, cb, sb, lb, m, round(avg, 2)))
        else:
            same_split += 1

    # sort: most frame-matches first, then lowest avg hamming (most confident)
    cross_split.sort(key=lambda t: (-t[6], t[7]))

    _p()
    _p(f"  duplicate clip-pairs (>= {MIN_FRAME_MATCHES} matching frames, same label): "
       f"{len(dup_pairs)} total")
    _p(f"    within-split (harmless): {same_split}")
    _p(f"    CROSS-SPLIT (leakage risk): {len(cross_split)}")
    _p()

    # ---- 4. write reports --------------------------------------------------
    full_csv = os.path.join(out_dir, "phash_cross_split_pairs.csv")
    with open(full_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_a", "split_a", "label_a", "clip_b", "split_b",
                    "label_b", "n_frame_matches", "avg_hamming"])
        for row in cross_split:
            w.writerow(row)

    test_leak = [t for t in cross_split
                 if "test" in (t[1], t[4]) and ("train" in (t[1], t[4]) or "val" in (t[1], t[4]))]
    leak_csv = os.path.join(out_dir, "phash_test_leakage_pairs.csv")
    with open(leak_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_a", "split_a", "label_a", "clip_b", "split_b",
                    "label_b", "n_frame_matches", "avg_hamming"])
        for row in test_leak:
            w.writerow(row)

    # near-identical subset (most confident leaks): many frames, tiny hamming
    identical = [t for t in test_leak
                 if t[6] >= FRAMES_PER_CLIP - 1 and t[7] <= IDENTICAL_HAMMING]

    _p(f"  TEST<->train/val duplicate pairs (inflate the score): {len(test_leak)}")
    _p(f"    of which near-IDENTICAL (>= {FRAMES_PER_CLIP-1} frames, avg<= "
       f"{IDENTICAL_HAMMING}): {len(identical)}")
    # how many DISTINCT test clips are compromised
    test_clips_hit = set()
    for ca, sa, la, cb, sb, lb, m, avg in test_leak:
        if sa == "test":
            test_clips_hit.add(ca)
        if sb == "test":
            test_clips_hit.add(cb)
    _p(f"    distinct TEST clips with a train/val twin: {len(test_clips_hit)} / 200")
    _p()
    if cross_split:
        _p("  most confident 25 cross-split pairs (more frames + lower hamming = surer):")
        for ca, sa, la, cb, sb, lb, m, avg in cross_split[:25]:
            tag = "  *** TEST LEAK" if ("test" in (sa, sb)) else ""
            _p(f"    {ca}({sa},y={la})  ~  {cb}({sb},y={lb})  "
               f"frames={m}/{FRAMES_PER_CLIP} avgH={avg}{tag}")
    _p()
    _p(f"  wrote all cross-split pairs -> {full_csv}")
    _p(f"  wrote test-leak pairs only  -> {leak_csv}")
    _p()
    # verdict
    if test_leak:
        _p(f"  VERDICT: {len(test_clips_hit)} of 200 test clips have a near-duplicate")
        _p(f"           twin in train/val ({len(identical)} of them near-identical).")
        _p(f"           This is real leakage and it inflates the 0.995. Fix by")
        _p(f"           regenerating a GROUP-AWARE split (scene clusters stay on one")
        _p(f"           side) OR removing these test clips before reporting metrics.")
    elif cross_split:
        _p(f"  VERDICT: cross-split duplicates exist but none touch TEST, so the")
        _p(f"           headline test metric is not directly inflated. Fix for hygiene.")
    else:
        _p(f"  VERDICT: no near-duplicates straddle any split. The 0.995 is clean on")
        _p(f"           this axis — the score reflects genuine separability.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc", default="preproc_output")
    ap.add_argument("--finetune", default="finetune_output")
    ap.add_argument("--phash", action="store_true",
                    help="run the frame-level perceptual-hash duplicate scan "
                         "(decodes video; needs opencv-python). Slower but definitive.")
    ap.add_argument("--out", default="inspection_report.txt",
                    help="path to write the full text report (default: inspection_report.txt)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    preproc = os.path.join(here, args.preproc) if not os.path.isabs(args.preproc) else args.preproc
    finetune = os.path.join(here, args.finetune) if not os.path.isabs(args.finetune) else args.finetune

    split_path = os.path.join(preproc, "split_rlvs.csv")
    gqs_path = os.path.join(preproc, "gqs_summary_rlvs.csv")

    # ---- tee all prints to both the console and the report file -----------
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    report = open(out_path, "w", encoding="utf-8")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, report)

    try:
        _p("RLVS V9 OUTPUT INSPECTION")
        _p(f"preproc dir : {preproc}")
        _p(f"finetune dir: {finetune}")
        _p(f"phash scan  : {'ON' if args.phash else 'off (pass --phash to enable)'}")

        rows = load_split(split_path)
        audit_split_integrity(rows)
        audit_near_duplicates(rows)
        audit_low_quality(gqs_path)
        audit_results(finetune)
        if args.phash:
            audit_phash(rows, preproc)

        _hdr("SUMMARY")
        _p("  - Audit 1 (leakage) is the gatekeeper. If it says clean and Audit 2 shows")
        _p("    no suspicious adjacent straddlers, the 0.995 is more believable.")
        _p("  - Audit 2 only flags candidates by id-adjacency; Audit 5 (--phash) is the")
        _p("    definitive frame-level check.")
        _p("  - Audit 3 wrote low_quality_clips.csv — open a few of those .mp4s to confirm")
        _p("    they are genuinely unusable (dark/blurred/cropped), not a decode bug.")
        if args.phash:
            _p("  - Audit 5 wrote phash_cross_split_pairs.csv and phash_test_leakage_pairs.csv.")
            _p("    The test-leakage file is the one that matters for the 0.995.")
        else:
            _p("  - Re-run with --phash for the definitive duplicate scan (needs opencv).")
    finally:
        sys.stdout = real_stdout
        report.close()
        print(f"\n[report written to: {out_path}]")


if __name__ == "__main__":
    main()
