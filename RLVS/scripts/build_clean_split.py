"""
build_clean_split.py — Regenerate a GROUP-AWARE (leakage-free) RLVS split.

Problem this fixes
------------------
inspect_outputs.py --phash (v2) showed the random seed=42 split scatters
near-duplicate scene re-cuts across train/val/test: 36/200 test clips had a
train/val twin. The re-eval proved this didn't inflate the score much, but for a
defensible protocol no scene may straddle a split boundary.

Approach
--------
1. Recompute near-duplicate pairs from the cached frame-hashes (phash_cache_v2.json)
   produced by inspect_outputs.py --phash. No video decode needed — instant.
   Same rules as the scan: DCT pHash, frame match <= FRAME_HAMMING bits,
   >= MIN_FRAME_MATCHES matching frames, and OPPOSITE-LABEL pairs suppressed.
2. Union-find over ALL such pairs (within-split AND cross-split) -> scene clusters.
   Every clip with no twin is its own singleton cluster. Clusters are same-label
   by construction (cross-label pairs are suppressed), which lets us keep balance
   exact by splitting each label independently.
3. Assign whole clusters (never individual clips) to test / val / train with a
   greedy largest-cluster-first bin-pack, hitting per-label targets:
       test = 100 pos + 100 neg, val = 100 pos + 100 neg, train = the rest.
   seed=42 controls cluster shuffling for reproducibility.
4. Verify: 0 cross-split duplicate pairs remain; balance and sizes are correct.
   Write split_rlvs_clean.csv (same columns as split_rlvs.csv) + a cluster report.

Usage
-----
    python build_clean_split.py
    python build_clean_split.py --cache preproc_output/phash_cache_v2.json \
        --split preproc_output/split_rlvs.csv --out preproc_output/split_rlvs_clean.csv
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- must match inspect_outputs.py v2 so clusters are consistent with the scan -
FRAMES_PER_CLIP = 8
FRAME_HAMMING = 8
MIN_FRAME_MATCHES = 3

# --- split design ------------------------------------------------------------
SEED = 42
VAL_POS, VAL_NEG = 100, 100
TEST_POS, TEST_NEG = 100, 100


def popcount(x):
    return bin(x).count("1")


def load_cache(cache_path):
    """clip_id -> list[int] frame hashes, from the v2 phash cache."""
    raw = json.load(open(cache_path, encoding="utf-8"))
    out = {}
    for cid, v in raw.items():
        hs = v.get("hashes") if isinstance(v, dict) else v
        if hs:
            out[cid] = hs
    return out


def load_split(split_path):
    """clip_id -> dict(label:int, row:dict) preserving original CSV columns."""
    rows = {}
    with open(split_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["clip_id"]] = r
    return rows


def pair_is_dup(ha, hb):
    """True if two clips match on >= MIN_FRAME_MATCHES frames within FRAME_HAMMING."""
    matches = 0
    for x in ha:
        # short-circuit: does any frame of B match this frame of A?
        for y in hb:
            if popcount(x ^ y) <= FRAME_HAMMING:
                matches += 1
                break
        if matches >= MIN_FRAME_MATCHES:
            return True
    return False


def build_pairs(hashes, labels):
    """All same-label near-duplicate clip pairs. O(n^2) over clips with hashes."""
    from tqdm import tqdm
    ids = [c for c in hashes if c in labels]
    pairs = []
    for i in tqdm(range(len(ids)), desc="pairing", unit="clip"):
        ca = ids[i]
        la = labels[ca]
        ha = hashes[ca]
        for j in range(i + 1, len(ids)):
            cb = ids[j]
            if labels[cb] != la:        # cross-label guard
                continue
            if pair_is_dup(ha, hashes[cb]):
                pairs.append((ca, cb))
    return pairs


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def make_clusters(all_ids, pairs):
    uf = UnionFind(all_ids)
    for a, b in pairs:
        uf.union(a, b)
    clusters = defaultdict(list)
    for cid in all_ids:
        clusters[uf.find(cid)].append(cid)
    return list(clusters.values())


def assign_clusters(clusters, labels, want_pos, want_neg, rng):
    """Greedy: place whole clusters (largest first) into this split until the
    per-label quota is met. Returns (chosen_ids set, remaining_clusters)."""
    # clusters are same-label; tag each with its label and size
    tagged = [(c, labels[c[0]], len(c)) for c in clusters]
    rng.shuffle(tagged)
    tagged.sort(key=lambda t: -t[2])  # largest first so big scenes can't overflow

    chosen, remaining = [], []
    got_pos = got_neg = 0
    for c, lab, sz in tagged:
        if lab == 1 and got_pos + sz <= want_pos:
            chosen += c; got_pos += sz
        elif lab == 0 and got_neg + sz <= want_neg:
            chosen += c; got_neg += sz
        else:
            remaining.append(c)
    return set(chosen), remaining, got_pos, got_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join("preproc_output", "phash_cache_v2.json"))
    ap.add_argument("--split", default=os.path.join("preproc_output", "split_rlvs.csv"))
    ap.add_argument("--out", default=os.path.join("preproc_output", "split_rlvs_clean.csv"))
    ap.add_argument("--report", default=os.path.join("preproc_output", "clean_split_report.txt"))
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    def _abs(p): return p if os.path.isabs(p) else os.path.join(here, p)
    cache_path, split_path = _abs(args.cache), _abs(args.split)
    out_path, report_path = _abs(args.out), _abs(args.report)

    report = open(report_path, "w", encoding="utf-8")

    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, d):
            for x in self.s: x.write(d)
        def flush(self):
            for x in self.s: x.flush()
    real = sys.stdout
    sys.stdout = _Tee(real, report)

    try:
        print("RLVS — GROUP-AWARE CLEAN SPLIT BUILDER")
        print("=" * 72)
        rng = random.Random(SEED)

        rows = load_split(split_path)
        labels = {cid: int(r["label"]) for cid, r in rows.items()}
        hashes = load_cache(cache_path)
        print(f"clips in split CSV : {len(rows)}")
        print(f"clips with hashes  : {len(hashes)}")
        no_hash = [c for c in rows if c not in hashes]
        if no_hash:
            print(f"  WARN: {len(no_hash)} clips have no cached hash; "
                  f"treated as singletons: {no_hash[:5]}")

        # ---- pairs + clusters ---------------------------------------------
        pairs = build_pairs(hashes, labels)
        all_ids = list(rows.keys())
        clusters = make_clusters(all_ids, pairs)
        multi = [c for c in clusters if len(c) > 1]
        print(f"\nduplicate pairs (same-label) : {len(pairs)}")
        print(f"scene clusters total         : {len(clusters)} "
              f"(singletons {len(clusters)-len(multi)}, multi-clip {len(multi)})")
        if multi:
            big = sorted(multi, key=len, reverse=True)[:8]
            print("  largest scene clusters:")
            for c in big:
                print(f"    size {len(c)} y={labels[c[0]]}: {sorted(c)}")

        # sanity: any cluster mixing labels? (should be impossible)
        mixed = [c for c in clusters if len({labels[x] for x in c}) > 1]
        if mixed:
            print(f"  ERROR: {len(mixed)} clusters mix labels — cross-label guard failed!")

        # ---- assign clusters to test, then val, rest -> train -------------
        test_ids, rem, tp, tn = assign_clusters(clusters, labels, TEST_POS, TEST_NEG, rng)
        val_ids, rem, vp, vn = assign_clusters(rem, labels, VAL_POS, VAL_NEG, rng)
        train_ids = set()
        for c in rem:
            train_ids.update(c)

        def counts(idset):
            p = sum(1 for c in idset if labels[c] == 1)
            n = sum(1 for c in idset if labels[c] == 0)
            return len(idset), p, n

        print("\n" + "=" * 72)
        print("RESULTING SPLIT")
        print("=" * 72)
        for name, idset in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            tot, p, n = counts(idset)
            print(f"  {name:5s}: n={tot:4d}  pos={p:4d}  neg={n:4d}")
        if (tp, tn) != (TEST_POS, TEST_NEG):
            print(f"  NOTE: test quota not hit exactly ({tp}/{tn}) — a large cluster")
            print(f"        couldn't fit. Acceptable if close; see counts above.")
        if (vp, vn) != (VAL_POS, VAL_NEG):
            print(f"  NOTE: val quota not hit exactly ({vp}/{vn}).")

        split_of = {}
        for c in train_ids: split_of[c] = "train"
        for c in val_ids:   split_of[c] = "val"
        for c in test_ids:  split_of[c] = "test"

        # ---- VERIFY: no duplicate pair straddles the new split ------------
        straddle = [(a, b) for a, b in pairs if split_of[a] != split_of[b]]
        print("\n" + "=" * 72)
        print("VERIFICATION")
        print("=" * 72)
        print(f"  cross-split duplicate pairs remaining: {len(straddle)}")
        if straddle:
            print("    ERROR: clusters were broken — this should be 0.")
            for a, b in straddle[:10]:
                print(f"      {a}({split_of[a]}) ~ {b}({split_of[b]})")
        else:
            print("    OK — every scene cluster is wholly inside one split. No leakage.")

        # ---- write the clean split CSV (same columns as the original) -----
        fieldnames = list(next(iter(rows.values())).keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for cid in all_ids:
                r = dict(rows[cid])
                r["split"] = split_of[cid]   # overwrite split column only
                w.writerow(r)
        print(f"\n  wrote clean split -> {out_path}")
        print(f"  (same columns as split_rlvs.csv; only the 'split' column changed)")
        print("\n  NEXT: re-run evaluation pointing at split_rlvs_clean.csv to get the")
        print("        duplicate-free test metric. Training data changes too, so for a")
        print("        fully clean number you may re-fine-tune on the new train set.")
    finally:
        sys.stdout = real
        report.close()
        print(f"\n[report written to: {report_path}]")


if __name__ == "__main__":
    main()
