"""
Audit the merged wildfire dataset for the two defects that break smoke detection.

Context
-------
The deployed detector scores ~0.006 on the `smoke` channel for real aerial
wildfire photos despite a reported smoke mAP@50 of 0.939. Two findings from
results/smoke_directions.csv motivated this audit:

  1. Every one of the 140 test images that produced a direction estimate has
     the `dba_vd` prefix. Other `dba_*` images (dba_img, dba_pic, dba_small,
     ...) are ground-level scenes annotated with `fire` only, even when a smoke
     plume is plainly visible. Unlabelled smoke trains as background and
     suppresses the class.

  2. Those 140 frames come from just FIVE source videos, and frames inside the
     test split sit as little as ONE frame apart. If the split was made per
     frame rather than per video, test frames are near-duplicates of training
     frames and the 0.939 smoke mAP measures memorisation, not generalisation.

This script quantifies both. Defect 2 is the one that decides whether the
dataset can be salvaged at all, so it is the headline check.

Run on Kaggle (that is where the dataset lives):
    !python audit_labels.py --root /kaggle/input/datasets/dangnguyenminhduy/wildfireuav/merged

Reads labels only (YOLO .txt), so it is fast and needs no GPU.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")
UUIDISH = re.compile(r"^[0-9a-f]{6,}(-[0-9a-f]+)*$", re.I)
# dba_vd<video><6-digit frame>, e.g. dba_vd3000444 -> video 3, frame 000444
VD_ID = re.compile(r"^dba_vd(\d+)$")


def subset_of(stem: str) -> str:
    """Group by leading alphabetic tokens: 'dba_vd3000444' -> 'dba_vd'.

    The distinction that matters is dba_vd (aerial, smoke annotated) versus
    everything else, so a coarse split on the first '_' is not enough.
    """
    out = []
    for tok in stem.split("_"):
        if not tok:
            break
        m = re.match(r"^([A-Za-z]+)", tok)
        if not m:
            break
        alpha, rest = m.group(1), tok[m.end():]
        if rest and UUIDISH.match(tok):
            out.append("uuid")
            break
        out.append(alpha.lower())
        if rest:  # token was alpha+digits (e.g. 'vd3000444') -> stop here
            break
    return "_".join(out) if out else "<other>"


def video_of(stem: str):
    """Source video id for dba_vd frames, else None."""
    m = VD_ID.match(stem)
    if not m or len(m.group(1)) <= 6:
        return None
    return m.group(1)[:-6]


def scan(root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        label_dir = root / "labels" / split
        if not label_dir.is_dir():
            print(f"  ! missing {label_dir}")
            continue
        for txt in sorted(label_dir.glob("*.txt")):
            counts = defaultdict(int)
            for line in txt.read_text().splitlines():
                parts = line.split()
                if parts:
                    counts[int(parts[0])] += 1
            rows.append({
                "split": split,
                "stem": txt.stem,
                "subset": subset_of(txt.stem),
                "video": video_of(txt.stem),
                "fire": counts[0],
                "smoke": counts[1],
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root containing labels/{train,val,test}")
    args = ap.parse_args()

    df = scan(Path(args.root))
    if df.empty:
        raise SystemExit(f"No labels found under {args.root}/labels/")
    df["has_smoke"] = df["smoke"] > 0

    # ---------------- Defect 1: subsets with no smoke annotation -----------
    tot = (df.groupby("subset")
             .agg(images=("fire", "size"), fire_boxes=("fire", "sum"),
                  smoke_boxes=("smoke", "sum"), imgs_with_smoke=("has_smoke", "sum"))
             .reset_index().sort_values("images", ascending=False))
    tot["pct_of_dataset"] = (100 * tot["images"] / len(df)).round(1)

    print("\n" + "=" * 78)
    print("DEFECT 1 — SMOKE ANNOTATION BY SUBSET")
    print("=" * 78)
    print(tot.to_string(index=False))

    clean = tot[tot["smoke_boxes"] > 0]
    poisoned = tot[tot["smoke_boxes"] == 0]
    keep, drop = int(clean["images"].sum()), int(poisoned["images"].sum())
    print(f"\n  subsets WITH smoke labels : {list(clean['subset'])}  -> {keep} imgs ({100*keep/len(df):.1f}%)")
    print(f"  subsets with ZERO smoke   : {list(poisoned['subset'])}  -> {drop} imgs ({100*drop/len(df):.1f}%)")
    print("  (the zero-smoke subsets train smoke->background and must be dropped or re-annotated)")

    # ---------------- Defect 2: video-level leakage ------------------------
    vd = df[df["video"].notna()]
    print("\n" + "=" * 78)
    print("DEFECT 2 — VIDEO-LEVEL LEAKAGE  (the decisive check)")
    print("=" * 78)
    if vd.empty:
        print("  No dba_vd frames found — check the naming assumption.")
        return

    per_split = {s: set(g["video"]) for s, g in vd.groupby("split")}
    for s in SPLITS:
        vids = per_split.get(s, set())
        print(f"  {s:<6} {len(vd[vd.split == s]):>6} frames from {len(vids):>3} videos: {sorted(vids)}")

    tr, va, te = per_split.get("train", set()), per_split.get("val", set()), per_split.get("test", set())
    print(f"\n  distinct videos in the WHOLE dataset: {vd['video'].nunique()}")
    print(f"  train n test overlap: {sorted(tr & te)}")
    print(f"  train n val  overlap: {sorted(tr & va)}")

    leak = bool((tr & te) or (tr & va))
    print("\n" + "-" * 78)
    if leak:
        print("  *** LEAKAGE CONFIRMED ***")
        print("  The same source videos appear in train AND val/test. Adjacent frames are")
        print("  near-duplicates, so the reported smoke mAP measures memorisation of a")
        print("  handful of scenes, not generalisation. The 0.939 figure is not credible.")
    else:
        print("  No video overlap across splits — the split is grouped correctly.")

    n_vid = vd["video"].nunique()
    print(f"\n  EFFECTIVE smoke diversity: {n_vid} distinct scenes "
          f"(not {len(vd)} independent images)")
    if n_vid < 30:
        print("  -> FAR too few scenes to learn a general 'smoke' concept.")
        print("  -> Dropping the zero-smoke subsets alone will NOT fix this.")
        print("  -> External aerial smoke data is MANDATORY.")
    print("-" * 78)

    out = Path("label_audit.csv")
    df.drop(columns=["stem"]).to_csv(out, index=False)
    print(f"\nSaved -> {out.resolve()}")


if __name__ == "__main__":
    main()
