#!/usr/bin/env python3
"""Score tracks.csv against ground truth from make_synthetic_video.py."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vtrack.detection import iou  # noqa: E402


def evaluate(tracks_csv: str, gt_json: str, min_visible: float = 0.6, min_width: int = 25) -> dict:
    gt = json.loads(Path(gt_json).read_text())
    pred: dict[str, dict[int, tuple]] = {n: {} for n in gt}
    with open(tracks_csv) as f:
        for r in csv.DictReader(f):
            if r["x"]:
                pred[r["target"]][int(r["frame"])] = tuple(int(r[k]) for k in "xywh")

    report = {}
    for name, frames in gt.items():
        others = [o for o in gt if o != name]
        should, hit, emitted, correct, swaps, hidden_emits = 0, 0, 0, 0, 0, 0
        for f, g in enumerate(frames):
            p = pred[name].get(f)
            visible = g is not None and g["visible"] >= min_visible and g["box"][2] >= min_width
            if visible:
                should += 1
                if p and iou(p, tuple(g["box"])) >= 0.5:
                    hit += 1
            if p:
                emitted += 1
                if g is not None and iou(p, tuple(g["box"])) >= 0.3:
                    correct += 1
                if g is None or g["visible"] < 0.2:
                    hidden_emits += 1
                for o in others:
                    og = gt[o][f]
                    if og and iou(p, tuple(og["box"])) > (iou(p, tuple(g["box"])) if g else 0):
                        swaps += 1
        report[name] = {
            "frames_clearly_visible": should,
            "recall_iou50": round(hit / should, 3) if should else None,
            "positions_emitted": emitted,
            "precision_iou30": round(correct / emitted, 3) if emitted else None,
            "identity_swaps": swaps,
            "positions_while_hidden": hidden_emits,
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks_csv")
    ap.add_argument("gt_json")
    a = ap.parse_args()
    print(json.dumps(evaluate(a.tracks_csv, a.gt_json), indent=2))


if __name__ == "__main__":
    main()
