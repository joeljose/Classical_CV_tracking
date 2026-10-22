#!/usr/bin/env python3
"""Turn per-frame detector boxes into pseudo ground truth for the two targets.

1. Link detections into tracklets (greedy IoU matching, short gaps allowed).
2. Score each long tracklet against each reference image with the same
   appearance model the tracker uses, on its largest crops.
3. Pick the best tracklet per reference, or take the ids passed with --pin
   after looking at the contact sheet.

Output gt.json has, per target, one entry per frame: {"box": [x, y, w, h],
"visible": 1.0} where the detector found the target, else null. It is
detector output, not hand labels, so check contact_sheet.jpg before trusting it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from track_vehicles import DEFAULT_REFS, load_refs  # noqa: E402
from vtrack.appearance import AppearanceModel  # noqa: E402
from vtrack.detection import iou  # noqa: E402


def load_detections(path: str) -> dict[int, list[tuple]]:
    dets = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            dets[int(r["frame"])].append(tuple(int(r[k]) for k in "xywh"))
    return dets


def link(dets: dict[int, list[tuple]], n_frames: int, min_iou: float = 0.3, max_gap: int = 10) -> list[dict]:
    tracks: list[dict] = []  # {"id", "boxes": {frame: box}, "last": frame}
    for f in range(n_frames):
        live = [t for t in tracks if f - t["last"] <= max_gap]
        pairs = sorted(((iou(t["boxes"][t["last"]], b), ti, bi)
                        for ti, t in enumerate(live) for bi, b in enumerate(dets.get(f, []))), reverse=True)
        used_t, used_b = set(), set()
        for s, ti, bi in pairs:
            if s < min_iou:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            live[ti]["boxes"][f] = dets[f][bi]
            live[ti]["last"] = f
        for bi, b in enumerate(dets.get(f, [])):
            if bi not in used_b:
                tracks.append({"id": len(tracks), "boxes": {f: b}, "last": f})
    return tracks


def read_frames(video: str, wanted: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(video)
    out, f = {}, 0
    last = max(wanted) if wanted else -1
    while f <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if f in wanted:
            out[f] = frame
        f += 1
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--detections", default="gt/detections.csv")
    ap.add_argument("--ref", action="append", metavar="NAME=PATH")
    ap.add_argument("--pin", action="append", default=[], metavar="NAME=ID[,ID...]",
                    help="use these tracklet ids for a target instead of the best-scoring one")
    ap.add_argument("--out-dir", default="gt")
    ap.add_argument("--min-len", type=int, default=25, help="ignore tracklets shorter than this")
    ap.add_argument("--samples", type=int, default=8, help="crops scored per tracklet")
    a = ap.parse_args()

    refs = load_refs(a.ref or DEFAULT_REFS)
    models = {n: AppearanceModel(n, img) for n, img in refs.items()}
    cap = cv2.VideoCapture(a.video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    tracks = [t for t in link(load_detections(a.detections), n_frames) if len(t["boxes"]) >= a.min_len]
    # Score on the widest boxes: that is where colour and SIFT carry the most.
    samples = {}
    for t in tracks:
        widest = sorted(t["boxes"], key=lambda f: t["boxes"][f][2], reverse=True)[:a.samples]
        samples[t["id"]] = widest
    frames = read_frames(a.video, {f for fs in samples.values() for f in fs})

    scores: dict[int, dict[str, float]] = {}
    for t in tracks:
        per = defaultdict(list)
        for f in samples[t["id"]]:
            x, y, w, h = t["boxes"][f]
            patch = frames[f][max(0, y):y + h, max(0, x):x + w]
            for n, m in models.items():
                per[n].append(m.score(patch).total)
        scores[t["id"]] = {n: float(np.median(v)) for n, v in per.items()}

    pins = {}
    for spec in a.pin:
        n, _, ids = spec.partition("=")
        pins[n] = [int(i) for i in ids.split(",")]
    by_id = {t["id"]: t for t in tracks}
    chosen = {}
    for n in models:
        if n in pins:
            chosen[n] = pins[n]
        else:
            # Margin over the other references, so a generic match does not win.
            def margin(tid, n=n):
                s = scores[tid]
                return s[n] - max((v for k, v in s.items() if k != n), default=0.0)
            chosen[n] = [max(scores, key=lambda tid, n=n: scores[tid][n] + 0.5 * margin(tid))]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt = {}
    for n, ids in chosen.items():
        rows = [None] * n_frames
        for tid in ids:
            for f, b in by_id[tid]["boxes"].items():
                rows[f] = {"box": list(b), "visible": 1.0}
        gt[n] = rows
    (out / "gt.json").write_text(json.dumps(gt))

    with open(out / "tracklets.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "first", "last", "length", *(f"score_{n}" for n in models)])
        for t in sorted(tracks, key=lambda t: -max(scores[t["id"]].values())):
            fs = sorted(t["boxes"])
            wr.writerow([t["id"], fs[0], fs[-1], len(fs), *(f"{scores[t['id']][n]:.3f}" for n in models)])

    # Contact sheet: the reference, then crops from start/middle/end of each chosen tracklet.
    rows_img = []
    for n, ids in chosen.items():
        fs = sorted(f for tid in ids for f in by_id[tid]["boxes"])
        pick = [fs[int(i * (len(fs) - 1) / 5)] for i in range(6)]
        frames.update(read_frames(a.video, set(pick) - set(frames)))
        tiles = [cv2.resize(refs[n], (160, 120))]
        for f in pick:
            x, y, w, h = gt[n][f]["box"]
            crop = frames[f][max(0, y):y + h, max(0, x):x + w]
            tile = cv2.resize(crop, (160, 120))
            cv2.putText(tile, str(f), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(tile)
        rows_img.append(np.hstack(tiles))
    cv2.imwrite(str(out / "contact_sheet.jpg"), np.vstack(rows_img))

    summary = {n: {"tracklets": ids, "frames": sum(r is not None for r in gt[n]),
                   "scores": {tid: scores[tid] for tid in ids}} for n, ids in chosen.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
