#!/usr/bin/env python3
"""Draw tracker boxes (solid) and ground-truth boxes (thin white) on the video."""

from __future__ import annotations

import argparse
import csv
import json

import cv2

COLORS = [(0, 215, 255), (255, 128, 0)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5, help="output size relative to the input")
    a = ap.parse_args()

    gt = json.loads(open(a.gt).read())
    names = list(gt)
    pred: dict[str, dict[int, tuple]] = {n: {} for n in names}
    status: dict[str, dict[int, str]] = {n: {} for n in names}
    with open(a.tracks) as f:
        for r in csv.DictReader(f):
            if r["target"] not in pred:
                continue
            status[r["target"]][int(r["frame"])] = r["status"]
            if r["x"]:
                pred[r["target"]][int(r["frame"])] = tuple(int(r[k]) for k in "xywh")

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer, f, s = None, 0, a.scale
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for k, n in enumerate(names):
            g = gt[n][f] if f < len(gt[n]) else None
            if g:
                x, y, w, h = g["box"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            p = pred[n].get(f)
            if p:
                x, y, w, h = p
                cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS[k], 4)
                cv2.putText(frame, n, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[k], 3)
            cv2.putText(frame, f"{n}: {status[n].get(f, '-')}  gt: {'yes' if g else 'no'}",
                        (20, 50 + 45 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS[k], 3)
        cv2.putText(frame, f"frame {f}", (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, small.shape[1::-1])
        writer.write(small)
        f += 1
    cap.release()
    if writer is not None:
        writer.release()
    print(f"{f} frames -> {a.out}")


if __name__ == "__main__":
    main()
