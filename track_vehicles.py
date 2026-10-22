#!/usr/bin/env python3
"""Track the reference vehicles through a video with classical CV only.

Example:
    python track_vehicles.py --video 1900-151662242.mp4 \
        --ref "Silver Audi=refs/1000073924.png" \
        --ref "Blue Toyota=refs/1000073925.png" \
        --out-dir output
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from vtrack import DetectorConfig, FrameResult, Status, TrackerConfig, VehicleTracker

DEFAULT_REFS = [
    "Silver Audi=refs/1000073924.png",
    "Blue Toyota=refs/1000073925.png",
]
COLORS = [(0, 215, 255), (255, 128, 0), (0, 200, 0), (200, 0, 200)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--ref", action="append", metavar="NAME=PATH",
                   help="reference image for a target (repeatable). Default: the two refs/ images.")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--process-width", type=int, default=960,
                   help="frames wider than this are downscaled for processing (0 = never); "
                        "reported boxes are always in original-resolution pixels")
    p.add_argument("--debug-candidates", action="store_true",
                   help="also write candidates.csv with every candidate box and its scores, "
                        "and draw candidates in the annotated video (useful for tuning thresholds)")
    d = TrackerConfig()
    p.add_argument("--acquire-threshold", type=float, default=d.acquire_threshold)
    p.add_argument("--keep-threshold", type=float, default=d.keep_threshold)
    p.add_argument("--identity-threshold", type=float, default=d.identity_threshold)
    p.add_argument("--confirm-frames", type=int, default=d.confirm_frames)
    p.add_argument("--max-coast", type=int, default=d.max_coast)
    p.add_argument("--min-area", type=int, default=DetectorConfig().min_area)
    p.add_argument("--acquire-min-width", type=int, default=d.acquire_min_width,
                   help="narrowest candidate (processing pixels) that may start a track")
    p.add_argument("--no-reacquire", action="store_true",
                   help="never acquire a target again after it left the frame")
    p.add_argument("--detect-every", type=int, default=d.detect_every,
                   help="run full detection every N frames and follow by template matching in between")
    p.add_argument("--forward-only", action="store_true",
                   help="skip the backward pass (faster, lower recall on cars approaching the camera)")
    return p.parse_args()


def load_refs(specs: list[str]) -> dict[str, "cv2.typing.MatLike"]:
    refs = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"cannot read reference image {path!r}")
        refs[name] = img
    return refs


def scale_box(box, f: float):
    return None if box is None else tuple(int(round(v * f)) for v in box)


def draw(frame, results, names, candidates=None, f: float = 1.0):
    out = frame.copy()
    if candidates:
        for box, scores in candidates:
            x, y, w, h = scale_box(box, f)
            cv2.rectangle(out, (x, y), (x + w, y + h), (160, 160, 160), 1)
            txt = " ".join(f"{s:.2f}" for s in scores.values())
            cv2.putText(out, txt, (x, y + h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
    for k, r in enumerate(results):
        color = COLORS[k % len(COLORS)]
        if r.box is not None:
            x, y, w, h = scale_box(r.box, f)
            thick = 2 if r.status == Status.TRACKED else 1
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thick)
            cv2.putText(out, f"{r.target} {r.score:.2f}", (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(out, f"{names[k]}: {r.status.value}", (10, 22 + 22 * k),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def read_frames(path: str, process_width: int):
    """Decode once, downscale for processing and keep the frames JPEG-encoded,
    so the clip can be replayed backwards without holding raw frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, inv = [], 1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if process_width and frame.shape[1] > process_width:
            inv = frame.shape[1] / process_width
            frame = cv2.resize(frame, (process_width, int(round(frame.shape[0] / inv))),
                               interpolation=cv2.INTER_AREA)
        frames.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1])
    cap.release()
    return frames, fps, inv


def run_pass(refs, cfg: TrackerConfig, frames, order, debug: bool):
    tracker = VehicleTracker(refs, cfg)
    cands = {}
    for idx in order:
        tracker.process(cv2.imdecode(frames[idx], cv2.IMREAD_COLOR))
        if debug:
            cands[idx] = tracker.last_candidates
    return tracker.history, cands


def merge(forward: list, backward: list) -> list:
    """Per frame and target, keep the pass that measured the target; if both
    did, keep the higher-scoring box. Statuses without a box prefer occluded,
    since that is the more specific answer."""
    out = []
    for f_row, b_row in zip(forward, backward):
        row = []
        for f, b in zip(f_row, b_row):
            if f.box is not None and b.box is not None:
                r = f if f.score >= b.score else b
            elif f.box is not None or b.box is not None:
                r = f if f.box is not None else b
            elif Status.OCCLUDED in (f.status, b.status):
                r = f if f.status == Status.OCCLUDED else b
            else:
                r = f
            row.append(FrameResult(f.frame, r.target, r.status, r.box, r.score))
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    refs = load_refs(args.ref or DEFAULT_REFS)
    names = list(refs)
    cfg = TrackerConfig(acquire_threshold=args.acquire_threshold, keep_threshold=args.keep_threshold,
                        confirm_frames=args.confirm_frames, max_coast=args.max_coast,
                        identity_threshold=args.identity_threshold,
                        acquire_min_width=args.acquire_min_width,
                        reacquire_after_exit=not args.no_reacquire,
                        detect_every=args.detect_every,
                        detector=DetectorConfig(min_area=args.min_area))

    frames, fps, inv = read_frames(args.video, args.process_width)
    if args.max_frames:
        frames = frames[:args.max_frames]
    n = len(frames)
    t0 = time.perf_counter()
    history, cands = run_pass(refs, cfg, frames, range(n), args.debug_candidates)
    timing = {"forward_s": round(time.perf_counter() - t0, 1)}
    if not args.forward_only:
        # Cars drive towards the camera, so they are identifiable late (large)
        # and not early (small). Running the clip backwards acquires each car
        # while it is large and follows it back to where it first appeared.
        t0 = time.perf_counter()
        back, _ = run_pass(refs, cfg, frames, range(n - 1, -1, -1), False)
        timing["backward_s"] = round(time.perf_counter() - t0, 1)
        back = [[FrameResult(n - 1 - r.frame, r.target, r.status, r.box, r.score) for r in row]
                for row in reversed(back)]
        history = merge(history, back)
    del frames

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_video:
        cap = cv2.VideoCapture(args.video)
        writer = None
        for idx in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(draw(frame, history[idx], names, cands.get(idx), inv))
        cap.release()
        if writer is not None:
            writer.release()

    with open(out_dir / "tracks.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "target", "status", "x", "y", "w", "h", "cx", "cy", "score"])
        for row in history:
            for r in row:
                if r.box is not None:
                    x, y, w, h = scale_box(r.box, inv)
                    pos = [x, y, w, h, f"{x + w / 2:.1f}", f"{y + h / 2:.1f}", f"{r.score:.4f}"]
                else:
                    pos = [""] * 7
                wr.writerow([r.frame, f"{r.frame / fps:.3f}", r.target, r.status.value, *pos])

    if args.debug_candidates:
        with open(out_dir / "candidates.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "x", "y", "w", "h", *(f"score_{n}" for n in names)])
            for idx, cs in cands.items():
                for box, scores in cs:
                    wr.writerow([idx, *scale_box(box, inv), *(f"{scores[n]:.4f}" for n in names)])

    track_s = sum(timing.values())
    timing["tracking_fps"] = round(n / track_s, 2) if track_s else None
    summary = {"video": args.video, "frames": n, "fps": fps, "timing": timing, "targets": {}}
    for k, name in enumerate(names):
        rows = [h[k] for h in history]
        located = [r.frame for r in rows if r.status.has_position]
        summary["targets"][name] = {
            "status_counts": dict(Counter(r.status.value for r in rows)),
            "frames_with_position": len(located),
            "first_located_frame": located[0] if located else None,
            "last_located_frame": located[-1] if located else None,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
