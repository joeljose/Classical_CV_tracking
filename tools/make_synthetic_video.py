#!/usr/bin/env python3
"""Render a synthetic highway clip for testing the tracker end to end.

The clip pastes the two reference images (and recoloured distractor vehicles)
onto a static road background, moving them towards the camera so they grow,
and passes a distractor in front of the Audi to create a full occlusion.
Ground truth boxes are written alongside as JSON.

This is a test fixture only: results on it say nothing about accuracy on the
real footage.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

W, H = 960, 540
HORIZON = 120


def background(rng: np.random.Generator) -> np.ndarray:
    bg = np.full((H, W, 3), (95, 100, 92), np.uint8)
    bg[:HORIZON] = (70, 110, 80)                      # verge / trees
    road = np.array([[W * 0.42, HORIZON], [W * 0.58, HORIZON], [W, H], [0, H]], np.int32)
    cv2.fillPoly(bg, [road], (105, 105, 108))
    for lane in (0.333, 0.667):                         # dashed lane markings
        top = (W * (0.42 + 0.16 * lane), HORIZON)
        bot = (W * lane, H)
        for t0 in np.arange(0, 1, 0.08):
            p0 = np.add(top, np.multiply(np.subtract(bot, top), t0 ** 1.5))
            p1 = np.add(top, np.multiply(np.subtract(bot, top), (t0 + 0.03) ** 1.5))
            cv2.line(bg, tuple(p0.astype(int)), tuple(p1.astype(int)), (220, 220, 220), 2)
    noise = rng.normal(0, 4, bg.shape)
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


def recolor(img: np.ndarray, hue_shift: int, sat_gain: float = 1.0, val_gain: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_gain, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def lane_position(lane: float, t: float, base_w: int) -> tuple[int, int, int]:
    """Centre x, bottom y and width of a vehicle at progress t in [0, 1]."""
    p = t ** 1.6
    y_bottom = HORIZON + 15 + p * (H + 60 - HORIZON)
    x = W * (0.5 + (lane - 0.5) * (0.16 + 0.84 * p))
    width = base_w * (0.12 + 0.88 * p)
    return int(x), int(y_bottom), int(width)


def paste(frame: np.ndarray, img: np.ndarray, cx: int, y_bottom: int, width: int):
    h = max(4, int(width * img.shape[0] / img.shape[1]))
    width = max(4, width)
    sprite = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    x0, y0 = cx - width // 2, y_bottom - h
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(W, x0 + width), min(H, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    frame[fy0:fy1, fx0:fx1] = sprite[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    return [fx0, fy0, fx1 - fx0, fy1 - fy0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audi", default="refs/1000073924.png")
    ap.add_argument("--toyota", default="refs/1000073925.png")
    ap.add_argument("--out", default="synthetic.mp4")
    ap.add_argument("--gt", default="synthetic_gt.json")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    audi, toyota = cv2.imread(a.audi), cv2.imread(a.toyota)
    red = recolor(toyota, hue_shift=-110, sat_gain=1.3)      # red hatchback
    dark = recolor(audi, 0, sat_gain=0.5, val_gain=0.45)  # dark-grey saloon
    bg = background(rng)

    # (image, name, lane, start frame, frames to reach the camera, base width)
    vehicles = [
        (dark, None, 0.80, 0, 200, 420),
        (toyota, "Blue Toyota", 0.22, 20, 190, 360),
        (audi, "Silver Audi", 0.55, 30, 200, 440),
        # Faster vehicle in an adjacent lane that drifts across and passes in
        # front of the Audi (closer to the camera), fully hiding it for a while.
        (red, None, 0.55, 95, 95, 520),
    ]

    vid = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    gt: dict[str, list] = {"Silver Audi": [], "Blue Toyota": []}
    for f in range(a.frames):
        frame = bg.copy()
        boxes = {}
        states = []
        for img, name, lane, start, dur, bw in vehicles:
            t = (f - start) / dur
            if 0 <= t <= 1:
                states.append((lane_position(lane, t, bw), img, name))
        # Paint far vehicles first so nearer ones occlude them.
        for (cx, yb, w), img, name in sorted(states, key=lambda s: s[0][1]):
            box = paste(frame, img, cx, yb, w)
            if name and box:
                boxes[name] = box
            for other in boxes:
                if other != name and box and boxes[other]:
                    ob = boxes[other]
                    # record visible fraction of already painted targets
                    ix = max(0, min(ob[0] + ob[2], box[0] + box[2]) - max(ob[0], box[0]))
                    iy = max(0, min(ob[1] + ob[3], box[1] + box[3]) - max(ob[1], box[1]))
                    boxes[other] = ob + [1 - ix * iy / (ob[2] * ob[3])] if len(ob) == 4 else \
                        ob[:4] + [ob[4] * (1 - ix * iy / (ob[2] * ob[3]))]
        gain = 1.0 + 0.05 * np.sin(f / 20)               # mild exposure flicker
        frame = np.clip(frame * gain + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)
        vid.write(frame)
        for name in gt:
            b = boxes.get(name)
            if b is None:
                gt[name].append(None)
            else:
                gt[name].append({"box": b[:4], "visible": round(b[4] if len(b) > 4 else 1.0, 3)})
    vid.release()
    with open(a.gt, "w") as fh:
        json.dump(gt, fh)


if __name__ == "__main__":
    main()
