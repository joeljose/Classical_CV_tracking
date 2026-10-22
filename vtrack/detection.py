"""Moving-object blobs from background subtraction, plus box helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class DetectorConfig:
    history: int = 300
    var_threshold: float = 25.0
    min_area: int = 300
    max_area_frac: float = 0.4
    # Loose on purpose: blobs of several merged vehicles are kept and
    # localised later by the tracker's window refinement.
    min_aspect: float = 0.3   # w / h
    max_aspect: float = 6.0


class MotionDetector:
    """MOG2 background subtraction + frame differencing -> candidate boxes."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.history, varThreshold=cfg.var_threshold, detectShadows=True)
        self._prev_gray: np.ndarray | None = None
        self._k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.last_mask: np.ndarray | None = None

    def foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        fg = self.bg.apply(frame)
        # MOG2 labels shadows 127; drop them so boxes hug the vehicle body.
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            # Frame differencing only reinforces MOG2 near existing foreground;
            # on its own it fires on hollow edges of moving objects.
            near_fg = cv2.dilate(fg, self._k7, iterations=2)
            fg = cv2.bitwise_or(fg, cv2.bitwise_and(diff, near_fg))
        self._prev_gray = gray

        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._k3)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k7, iterations=2)
        self.last_mask = fg
        return fg

    def detect(self, frame: np.ndarray) -> list[Box]:
        fg = self.foreground_mask(frame)
        H, W = fg.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        boxes: list[Box] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < self.cfg.min_area or w * h > self.cfg.max_area_frac * W * H:
                continue
            if self.cfg.min_aspect <= w / h <= self.cfg.max_aspect:
                boxes.append((x, y, w, h))
        return boxes


def clip_box(box: Box, shape: tuple[int, ...]) -> Box | None:
    x, y, w, h = box
    H, W = shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1 - x0, y1 - y0


def crop(frame: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    return frame[y:y + h, x:x + w]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
