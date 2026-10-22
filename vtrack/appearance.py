"""Appearance models built from a single reference image.

Every cue here is hand-crafted (colour histograms, HOG, SIFT keypoints,
normalised cross-correlation). No learned features are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Fraction of the box kept when computing colour statistics. Vehicle boxes
# from background subtraction include some road at the edges; the reference
# crops include some background too. Using the central region reduces both.
_INNER_W = 0.80
_INNER_H = 0.85

_HS_BINS = (18, 16)
_V_BINS = 16
_HOG_SIZE = (64, 48)  # (w, h) window the patch is resized to for HOG
_HOG_CELL = 8
_HOG_BINS = 9


def inner_crop(patch: np.ndarray, fw: float = _INNER_W, fh: float = _INNER_H) -> np.ndarray:
    h, w = patch.shape[:2]
    dw, dh = int(w * (1 - fw) / 2), int(h * (1 - fh) / 2)
    return patch[dh:h - dh or h, dw:w - dw or w]


def color_histograms(patch_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised H-S histogram and V histogram of the central region."""
    hsv = cv2.cvtColor(inner_crop(patch_bgr), cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, list(_HS_BINS), [0, 180, 0, 256])
    v = cv2.calcHist([hsv], [2], None, [_V_BINS], [0, 256])
    cv2.normalize(hs, hs, 1.0, 0.0, cv2.NORM_L1)
    cv2.normalize(v, v, 1.0, 0.0, cv2.NORM_L1)
    return hs, v


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 - Bhattacharyya distance, in [0, 1]."""
    return float(1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def hog_vector(patch_bgr: np.ndarray) -> np.ndarray:
    """Dalal-Triggs style HOG: unsigned gradients, 8x8 cells, 2x2 L2-Hys blocks."""
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, _HOG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = ((ang % 180.0) / (180.0 / _HOG_BINS)).astype(np.int32) % _HOG_BINS
    cy, cx = _HOG_SIZE[1] // _HOG_CELL, _HOG_SIZE[0] // _HOG_CELL
    onehot = (bins[..., None] == np.arange(_HOG_BINS)) * mag[..., None]
    cells = onehot.reshape(cy, _HOG_CELL, cx, _HOG_CELL, _HOG_BINS).sum(axis=(1, 3))
    blocks = np.concatenate([cells[:-1, :-1], cells[:-1, 1:], cells[1:, :-1], cells[1:, 1:]], axis=2)
    blocks = blocks / (np.linalg.norm(blocks, axis=2, keepdims=True) + 1e-6)
    blocks = np.minimum(blocks, 0.2)
    blocks = blocks / (np.linalg.norm(blocks, axis=2, keepdims=True) + 1e-6)
    v = blocks.ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass
class ScoreBreakdown:
    total: float
    color: float
    hog: float
    keypoints: float | None  # None when the patch is too small for SIFT


class AppearanceModel:
    """Similarity of an image patch to one reference vehicle image."""

    def __init__(self, name: str, reference_bgr: np.ndarray, *,
                 min_kp_width: int = 60, kp_full_inliers: int = 12):
        self.name = name
        self.reference = reference_bgr
        self.hs, self.v = color_histograms(reference_bgr)
        self.hog = hog_vector(reference_bgr)
        self.aspect = reference_bgr.shape[1] / reference_bgr.shape[0]
        self.min_kp_width = min_kp_width
        self.kp_full_inliers = kp_full_inliers

        self._sift = cv2.SIFT_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_desc = self._sift.detectAndCompute(gray, None)

        # Online colour model, adapted while the target is confidently tracked
        # so that gradual changes (scale, lighting) do not break association.
        # Identity checks always use the fixed reference model above.
        self.online_hs = self.hs.copy()
        self.online_v = self.v.copy()

    # ------------------------------------------------------------------ cues
    def color_score(self, patch_bgr: np.ndarray, online: bool = False) -> float:
        hs, v = color_histograms(patch_bgr)
        ref_hs, ref_v = (self.online_hs, self.online_v) if online else (self.hs, self.v)
        # Brightness matters as much as chroma: silver vs. grey/white cars share
        # near-zero saturation and differ mostly in V.
        return 0.6 * hist_similarity(hs, ref_hs) + 0.4 * hist_similarity(v, ref_v)

    def hog_score(self, patch_bgr: np.ndarray) -> float:
        return float(max(0.0, np.dot(hog_vector(patch_bgr), self.hog)))

    def keypoint_score(self, patch_bgr: np.ndarray) -> float | None:
        """Fraction of a 'full' geometric match: SIFT + ratio test + RANSAC."""
        if patch_bgr.shape[1] < self.min_kp_width or self.ref_desc is None:
            return None
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        # Bring the patch roughly to reference scale so SIFT octaves line up.
        scale = self.reference.shape[1] / gray.shape[1]
        if scale > 1.0:
            gray = cv2.resize(gray, None, fx=min(scale, 3.0), fy=min(scale, 3.0),
                              interpolation=cv2.INTER_CUBIC)
        kp, desc = self._sift.detectAndCompute(gray, None)
        if desc is None or len(kp) < 4:
            return 0.0
        pairs = self._matcher.knnMatch(desc, self.ref_desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 4:
            return 0.0
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # A similarity transform is enough for a frontal view that only
        # changes scale/position, and is much harder to fit by chance than a
        # full homography.
        _, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=5.0)
        n = int(inliers.sum()) if inliers is not None else 0
        return min(1.0, n / self.kp_full_inliers)

    # -------------------------------------------------------------- combined
    def fast_score(self, patch_bgr: np.ndarray) -> float:
        """Colour + HOG only; cheap enough for dense window search."""
        if patch_bgr.size == 0 or min(patch_bgr.shape[:2]) < 8:
            return 0.0
        return 0.75 * self.color_score(patch_bgr) + 0.25 * self.hog_score(patch_bgr)

    def score(self, patch_bgr: np.ndarray, online: bool = False) -> ScoreBreakdown:
        if patch_bgr.size == 0 or min(patch_bgr.shape[:2]) < 8:
            return ScoreBreakdown(0.0, 0.0, 0.0, None)
        c = self.color_score(patch_bgr, online=online)
        g = self.hog_score(patch_bgr)
        k = self.keypoint_score(patch_bgr)
        if k is None:
            total = 0.75 * c + 0.25 * g
        else:
            # Keypoints can only add evidence; a failed match on a blurred
            # patch should not veto a strong colour/shape match.
            total = max(0.75 * c + 0.25 * g, 0.5 * c + 0.15 * g + 0.35 * k)
        return ScoreBreakdown(float(total), c, g, k)

    def update_online(self, patch_bgr: np.ndarray, rate: float) -> None:
        hs, v = color_histograms(patch_bgr)
        self.online_hs = (1 - rate) * self.online_hs + rate * hs
        self.online_v = (1 - rate) * self.online_v + rate * v

    def reset_online(self) -> None:
        self.online_hs = self.hs.copy()
        self.online_v = self.v.copy()
