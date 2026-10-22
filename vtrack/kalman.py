"""Constant-velocity Kalman filter over box centre and size."""

from __future__ import annotations

import cv2
import numpy as np

from .detection import Box


class BoxKalman:
    """State: [cx, cy, w, h, vcx, vcy, vw, vh]; measurement: [cx, cy, w, h]."""

    def __init__(self, box: Box):
        kf = cv2.KalmanFilter(8, 4)
        kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.transitionMatrix[i, i + 4] = 1.0
        kf.measurementMatrix = np.hstack([np.eye(4), np.zeros((4, 4))]).astype(np.float32)
        kf.processNoiseCov = np.diag([1, 1, 1, 1, 0.5, 0.5, 0.2, 0.2]).astype(np.float32)
        kf.measurementNoiseCov = np.diag([4, 4, 9, 9]).astype(np.float32)
        kf.errorCovPost = np.diag([10, 10, 10, 10, 100, 100, 25, 25]).astype(np.float32)
        kf.statePost = np.array([*self._to_meas(box), 0, 0, 0, 0], np.float32).reshape(8, 1)
        self.kf = kf

    @staticmethod
    def _to_meas(box: Box) -> np.ndarray:
        x, y, w, h = box
        return np.array([x + w / 2, y + h / 2, w, h], np.float32)

    @staticmethod
    def _to_box(state: np.ndarray) -> Box:
        cx, cy, w, h = (float(v) for v in state[:4].ravel())
        w, h = max(w, 2.0), max(h, 2.0)
        return int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(w)), int(round(h))

    def predict(self) -> Box:
        return self._to_box(self.kf.predict())

    def correct(self, box: Box) -> Box:
        return self._to_box(self.kf.correct(self._to_meas(box).reshape(4, 1)))

    @property
    def box(self) -> Box:
        return self._to_box(self.kf.statePost)
