"""Classical-CV vehicle tracking from reference images."""

from .appearance import AppearanceModel
from .detection import DetectorConfig, MotionDetector
from .tracker import FrameResult, Status, TrackerConfig, VehicleTracker

__all__ = [
    "AppearanceModel", "DetectorConfig", "MotionDetector",
    "FrameResult", "Status", "TrackerConfig", "VehicleTracker",
]
