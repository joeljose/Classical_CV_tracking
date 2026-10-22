import cv2
import numpy as np
import pytest

from vtrack.appearance import AppearanceModel, hog_vector
from vtrack.detection import iou
from vtrack.kalman import BoxKalman

AUDI = cv2.imread("refs/1000073924.png")
TOYOTA = cv2.imread("refs/1000073925.png")


@pytest.fixture(scope="module")
def models():
    return AppearanceModel("audi", AUDI), AppearanceModel("toyota", TOYOTA)


def test_reference_matches_itself_best(models):
    audi, toyota = models
    assert audi.score(AUDI).total > toyota.score(AUDI).total + 0.1
    assert toyota.score(TOYOTA).total > audi.score(TOYOTA).total + 0.1


@pytest.mark.parametrize("width", [40, 80, 160])
def test_scaled_reference_still_identified(models, width):
    audi, toyota = models
    for img, own, other in ((AUDI, audi, toyota), (TOYOTA, toyota, audi)):
        small = cv2.resize(img, (width, int(width * img.shape[0] / img.shape[1])),
                           interpolation=cv2.INTER_AREA)
        assert own.score(small).total > other.score(small).total


def test_keypoints_only_for_large_patches(models):
    audi, _ = models
    assert audi.score(cv2.resize(AUDI, (40, 28))).keypoints is None
    assert audi.score(AUDI).keypoints == pytest.approx(1.0)


def test_hog_is_unit_length():
    assert np.linalg.norm(hog_vector(AUDI)) == pytest.approx(1.0, abs=1e-5)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_kalman_follows_constant_velocity():
    kf = BoxKalman((0, 0, 20, 10))
    for k in range(1, 15):
        kf.predict()
        kf.correct((3 * k, 2 * k, 20, 10))
    x, y, w, h = kf.predict()
    assert abs(x - 45) <= 2 and abs(y - 30) <= 2 and (w, h) == (20, 10)
