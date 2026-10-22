from track_vehicles import merge
from vtrack import FrameResult, Status


def r(status, box=None, score=None):
    return FrameResult(0, "car", status, box, score)


def test_merge_prefers_measured_then_higher_score_then_occluded():
    fwd = [[r(Status.SEARCHING)], [r(Status.TRACKED, (1, 1, 5, 5), 0.7)], [r(Status.LOST)]]
    bwd = [[r(Status.TRACKED, (2, 2, 5, 5), 0.8)], [r(Status.TRACKED, (3, 3, 5, 5), 0.9)],
           [r(Status.OCCLUDED)]]
    out = [row[0] for row in merge(fwd, bwd)]
    assert out[0].box == (2, 2, 5, 5)
    assert out[1].box == (3, 3, 5, 5)
    assert out[2].status == Status.OCCLUDED and out[2].box is None
