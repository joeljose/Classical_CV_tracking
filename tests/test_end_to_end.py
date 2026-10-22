"""Run the full CLI on a synthetic clip rendered from the reference images.

The clip has a look-alike distractor (the Audi darkened) and a vehicle that
fully occludes the Audi for ~30 frames. This guards the behaviour, not the
accuracy on real footage.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from evaluate import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    subprocess.run([sys.executable, "tools/make_synthetic_video.py", "--out", d / "syn.mp4",
                    "--gt", d / "gt.json"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "track_vehicles.py", "--video", d / "syn.mp4",
                    "--out-dir", d / "out", "--no-video"], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    return d, evaluate(str(d / "out" / "tracks.csv"), str(d / "gt.json"))


def test_no_identity_swaps(run):
    _, report = run
    for name, r in report.items():
        assert r["identity_swaps"] == 0, name


def test_no_positions_while_hidden(run):
    _, report = run
    for name, r in report.items():
        assert r["positions_while_hidden"] == 0, name


def test_reported_positions_are_on_target(run):
    _, report = run
    for name, r in report.items():
        assert r["precision_iou30"] >= 0.95, name


def test_targets_are_tracked_most_of_the_time(run):
    _, report = run
    for name, r in report.items():
        assert r["recall_iou50"] >= 0.6, name


def test_every_frame_has_a_row_per_target(run):
    d, _ = run
    summary = json.loads((d / "out" / "summary.json").read_text())
    for t in summary["targets"].values():
        assert sum(t["status_counts"].values()) == summary["frames"]
