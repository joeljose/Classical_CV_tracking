# Classical computer-vision vehicle tracking

This tracks two specific cars through a fixed-camera road clip. The only
description of each car is one cropped photo. No box is drawn by hand in
the video, and no deep learning is used to find or follow the cars. The
pieces are all classical: MOG2 background subtraction, frame differencing,
colour histograms, HOG, SIFT with RANSAC, normalised cross-correlation and a
Kalman filter.

| Target | Reference image | Label in outputs |
| --- | --- | --- |
| A | `refs/1000073924.png` | `Silver Audi` |
| B | `refs/1000073925.png` | `Blue Toyota` |

![Tracker boxes (colour) and ground truth (white) on the real clip](docs/example.jpg)

*Real clip. Coloured boxes are the tracker, thin white boxes are the
detector-based ground truth used for evaluation.*

The input clip and the result videos are attached to the
[v1.0 release](https://github.com/joeljose/Classical_CV_tracking/releases/tag/v1.0):
`1900-151662242.mp4` (input), `annotated.mp4` (tracker output),
`compare.mp4` (tracker against ground truth) and `tracks.csv`.

## Quick start

Everything runs in Docker through `make`. The tracker image is CPU only.

```bash
make setup                      # build the image, create data/ and output/
cp /path/to/1900-151662242.mp4 data/
make track                      # -> output/run/
make test
```

`make help` lists every target. Without Docker:

```bash
pip install -r requirements.txt
python track_vehicles.py --video data/1900-151662242.mp4 --out-dir output/run
```

Outputs:

| File | Contents |
| --- | --- |
| `tracks.csv` | one row per frame per target (see below) |
| `annotated.mp4` | the video with each target boxed in its colour and its status |
| `summary.json` | status counts and first and last located frame per target |
| `candidates.csv` | with `--debug-candidates`: every candidate box and its score against each reference |

## The CSV

`tracks.csv` has one row for every frame and every target. The position
columns (`x, y, w, h, cx, cy, score`) are filled only when the target was
actually measured in that frame. Otherwise they are empty, and `status`
says why.

| `status` | Position? | Meaning |
| --- | --- | --- |
| `searching` | no | Not acquired yet. |
| `tracked` | yes | A moving-object candidate matched the reference and was associated with the track. |
| `tracked_template` | yes | No candidate matched, but a short-range NCC search near the prediction found the car and it still matches the reference. |
| `occluded` | no | The track is alive but there is no reliable measurement. The Kalman prediction is used internally and never written out. |
| `lost` | no | Unmeasured for more than `--max-coast` frames, or the matches kept scoring poorly. Acquisition starts again. |
| `exited` | no | The car drove out of view (its box reached the bottom edge and it is about half out, or the predicted box left the frame). |

Boxes are `x, y, w, h` in original-resolution pixels and `time_s = frame / fps`.

## How it works

The short version is below. `docs/design.md` has the full design and
`docs/literature.md` the survey of methods behind it.

1. Foreground. MOG2 with shadows removed, reinforced by frame differencing
   near existing foreground, then morphology and connected components.
2. Proposals. Neighbouring cars often merge into one blob, so each blob is
   searched for windows with the reference's aspect ratio that best match
   its colour and HOG. Active tracks also search around their Kalman
   prediction.
3. Appearance score against each reference. Hue-saturation and brightness
   histograms (brightness matters for silver cars), HOG, and for patches at
   least 60 px wide a SIFT match verified with RANSAC.
4. Association. Gated by distance and size around the prediction, solved
   jointly so the two targets never share a box. A candidate that looks
   more like the other reference, scores far below the track's usual
   score, or does not resemble the car's recent looks, is refused.
   Each track keeps a small gallery of its last 10 confident looks (colour
   and HOG). The reference photo decides identity; the gallery tells the
   car apart from a neighbour that fits the photo just as well.
5. Occlusion. With no candidate, a local NCC search runs around the
   prediction. If that fails the frame is `occluded` and no box is written.
6. Acquisition. A target starts only when a candidate at least 80 px wide
   (processing pixels) scores 0.80 or more against its reference for 10
   frames in a row, at a consistent place.
7. Two passes. The clip is tracked forwards and then backwards, and the
   two results are merged per frame. Cars drive toward the camera, so they
   can only be identified once they are close. Played backwards, each car
   is picked up while it is large and followed back to where it first
   appeared.
8. Full detection every 10th frame (`--detect-every`). In between, each
   track is followed by a template match at three scales near its Kalman
   prediction, checked against the foreground mask, the gallery and the
   other car's colours.
   Anything doubtful (a weak match, a car leaving the frame, an acquisition
   in progress) sends that frame through the full pipeline instead.

## Two passes, and why it needs a recorded video

By default the clip is tracked twice: once forwards, from the first frame
to the last, and once backwards, from the last frame to the first. The two
results are then merged frame by frame.

The reason is the direction the cars drive. They come toward the camera, so
each car first shows up small near the horizon and grows as it gets
closer. A car can only be told apart from other cars of the same colour
once it is big, roughly 160 px wide or more. Going forwards, the tracker
therefore only becomes sure it has the Audi late in the car's trip, and all
the small, far-away frames before that are lost. Going backwards, the same
car starts big at the bottom of the frame, is recognised straight away, and
is then followed as it shrinks back toward the horizon. That recovers the
early frames. For each frame the merge keeps whichever pass found the car,
or the higher-scoring box if both did.

This only works on a recorded video. The backward pass starts from the last
frame, so the whole clip has to exist before it can begin. For a live
camera use `--forward-only`, which processes frames as they arrive. It is
about twice as fast, but recall drops from about 0.87 to about 0.5 because
the far-away part of each car's trip is missed.

| | `--forward-only` | two passes (default) |
| --- | --- | --- |
| input | live stream or recorded video | recorded video only |
| recall on this clip | about 0.5 | about 0.87 |
| speed | about 2x faster | slower, both passes run |

## Results on the real clip

There are no labels for this clip, so I built ground truth for comparison
only. A COCO-trained YOLOv8s runs through TensorRT on every frame, the boxes
are linked into tracklets, and the tracklets that match the two reference
photos best are picked and checked by eye ([contact sheet](docs/gt_contact_sheet.jpg): reference photo first, then crops along each picked tracklet). The deep
model never feeds the tracker. It is also not perfect, so read these as
agreement with a strong detector rather than accuracy against hand labels.

| | recall (IoU 0.5) | precision (IoU 0.3) | identity swaps | boxes where GT has none |
| --- | --- | --- | --- | --- |
| Silver Audi (499 frames in view) | 0.87 | 1.00 | 0 | 0 |
| Blue Toyota (608 frames in view) | 0.88 | 1.00 | 0 | 0 |

The settings that came from tuning on a synthetic clip scored zero on the
real one. They acquired look-alike cars in the first second and never
recovered. `docs/experiments.md` walks through what went wrong, the score
analysis that fixed it, and every run.

Most of the remaining misses are at the far end of each pass, when the car
is under about 100 px wide and looks like several other cars of the same
colour. The tracker reports `searching` there instead of guessing.

## Speed

Measured on a 12-core laptop CPU at the default processing width of 960 px,
tracking only (decoding adds about 10 s for the clip). Other work was often
running on the machine, and these were measured before the gallery was
added (which adds a little work per candidate), so treat them as
approximate. Every run writes its own timing to `summary.json` under
`timing`.

| | full detection every frame | every 5th frame | every 10th frame (default) |
| --- | --- | --- | --- |
| forward only (`--forward-only`) | 6 fps | 23 fps | 39 fps |
| two passes (default) | 3 fps | 11 fps | 18 fps |

Forward-only can run on a live stream and, at the default setting, keeps up
with a 25 to 30 fps camera. Two passes need a recorded video (see above).
Details are in `docs/experiments.md`.

## Tuning on new footage

```bash
make track-debug OUT=output/debug
```

`annotated.mp4` then shows every candidate box with its score against each
reference, and `candidates.csv` has the same numbers. Set
`--acquire-threshold` below what the true cars reach when close and above
the best look-alikes. Other options are `--keep-threshold`,
`--identity-threshold`, `--acquire-min-width`, `--confirm-frames`,
`--max-coast`, `--min-area` and `--process-width` (frames wider than this
are downscaled, default 960). `--forward-only` skips the backward pass and
`--detect-every N` sets how often the full detection runs.

## Testing

`make test` runs unit tests and an end-to-end test on a synthetic clip
rendered from the two reference images, with a dark look-alike Audi, a red
hatchback and a full occlusion of the Audi. `make eval-synthetic` prints the
scores on that clip. The synthetic clip only guards behaviour. It is much
easier than real footage, as the baseline above shows.

## Layout

```
track_vehicles.py          CLI: video in, tracks.csv + annotated.mp4 + summary.json out
vtrack/appearance.py       reference appearance models (colour, HOG, SIFT)
vtrack/detection.py        MOG2 + differencing blobs, box helpers
vtrack/kalman.py           constant-velocity box Kalman filter
vtrack/tracker.py          proposals, association, occlusion handling, acquisition
tools/detect_trt.py        YOLOv8 + TensorRT detections (ground truth only)
tools/build_gt.py          link detections, pick the target tracklets
tools/evaluate.py          score tracks.csv against a ground-truth file
tools/compare_video.py     tracker vs ground truth video
tools/make_synthetic_video.py  synthetic test clip
docker/                    CPU image (tracker) and GPU image (ground truth)
docs/                      design, literature survey, experiments
refs/                      the two reference images
```

## Limitations

- A car only a few dozen pixels wide carries too little colour and shape to
  tell apart from similar cars. Those frames stay `searching`.
- When the two targets drive side by side their blob merges, and the box
  can come out loose (IoU between 0.3 and 0.5 with the ground truth).
- The default two-pass mode works on recorded video only and runs at
  roughly 18 fps.
  Live use needs `--forward-only`, which loses most of the early, far-away
  frames (recall about 0.5 instead of 0.85).
- A car is called `exited` once it is about half out of the bottom of the
  frame, so its last few partly visible frames are not reported.
- A car of very similar colour and shape, in the same place, during an
  occlusion could still be mistaken for a target. The relative-score check
  reduces this but does not remove it.
