# Design

## Problem

Input is a one-minute 1080p clip from a fixed camera over a two-lane road,
3000 frames at 50 fps, plus two cropped photos: a silver Audi A4 and a blue
Toyota Aygo, both seen from the front. Output is a CSV with the position of
each car in every frame and an MP4 with the cars highlighted. Deep learning
may not be used to find or follow the cars.

Some facts about the clip shaped the design:

- The camera does not move, so background subtraction is cheap and reliable.
- Cars drive toward the camera. They appear near the horizon at 15 to 20
  pixels wide and leave through the bottom edge at 300 pixels or more. Scale
  changes by more than ten times during one pass.
- Traffic is dense. The Audi and the Toyota drive in adjacent lanes, close
  together, for most of their time in view. Background subtraction often
  returns the two of them, plus a neighbour, as one blob.
- Many other cars share the targets' colours. There are at least four other
  silver saloons and hatchbacks, and dark blue cars that look like the
  Toyota when small.
- Each target is in view for about 10 to 12 seconds (500 to 600 frames).

## Requirements we set ourselves

1. Never give a box for a car that is not visible. A wrong box in the CSV is
   worse than an empty row, because the user cannot tell it is wrong.
2. Never swap identities. If the tracker is unsure which silver car is the
   Audi, it should say "searching", not guess.
3. Nothing is selected by hand in the video. The only target description is
   the reference photo.
4. Run on a laptop CPU with OpenCV and NumPy only. Offline at a useful
   speed by default, and fast enough for a live 20 to 25 fps stream in
   forward-only mode.

## Architecture

```
frame ─► MOG2 + frame differencing ─► blobs ─► window refinement ─► proposals
                                                                      │
 references ─► appearance models (HSV hist, HOG, SIFT) ─► score every proposal
                                                                      │
 Kalman prediction ─► gated joint association ─► template fallback ─► status/box
                                     │
              inactive targets ─► acquisition with N-frame confirmation
```

Frames wider than `--process-width` (960 by default) are downscaled for
processing. Boxes are scaled back to original pixels on output.

### Proposals (`vtrack/detection.py`, `vtrack/tracker.py:_proposals`)

MOG2 with shadow detection gives the foreground. Shadow pixels (value 127)
are dropped. Frame differencing is added only where it touches existing
foreground, because on its own it fires on the edges of every moving thing.
An open and a close clean the mask, and connected components give blobs.

A blob is often several cars. For each blob and each target, the tracker
slides a window with the reference's aspect ratio over the blob at a few
scales and keeps the window with the best cheap score (colour plus HOG).
Active tracks also search a small area around their Kalman prediction at the
predicted size. The proposals for a frame are the raw blobs plus these
refined windows.

### Appearance score (`vtrack/appearance.py`)

Each proposal is scored against each reference photo. The score combines:

- Colour. A hue-saturation histogram and a separate value histogram over the
  central part of the box, compared with the Bhattacharyya coefficient. The
  value histogram has 40 percent of the colour weight, because silver, grey
  and white cars only differ in brightness.
- Shape. A HOG vector in the Dalal and Triggs layout, compared by cosine
  similarity.
- Keypoints, when the patch is at least 60 pixels wide. SIFT matches with
  the ratio test, then the inlier count of a RANSAC similarity transform.
  This term can raise the score but never lower it, because a blurred patch
  has few keypoints even when it is the right car.

### Association

Candidates are gated around each track's prediction by centre distance and
by size ratio. With two targets there are few enough combinations to try all
of them and pick the best total score, so two targets never share a box. A
candidate is refused if it scores higher against the other reference, or if
its score falls well below the track's running average. That second test is
the main defence against latching onto a look-alike when the real target is
hidden.

### Frames between full detections

The full pipeline runs every `--detect-every` frames (5 by default). On the
frames in between, MOG2 still updates, and each active track is followed by
a normalised cross-correlation search near its prediction at three scales.
The match is used only if it is at least 30 percent moving foreground and
does not look more like the other reference by colour. If any track fails
those checks, or a car is leaving the frame, or an acquisition is in
progress, that frame runs the full pipeline instead.

Predictions are kept car-shaped everywhere: the width comes from the Kalman
filter and the height from the reference's aspect ratio. Without that, the
filter's width and height drift apart when only template matches correct
them, and the gates and searches then look in the wrong place.

### Occlusion and loss

When no candidate is associated, a normalised cross-correlation search runs
in a small window around the prediction, using the last confirmed patch as
the template. If it finds a match that still scores well against the
reference, the frame is `tracked_template`. Otherwise the frame is
`occluded` and no box is written. After `--max-coast` frames without a
measurement the track becomes `lost` and acquisition starts again.

A car leaves through the bottom of the frame. Once its box reaches the
bottom edge it gets as many frames as it needs, at its current speed, to be
half out of view, and then it is `exited`. Waiting longer is risky: with
the car partly out, a car-shaped window slides up onto whatever is next to
it. The exit check runs on the prediction before association, so a car
that is leaving cannot hand its identity to a neighbour. A target can be
acquired again after it exits, so one wrong early acquisition does not end
the run.

### Acquisition

An inactive target is acquired only when the same candidate scores above
`--acquire-threshold` for `--confirm-frames` frames in a row at a consistent
location, and beats the other reference by a margin. The confirming frames
are back-filled as `tracked`, since they were real matches.

### Two passes (`track_vehicles.py`)

A car can only be identified once it is close: below about 160 original
pixels wide, the true car's score overlaps with other cars of the same
colour (see `experiments.md`). Cars drive toward the camera, so in the
forward direction the far-away part of each pass is lost. The clip is
therefore tracked twice, forwards and then backwards over the same decoded
frames, and the results are merged per frame. In the backward pass each car
is acquired while it is large and followed back toward the horizon. Where
both passes measured the car, the higher-scoring box is kept. Frames are
decoded once, downscaled and kept JPEG-encoded in memory, so the backward
pass does not have to seek in the video.

This makes the default mode offline. `--forward-only` skips the backward
pass for live use.

## Ground truth and evaluation

No labels came with the clip, and hand-labelling 3000 frames was not worth
it. For comparison only, `tools/detect_trt.py` runs a COCO-trained YOLOv8s
through TensorRT on every frame and keeps car, motorcycle, bus and truck
boxes. `tools/build_gt.py` links those boxes into tracklets by IoU, scores
each tracklet against the reference photos, picks the best one per target,
and writes a contact sheet so a person can check the choice. On this clip
the picks were checked by eye and are correct: the Audi is one tracklet from
frame 724 to 1222 and the Toyota one tracklet from frame 790 to 1397.

The ground truth runs in its own GPU container (`docker/Dockerfile.gt`,
plain CUDA runtime plus pip TensorRT). The classical tracker never reads it.

`tools/evaluate.py` reports, per target, recall at IoU 0.5 over frames where
the ground-truth box is at least 25 pixels wide, precision at IoU 0.3 over
the boxes the tracker emitted, identity swaps (frames where our box overlaps
the other target's ground truth more than our own), and boxes emitted where
the ground truth has none.

## Alternatives considered

- OpenCV's built-in single-object trackers (KCF, CSRT, MOSSE). They need a
  starting box, which we are not allowed to draw, and they drift onto
  neighbours of the same colour during occlusions. See `literature.md`.
- A trained HOG plus SVM vehicle detector. Two images are not a training
  set, and the background subtractor already finds moving cars.
- Dense optical flow to split merged blobs. Cars in adjacent lanes move at
  almost the same velocity, so flow does not separate them, and it is slow
  at 1080p.
- Coasting on the Kalman prediction during occlusion and drawing the box.
  It looks smooth, but it breaks requirement 1.
