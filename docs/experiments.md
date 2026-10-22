# Experiments on the real clip

All numbers here are on `1900-151662242.mp4` (3000 frames, 1920x1080, 50
fps), scored against the pseudo ground truth described in `design.md`. The
ground truth comes from a deep detector and is used only to measure the
classical tracker. Nothing in the tracker reads it.

Four numbers are reported per target. Recall is the share of frames where
the ground-truth box is at least 25 px wide and our box overlaps it with
IoU 0.5 or more. Precision is the share of the boxes we emitted that overlap
the ground truth with IoU 0.3 or more. Swaps counts frames where our box
overlaps the other target's ground truth more than our own. Hidden counts
boxes emitted in frames where the ground truth has no box.

The Audi is in view for 499 frames (724 to 1222), the Toyota for 608 frames
(790 to 1397).

## Baseline: settings tuned on the synthetic clip

The first version of the tracker was tuned on a synthetic clip made by
pasting the reference photos onto a drawn road. On the real clip it scored
zero.

| run | Audi recall | Audi precision | Toyota recall | Toyota precision | swaps | hidden |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 355 |

What happened: in the first second, a silver car and a blue car near the
camera scored above the acquisition threshold of 0.66 for three frames in a
row. The tracker took them as the targets, followed them out of the bottom
of the frame, and marked both targets `exited`. By design an exited target
is never picked up again, so the real Audi and Toyota, which show up 15
seconds later, were never looked for.

Two settings were wrong for real footage. The threshold was too low, and
"never re-acquire after exit" assumed the first acquisition is always right.

## How well do the cues separate the targets?

With `--debug-candidates` the tracker writes every candidate box and its
score against both references. Joining those with the ground truth gives
the score of the true car against the best look-alikes, by box width in
original pixels:

| width | Audi: true car (median / 10th pct) | Audi: other cars (99th pct / max) | Toyota: true car | Toyota: other cars |
| --- | --- | --- | --- | --- |
| 40 to 80 | 0.61 / 0.58 | 0.66 / 0.69 | 0.59 / 0.57 | 0.64 / 0.67 |
| 80 to 160 | 0.65 / 0.61 | 0.71 / 0.73 | 0.65 / 0.60 | 0.69 / 0.80 |
| 160 and up | 0.83 / 0.79 | 0.78 / 0.84 | 0.84 / 0.80 | 0.73 / 0.81 |

Below 160 px the true car is not separable from other cars of the same
colour by appearance alone. Above 160 px it mostly is. So a car can only be
identified once it is close, and after that it has to be followed by motion
and local appearance.

## Changes

1. Acquisition only on candidates at least 80 processing pixels wide (160
   original), with score 0.80 or more for 10 frames in a row. Re-acquisition
   after exit is allowed, so one bad acquisition does not end the run.
2. A backward pass. Cars drive toward the camera, so in the clip played
   backwards they start large, where they are easy to identify, and shrink
   toward the horizon. The tracker runs once forwards and once backwards,
   and for every frame keeps whichever pass measured the target (the higher
   score if both did).
3. The identity floor (the score below which a match counts as suspicious)
   was tied to the acquisition threshold as `0.9 * acquire_threshold`.
   Raising the threshold to 0.80 moved the floor to 0.72, above what the
   true car scores at medium size, so tracks were dropped as the car shrank.
   It is now its own setting, `--identity-threshold`.

## Results

| run | change | Audi recall | Audi precision | Toyota recall | Toyota precision | swaps | hidden |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | synthetic-clip settings | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 355 |
| E1 | strict acquisition, forward only | 0.44 | 0.93 | 0.39 | 0.79 | 0 | 0 |
| E2 | E1 + backward pass | 0.65 | 1.00 | 0.60 | 0.99 | 0 | 0 |
| E3 | E2 with threshold 0.76 | 0.57 | 0.22 | 0.56 | 0.97 | 0 | 1262 |
| E4 | E2 + separate identity floor (0.59) | 0.66 | 0.98 | 0.61 | 0.97 | 0 | 7 |
| E5 | E4 with identity floor 0.55 | 0.66 | 0.98 | 0.61 | 0.97 | 0 | 7 |
| E6 | E4 with `--max-coast 50` | 0.65 | 0.96 | 0.61 | 0.97 | 0 | 7 |
| E7 | E4, drop raw blobs whose w/h is more than 1.35x off the reference | 0.69 | 0.84 | 0.61 | 0.97 | 0 | 64 |
| E8 | same, tolerance 1.2 | 0.69 | 0.82 | 0.54 | 0.89 | 32 | 63 |
| E9 | same, refined windows only | 0.55 | 0.79 | 0.63 | 0.97 | 0 | 68 |

E4 became the default at this stage. E5 shows the identity floor is not the limit any
more. E7 to E9 tried to fix loose boxes on merged blobs by refusing raw
blobs that are not car-shaped. Recall went up a little for the Audi, but
the tracker then picked up another silver car for a while, and with a
tighter tolerance the Toyota swapped onto the Audi for 32 frames. That
change was dropped.

Where E4 still misses, by ground-truth box width (original pixels):

| width | Audi: hit / loose / no box | Toyota: hit / loose / no box |
| --- | --- | --- |
| 50 to 100 | 0 / 0 / 26 | 0 / 0 / 39 |
| 100 to 200 | 159 / 46 / 66 | 180 / 96 / 98 |
| 200 and up | 169 / 33 / 0 | 191 / 4 / 0 |

"Loose" means a box was emitted but its IoU with the ground truth is
between 0.3 and 0.5. Almost all of those are frames where the Audi and the
Toyota drive side by side and their foreground merges, so the box is too
wide or too short. Frames with no box are the small, far-away part of
each car's pass, where appearance cannot separate it from other cars of
the same colour (see the score table above). Once a car is 200 px wide it
is found in every frame.

On the synthetic clip the E4 settings also did better than the old ones:
Audi recall 0.70 to 0.74, Toyota 0.86 to 0.995, still with no swaps and no
boxes while hidden.

## Speed: full detection every N frames

E4 ran at about 4 fps for one pass, so about 2 fps with both passes. A
profile of the densest stretch (frames 900 to 1200) put 59 percent of the
time in HOG, 25 percent in SIFT and under 3 percent in MOG2. HOG was slow
because the window search scores about 600 windows per frame.

Two changes went in.

1. HOG now builds its cell histograms with one `np.bincount` call instead
   of a one-hot array. Same output (largest difference 2e-8), 195 to 86
   microseconds per patch. OpenCV's own `HOGDescriptor` was the first
   choice, but OpenCV 5.0 moved it out of the main package.
2. `--detect-every N` runs the full pipeline only every Nth frame. At 50
   fps a car moves a few pixels per frame, so in between each track is
   followed by a template match near its Kalman prediction.

The first version of the cheap frames drifted badly. Each problem it showed
had a specific cause, and fixing it also helped the every-frame mode:

| problem seen | cause | fix |
| --- | --- | --- |
| box turned into a thin strip (30x172, 398x62) | the filter's width and height drift apart when only template matches correct it | predictions are always car-shaped: width from the filter, height from the reference's aspect ratio |
| Toyota slid onto the Audi next to it | nothing checked identity between detections | a followed box that looks more like the other reference (colour) sends the frame to full detection |
| track sat on the road at the horizon for 400 frames (backward pass) | NCC matches static background very well | a followed box must be at least 30 percent moving foreground |
| false start at frame 0 with N=10 | acquisition needed only 2 detection frames | every frame runs the full pipeline while an acquisition is pending |
| Audi box jumped to a neighbour as the Audi left the frame | while the car is partly out, the car-shaped window slides up onto the next car | once the box reaches the bottom edge, the car gets as many frames as it needs to be half out at its current speed, then it is `exited` |
| box stopped growing between detections (synthetic clip, 25 fps) | template matched at one size only | template match at three scales |

Results after these changes (same ground truth as above):

| mode | N | Audi recall | Audi precision | Toyota recall | Toyota precision | swaps | hidden |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward only | 1 | 0.55 | 0.89 | 0.53 | 1.00 | 0 | 21 |
| forward only | 5 | 0.55 | 0.99 | 0.49 | 1.00 | 0 | 0 |
| forward only | 10 | 0.46 | 0.80 | 0.52 | 1.00 | 0 | 54 |
| two passes | 1 | 0.89 | 0.93 | 0.86 | 0.97 | 0 | 21 |
| two passes | 5 | 0.89 | 0.96 | 0.81 | 0.96 | 0 | 0 |
| two passes | 10 | 0.89 | 0.87 | 0.85 | 1.00 | 0 | 54 |

Car-shaped predictions are what lifted two-pass recall from about 0.65 to
about 0.85. N=5 is the only setting with no boxes on a wrong car in either
mode, so it is the default. At N=1 and N=10 the Audi track still slides
onto another silver car for a short stretch (21 and 54 frames). On the
synthetic clip N=5 gives Audi 0.65 / 0.95 and Toyota 1.00 / 1.00 (recall /
precision), with no swaps and no boxes while hidden.

Timing, tracking only, one run at a time. Other work was sometimes running
on the machine, and these were measured just before the last two fixes
(exit countdown, three-scale template), which add a little work on the
cheap frames. Treat them as approximate.

| mode | N=1 | N=5 | N=10 |
| --- | --- | --- | --- |
| forward only | 6.2 fps | 23.3 fps | 39.1 fps |
| two passes | 3.2 fps | 11.5 fps | 17.6 fps |

The source is 50 fps. Forward-only at N=5 keeps up with a 20 to 25 fps
stream. Two passes cannot run live at any speed, because the backward pass
starts from the last frame.

## Reproducing

```bash
make gt-all                                          # ground truth (GPU)
make track OUT=output/final                          # current defaults
make track OUT=output/n1 ARGS="--detect-every 1"     # full detection every frame
make track OUT=output/fwd ARGS="--forward-only"      # live-capable mode
make eval OUT=output/final
```

The earlier runs (baseline, E1 to E9) used earlier versions of the code,
and their numbers will not come back exactly from flags on the current
version. The first commit of this repository holds the E4 version.
