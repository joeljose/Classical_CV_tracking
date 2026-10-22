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

E4 is the default now. E5 shows the identity floor is not the limit any
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

On the synthetic clip the same defaults also do better than the old ones:
Audi recall 0.70 to 0.74, Toyota 0.86 to 0.995, still with no swaps and no
boxes while hidden.

## Reproducing

```bash
make gt-all                                   # ground truth (GPU)
make track OUT=output/e4                      # current defaults
make track OUT=output/e1 ARGS="--forward-only --identity-threshold 0.72"
make track OUT=output/baseline ARGS="--forward-only --no-reacquire --acquire-threshold 0.66 \
    --acquire-min-width 0 --confirm-frames 3 --identity-threshold 0.594"
make eval OUT=output/e4
```
