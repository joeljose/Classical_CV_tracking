# Literature survey

The task is narrow. We get a fixed highway camera, about one minute of 1080p
video at 50 fps, and two cropped photos of the cars we care about (a silver
Audi and a blue Toyota). We have to report where each car is in every frame,
without deep learning. This note goes through the classical methods that
could do parts of that job, what each one assumes, and why some made it into
the tracker and others did not.

The problem splits into three questions, and the sections follow them.

1. What in this frame is a vehicle? (detection)
2. Which vehicle is the one in the photo? (identification)
3. Where did it go since the last frame, including when it is hidden? (tracking)

## 1. Detection

### Background subtraction

With a static camera the cheapest vehicle detector is "whatever is not
road". Stauffer and Grimson [1] model each pixel as a mixture of Gaussians
and call a pixel foreground when it does not fit the mixture. Zivkovic [2, 3]
made the number of components adaptive and added a shadow test, which is
the MOG2 subtractor in OpenCV. KaewTraKulPong and Bowden [4] proposed a
similar shadow-aware variant. ViBe [5] keeps a set of past samples per pixel
instead of a parametric model, and SuBSENSE [6] adds texture features. The
CDnet benchmark [7] compares many of these and shows that on a static
highway scene with good light, the simple methods are already close to the
best.

Background subtraction fails in known ways. Cars that touch in the image
come out as one blob, shadows get attached to the car unless removed, and a
car that stops slowly fades into the background. On our clip the first two
matter a lot: traffic is dense and the cars travel in two lanes close to
each other, so the blob is often two or three cars.

Frame differencing, the absolute difference of consecutive frames, reacts
faster than a background model but only fires on the moving edges of an
object. It is useful as a second signal near existing foreground and noisy
on its own.

### Sliding-window detectors

Viola and Jones [8] showed that boosted Haar features with a cascade can
find objects in real time. Dalal and Triggs [9] replaced Haar features with
histograms of oriented gradients (HOG) and a linear SVM, and Felzenszwalb et
al. [10] extended that to deformable part models. These all need a trained
classifier and a labelled training set of vehicles. We have two images, so
we cannot train one properly. HOG itself is still a good shape descriptor
and we use it to compare a candidate window with the reference photo.

### Optical flow

Lucas and Kanade [11] and Horn and Schunck [12] estimate per-pixel motion.
Clustering flow vectors can separate two cars in one blob if they move at
different speeds. On a highway the cars in adjacent lanes move at nearly the
same speed and direction, so flow does not split them well here, and dense
flow at 1080p is slow on a CPU.

## 2. Identification against a reference photo

This is the part that makes the task different from ordinary multi-object
tracking. Most of the image is other cars, several of them silver or blue.

### Colour

Swain and Ballard [13] introduced colour histograms for recognition, and
histogram comparison with the Bhattacharyya coefficient is the core of
mean-shift tracking [14]. Hue and saturation separate a blue car from a red
one easily. They do not separate silver, grey and white cars, which all have
near-zero saturation and differ mainly in brightness. That is why our model
keeps a separate value (brightness) histogram and gives it real weight.
Colour also changes with distance and lighting. The tracker therefore
keeps a short gallery of how each car looked recently, for association,
but checks identity against the fixed reference. Updating an appearance
model from the tracker's own output is the template update problem
described by Matthews et al. [37]: update too eagerly and the model drifts
onto whatever it last matched, never update and it stops fitting the
object. Their fix is to keep the first template as an anchor and accept an
update only when it is consistent with it. The gallery follows the same
idea, with the reference photo as the anchor.

### Shape and texture

HOG [9] gives a coarse shape signature that survives blur and small scale
changes. It tells a hatchback from a saloon but not one saloon from another.

### Local features

SIFT [15] and its faster cousins SURF [16] and ORB [17] match distinctive
points between two images, and RANSAC [18] keeps only matches that agree on
one geometric transform. When the car is large enough, this is the strongest
evidence of identity we have: the Audi grille and badge give stable
keypoints. When the car is far away (under about 60 pixels wide) there are
too few keypoints, so this cue only adds to the score and never vetoes.

### Template matching

Normalised cross-correlation (NCC) [19] compares raw pixels. It is fragile
across scale and lighting, so it is a poor detector for the reference photo,
but it works well over a few frames to follow a patch we have just seen. We
use it only as a short-range fallback around the predicted position.

## 3. Tracking

### Motion models

The Kalman filter [20] with a constant-velocity model is the standard choice
for vehicles on a straight road. It gives a predicted box each frame, which
limits where to look and carries the track through short gaps. Particle
filters such as CONDENSATION [21] handle multimodal uncertainty better, but
our motion is close to linear and the extra cost is not worth it.

### Single-object appearance trackers

Mean shift [14] and CAMShift [22] climb a colour-similarity surface. Online
learning trackers like online boosting [23], MIL [24] and TLD [25] learn the
object as they go. Correlation-filter trackers, MOSSE [26], KCF [27] and
CSRT [28], are fast and accurate on benchmarks like OTB [29]. All of these
need an initial bounding box in the first frame. We are not allowed to give
one by hand, and their online update can drift onto a neighbouring car of
the same colour during an occlusion, which is exactly the failure we most
want to avoid. Our tracker borrows their ideas (local search, online
appearance) but always re-checks identity against the reference photo.

### Multi-object tracking by detection

SORT [30] associates detections to Kalman-predicted tracks with the
Hungarian algorithm [31] on IoU. DeepSORT [32] adds a learned appearance
embedding, and ByteTrack [33] and BoT-SORT [34] keep low-confidence
detections and add camera-motion compensation. The structure of SORT fits
our problem well. We replace the learned embedding with the hand-built
appearance score, and we only have two tracks with known identities, so the
assignment is solved by checking every combination instead of with the
Hungarian algorithm.

### Occlusion

A hidden car must not get a position. Many trackers coast on the prediction
and keep drawing a box, which looks fine in a demo and is wrong in a CSV.
The approach here follows the tracking-by-detection literature: report
"occluded" with no box while the Kalman prediction is used internally to
pick the car up again, and drop the track after a fixed number of frames.

## 4. Evaluation

CLEAR MOT [35] defines MOTA and MOTP, and HOTA [36] balances detection and
association. For two known targets the simpler per-target numbers are
easier to read: recall at IoU 0.5 on frames where the car is visible,
precision of the boxes we did emit, identity swaps, and boxes emitted while
the car was hidden. That is what `tools/evaluate.py` reports.

Hand-labelling 3000 frames for two cars was not worth it for this exercise.
Instead we build pseudo ground truth with a YOLOv8s detector (TensorRT) and
an IoU linker, pick the two target tracks by appearance and check them by
eye. A deep detector is not perfect either, so these numbers are a
comparison against a strong baseline, not against truth. The deep model is
used only for this comparison and never feeds the tracker.

## References

1. C. Stauffer, W. E. L. Grimson. Adaptive background mixture models for real-time tracking. CVPR 1999.
2. Z. Zivkovic. Improved adaptive Gaussian mixture model for background subtraction. ICPR 2004.
3. Z. Zivkovic, F. van der Heijden. Efficient adaptive density estimation per image pixel for the task of background subtraction. Pattern Recognition Letters 27(7), 2006.
4. P. KaewTraKulPong, R. Bowden. An improved adaptive background mixture model for real-time tracking with shadow detection. AVBS 2001.
5. O. Barnich, M. Van Droogenbroeck. ViBe: a universal background subtraction algorithm for video sequences. IEEE TIP 20(6), 2011.
6. P.-L. St-Charles, G.-A. Bilodeau, R. Bergevin. SuBSENSE: a universal change detection method with local adaptive sensitivity. IEEE TIP 24(1), 2015.
7. Y. Wang et al. CDnet 2014: an expanded change detection benchmark dataset. CVPR Workshops 2014.
8. P. Viola, M. Jones. Rapid object detection using a boosted cascade of simple features. CVPR 2001.
9. N. Dalal, B. Triggs. Histograms of oriented gradients for human detection. CVPR 2005.
10. P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan. Object detection with discriminatively trained part-based models. IEEE TPAMI 32(9), 2010.
11. B. D. Lucas, T. Kanade. An iterative image registration technique with an application to stereo vision. IJCAI 1981.
12. B. K. P. Horn, B. G. Schunck. Determining optical flow. Artificial Intelligence 17, 1981.
13. M. J. Swain, D. H. Ballard. Color indexing. IJCV 7(1), 1991.
14. D. Comaniciu, V. Ramesh, P. Meer. Kernel-based object tracking. IEEE TPAMI 25(5), 2003.
15. D. G. Lowe. Distinctive image features from scale-invariant keypoints. IJCV 60(2), 2004.
16. H. Bay, T. Tuytelaars, L. Van Gool. SURF: speeded up robust features. ECCV 2006.
17. E. Rublee, V. Rabaud, K. Konolige, G. Bradski. ORB: an efficient alternative to SIFT or SURF. ICCV 2011.
18. M. A. Fischler, R. C. Bolles. Random sample consensus. Communications of the ACM 24(6), 1981.
19. J. P. Lewis. Fast normalized cross-correlation. Vision Interface 1995.
20. R. E. Kalman. A new approach to linear filtering and prediction problems. Journal of Basic Engineering 82(1), 1960.
21. M. Isard, A. Blake. CONDENSATION: conditional density propagation for visual tracking. IJCV 29(1), 1998.
22. G. R. Bradski. Computer vision face tracking for use in a perceptual user interface. Intel Technology Journal, 1998.
23. H. Grabner, M. Grabner, H. Bischof. Real-time tracking via on-line boosting. BMVC 2006.
24. B. Babenko, M.-H. Yang, S. Belongie. Visual tracking with online multiple instance learning. CVPR 2009.
25. Z. Kalal, K. Mikolajczyk, J. Matas. Tracking-learning-detection. IEEE TPAMI 34(7), 2012.
26. D. S. Bolme, J. R. Beveridge, B. A. Draper, Y. M. Lui. Visual object tracking using adaptive correlation filters. CVPR 2010.
27. J. F. Henriques, R. Caseiro, P. Martins, J. Batista. High-speed tracking with kernelized correlation filters. IEEE TPAMI 37(3), 2015.
28. A. Lukezic, T. Vojir, L. Cehovin Zajc, J. Matas, M. Kristan. Discriminative correlation filter with channel and spatial reliability. CVPR 2017.
29. Y. Wu, J. Lim, M.-H. Yang. Online object tracking: a benchmark. CVPR 2013.
30. A. Bewley, Z. Ge, L. Ott, F. Ramos, B. Upcroft. Simple online and realtime tracking. ICIP 2016.
31. H. W. Kuhn. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly 2, 1955.
32. N. Wojke, A. Bewley, D. Paulus. Simple online and realtime tracking with a deep association metric. ICIP 2017.
33. Y. Zhang et al. ByteTrack: multi-object tracking by associating every detection box. ECCV 2022.
34. N. Aharon, R. Orfaig, B.-Z. Bobrovsky. BoT-SORT: robust associations multi-pedestrian tracking. arXiv:2206.14651, 2022.
35. K. Bernardin, R. Stiefelhagen. Evaluating multiple object tracking performance: the CLEAR MOT metrics. EURASIP JIVP 2008.
36. J. Luiten et al. HOTA: a higher order metric for evaluating multi-object tracking. IJCV 129, 2021.
37. I. Matthews, T. Ishikawa, S. Baker. The template update problem. IEEE TPAMI 26(6), 2004.
