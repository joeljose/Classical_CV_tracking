"""Reference-driven multi-target tracker.

Pipeline per frame:
  1. MOG2 + frame differencing -> blobs of moving objects.
  2. Each blob is refined into vehicle-shaped windows (one per target, found
     by a colour+HOG window search) so that blobs of several merged vehicles
     still yield a tight box per vehicle. Every proposal is then scored
     against every reference appearance model.
  3. Active targets are associated to candidates (Kalman-predicted motion gate
     + appearance), jointly so that one candidate never feeds two targets.
  4. Targets with no associated candidate try a local template search (NCC)
     around the prediction; if that also fails the frame is reported as
     occluded/lost with *no* position.
  5. Inactive targets are (re)acquired only after a candidate matches the
     reference for several consecutive frames at a consistent location.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .appearance import AppearanceModel
from .detection import Box, DetectorConfig, MotionDetector, clip_box, crop, iou
from .kalman import BoxKalman


class Status(str, Enum):
    SEARCHING = "searching"              # never acquired yet
    TRACKED = "tracked"                  # position from a matched detection
    TRACKED_TEMPLATE = "tracked_template"  # position from local NCC search
    OCCLUDED = "occluded"                # track alive, no reliable measurement
    LOST = "lost"                        # track dropped, searching again
    EXITED = "exited"                    # target left the frame

    @property
    def has_position(self) -> bool:
        return self in (Status.TRACKED, Status.TRACKED_TEMPLATE)


@dataclass
class TrackerConfig:
    acquire_threshold: float = 0.80   # reference score needed to start a track
    acquire_margin: float = 0.05      # must beat the other targets' score by this
    acquire_min_width: int = 80       # candidates narrower than this cannot start a track
    confirm_frames: int = 10          # consecutive consistent matches to acquire
    keep_threshold: float = 0.45      # reference score needed to keep associating
    identity_threshold: float = 0.59  # measurements below this count towards identity_patience
    identity_patience: int = 8        # consecutive low-score measurements -> drop
    score_drop_tolerance: float = 0.15  # reject matches this far below the track's usual score
    gate_factor: float = 0.6          # centre-distance gate, x max(w, h) of prediction
    size_ratio_gate: float = 1.35     # max linear size ratio, candidate vs prediction
    overlap_exclusion: float = 0.3    # IoU above which two targets cannot share boxes
    refine_scales: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.42)
    max_coast: int = 25               # frames a track may go unmeasured before LOST
    template_threshold: float = 0.65  # NCC needed for template fallback
    online_rate: float = 0.1          # online colour-model adaptation rate
    exit_margin: float = 0.35         # fraction of predicted box outside frame -> exited
    reacquire_after_exit: bool = True
    detect_every: int = 5             # full detection every N frames, template following in between
    detector: DetectorConfig = field(default_factory=DetectorConfig)


@dataclass
class FrameResult:
    frame: int
    target: str
    status: Status
    box: Box | None = None
    score: float | None = None


@dataclass
class _Target:
    model: AppearanceModel
    status: Status = Status.SEARCHING
    kf: BoxKalman | None = None
    missed: int = 0
    low_score_run: int = 0
    score_ema: float | None = None  # running mean of accepted reference scores
    template: np.ndarray | None = None
    pending: list[tuple[int, Box, float]] = field(default_factory=list)
    exit_in: int | None = None  # frames left once the box reached the bottom edge

    @property
    def name(self) -> str:
        return self.model.name

    @property
    def active(self) -> bool:
        return self.kf is not None


class VehicleTracker:
    def __init__(self, references: dict[str, np.ndarray], cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.targets = [_Target(AppearanceModel(n, img)) for n, img in references.items()]
        self.detector = MotionDetector(self.cfg.detector)
        self.frame_idx = -1
        # Results are kept per frame so acquisition can back-fill the frames
        # that confirmed it (those were real, matched detections).
        self.history: list[list[FrameResult]] = []
        self.last_candidates: list[tuple[Box, dict[str, float]]] = []

    # ---------------------------------------------------------------- public
    def process(self, frame: np.ndarray) -> list[FrameResult]:
        self.frame_idx += 1
        cfg = self.cfg
        # The filter tracks width and height separately and its size can drift
        # without a measurement; a car keeps its shape, so derive the height.
        predictions = {i: self._car_shaped(t.kf.predict(), t.model.aspect)
                       for i, t in enumerate(self.targets) if t.active}
        # A car leaving the frame is done before association, otherwise its
        # large gate can hand the identity to a neighbour that is still in view.
        exits = {}
        for i, pred in list(predictions.items()):
            if self._outside_fraction(pred, frame.shape) > cfg.exit_margin:
                self._drop(self.targets[i], Status.EXITED)
                exits[i] = FrameResult(self.frame_idx, self.targets[i].name, Status.EXITED)
                del predictions[i]
        if self.frame_idx % cfg.detect_every and not exits:
            out = self._follow(frame, predictions)
            if out is not None:
                self.history.append(out)
                return out
        blobs = [b for b in (clip_box(b, frame.shape) for b in self.detector.detect(frame)) if b]
        boxes = self._proposals(frame, blobs, predictions)
        patches = [crop(frame, b) for b in boxes]
        ref_scores = np.array([[t.model.score(p).total for p in patches] for t in self.targets]
                              ).reshape(len(self.targets), len(boxes))
        self.last_candidates = [
            (b, {t.name: float(ref_scores[i, j]) for i, t in enumerate(self.targets)})
            for j, b in enumerate(boxes)]

        results: dict[int, FrameResult] = dict(exits)

        # --- 1. associate active tracks --------------------------------------
        assignment = self._associate(predictions, boxes, patches, ref_scores)
        used = [boxes[j] for j in assignment.values()]
        for i, pred in predictions.items():
            t = self.targets[i]
            if i in assignment:
                j = assignment[i]
                results[i] = self._accept(t, frame, boxes[j], float(ref_scores[i, j]), Status.TRACKED)
                continue
            fallback = self._template_search(t, frame, pred)
            if fallback is not None:
                box, s = fallback
                results[i] = self._accept(t, frame, box, s, Status.TRACKED_TEMPLATE)
                continue
            results[i] = self._miss(t, frame.shape, pred)

        # --- 2. acquisition for inactive targets ------------------------------
        for i, t in enumerate(self.targets):
            if i in results:
                continue
            if t.status == Status.EXITED and not cfg.reacquire_after_exit:
                results[i] = FrameResult(self.frame_idx, t.name, Status.EXITED)
                continue
            results[i] = self._try_acquire(i, t, frame, boxes, ref_scores, used)
            if results[i].box is not None:
                used.append(results[i].box)

        out = [results[i] for i in range(len(self.targets))]
        self.history.append(out)
        return out

    def _follow(self, frame, predictions: dict[int, Box]) -> list[FrameResult] | None:
        """Cheap frame between full detections: keep the background model
        current and follow active tracks by template matching. Returns None
        when anything needs a full look (an acquisition in progress, a track
        leaving the frame, a weak or suspicious match), and the caller then
        runs the full pipeline on this frame."""
        if any(t.pending for t in self.targets):
            return None
        fg = self.detector.foreground_mask(frame)
        self.last_candidates = []
        found = {}
        for i, pred in predictions.items():
            t = self.targets[i]
            if self._outside_fraction(pred, frame.shape) > 0.1:
                return None
            hit = self._template_search(t, frame, pred, verify=False)
            if hit is None:
                return None
            # A template can lock onto static background (the road at the
            # horizon matches well); a real vehicle is moving foreground.
            if np.count_nonzero(crop(fg, hit[0])) < 0.3 * hit[0][2] * hit[0][3]:
                return None
            patch = crop(frame, hit[0])
            own = t.model.color_score(patch)
            if any(o.model.color_score(patch) > own for o in self.targets if o is not t):
                return None
            found[i] = hit[0]
        out = []
        for i, t in enumerate(self.targets):
            if i in found:
                t.kf.correct(found[i])
                t.missed = 0
                t.status = Status.TRACKED_TEMPLATE
                out.append(FrameResult(self.frame_idx, t.name, t.status, found[i], t.score_ema))
            else:
                out.append(FrameResult(self.frame_idx, t.name, t.status))
        return out

    # ------------------------------------------------------------- proposals
    def _proposals(self, frame, blobs: list[Box], predictions: dict[int, Box]) -> list[Box]:
        cands: list[Box | None] = []
        for blob in blobs:
            cands.append(blob)
            cands.extend(self._refine(frame, blob, t.model) for t in self.targets)
        # Active tracks also search near their prediction at the predicted
        # size, which localises a target inside a blob it shares with others
        # far better than the blob-relative scales above.
        for i, pred in predictions.items():
            if any(iou(pred, b) > 0.05 for b in blobs):
                cands.append(self._refine_near(frame, pred, self.targets[i].model))
        out: list[Box] = []
        for c in cands:
            c = clip_box(c, frame.shape) if c is not None else None
            if c is not None and all(iou(c, o) < 0.85 for o in out):
                out.append(c)
        return out

    def _refine_near(self, frame, pred: Box, model: AppearanceModel) -> Box | None:
        px, py, pw, ph = pred
        best, best_s = None, -1.0
        for s in (0.92, 1.0, 1.08):
            ww, wh = int(pw * s), int(ph * s)
            if ww < 20 or wh < 16:
                continue
            sx, sy = max(2, ww // 8), max(2, wh // 8)
            cx, cy = px + pw / 2, py + ph / 2
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    box = clip_box((int(cx - ww / 2 + dx * sx), int(cy - wh / 2 + dy * sy), ww, wh),
                                   frame.shape)
                    if box is None or box[2] < 0.8 * ww or box[3] < 0.8 * wh:
                        continue
                    sc = model.fast_score(crop(frame, box))
                    if sc > best_s:
                        best, best_s = box, sc
        return best

    def _refine(self, frame, blob: Box, model: AppearanceModel) -> Box | None:
        """Best window of the reference's aspect ratio inside a blob."""
        x, y, w, h = blob
        a = model.aspect
        if 0.8 < (w / h) / a < 1.25 and w < 80:
            return None  # already vehicle-shaped and small: nothing to split
        h_max = min(h, w / a)
        best, best_s = None, -1.0
        for s in self.cfg.refine_scales:
            wh = int(h_max * s)
            ww = int(wh * a)
            if wh < 16 or ww < 20:
                continue
            sx, sy = max(2, ww // 4), max(2, wh // 4)
            for yy in range(y, y + h - wh + 1, sy):
                for xx in range(x, x + w - ww + 1, sx):
                    sc = model.fast_score(frame[yy:yy + wh, xx:xx + ww])
                    if sc > best_s:
                        best, best_s = (xx, yy, ww, wh), sc
        return best

    # ----------------------------------------------------------- association
    def _associate(self, predictions, boxes, patches, ref_scores) -> dict[int, int]:
        """Jointly assign active targets to candidates maximising total score."""
        cfg = self.cfg
        options: dict[int, list[tuple[int, float]]] = {}
        for i, pred in predictions.items():
            t = self.targets[i]
            px, py, pw, ph = pred
            pcx, pcy = px + pw / 2, py + ph / 2
            gate = cfg.gate_factor * max(pw, ph) * min(2.0, 1 + 0.1 * t.missed)
            size_gate = cfg.size_ratio_gate * min(1.3, 1 + 0.03 * t.missed)
            min_ref = self._min_ref(t)
            opts = []
            for j, b in enumerate(boxes):
                cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                dist = float(np.hypot(cx - pcx, cy - pcy))
                ratio = np.sqrt(max(b[2] * b[3], 1) / max(pw * ph, 1))
                if dist > gate or not (1 / size_gate <= ratio <= size_gate):
                    continue
                ref = float(ref_scores[i, j])
                if ref < min_ref:
                    continue
                # Do not take a candidate that clearly looks like another target.
                others = [ref_scores[k, j] for k in range(len(self.targets)) if k != i]
                if others and max(others) > ref + cfg.acquire_margin:
                    continue
                online = t.model.color_score(patches[j], online=True)
                motion = 1.0 - dist / gate
                opts.append((j, 0.5 * ref + 0.25 * online + 0.25 * motion))
            options[i] = opts

        best: dict[int, int] = {}
        best_total = -1.0
        ids = list(options)
        choices = [[None] + options[i] for i in ids]
        for combo in itertools.product(*choices):
            picked = [boxes[c[0]] for c in combo if c is not None]
            if any(iou(a, b) > cfg.overlap_exclusion for a, b in itertools.combinations(picked, 2)):
                continue
            total = sum(c[1] for c in combo if c is not None)
            if total > best_total:
                best_total = total
                best = {i: c[0] for i, c in zip(ids, combo) if c is not None}
        return best

    def _accept(self, t: _Target, frame, box: Box, score: float, status: Status) -> FrameResult:
        cfg = self.cfg
        if box[1] + box[3] >= frame.shape[0] - 2:
            # Reached the bottom edge: the car is driving out of view. Once it
            # is partly out, a car-shaped window slides up onto whatever is
            # next to it, so give it only as many frames as it takes to be
            # half out at its current speed, then call it exited.
            if t.exit_in is None:
                v = t.kf.kf.statePost.ravel()
                t.exit_in = max(1, int(0.5 * box[3] / max(float(v[5] + v[7] / 2), 0.5)))
            t.exit_in -= 1
            if t.exit_in <= 0:
                self._drop(t, Status.EXITED)
                return FrameResult(self.frame_idx, t.name, Status.EXITED)
        if self._mostly_out(box, frame.shape, t.model.aspect):
            self._drop(t, Status.EXITED)
            return FrameResult(self.frame_idx, t.name, Status.EXITED)
        t.kf.correct(box)
        t.missed = 0
        t.status = status
        if status == Status.TRACKED:
            t.score_ema = score if t.score_ema is None else 0.9 * t.score_ema + 0.1 * score
        if score < cfg.identity_threshold:
            t.low_score_run += 1
        else:
            t.low_score_run = 0
        patch = crop(frame, box)
        if status == Status.TRACKED and score >= cfg.keep_threshold:
            t.model.update_online(patch, cfg.online_rate)
            t.template = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if t.low_score_run > cfg.identity_patience:
            # Sustained weak resemblance to the reference: likely drifted onto
            # another vehicle. Drop instead of reporting a wrong identity.
            self._drop(t, Status.LOST)
            return FrameResult(self.frame_idx, t.name, Status.LOST)
        return FrameResult(self.frame_idx, t.name, status, box, score)

    def _miss(self, t: _Target, shape, pred: Box) -> FrameResult:
        t.missed += 1
        if self._outside_fraction(pred, shape) > self.cfg.exit_margin:
            self._drop(t, Status.EXITED)
        elif t.missed > self.cfg.max_coast:
            self._drop(t, Status.LOST)
        else:
            t.status = Status.OCCLUDED
        return FrameResult(self.frame_idx, t.name, t.status)

    def _drop(self, t: _Target, status: Status) -> None:
        t.kf = None
        t.status = status
        t.missed = 0
        t.low_score_run = 0
        t.score_ema = None
        t.template = None
        t.exit_in = None
        t.pending.clear()
        t.model.reset_online()

    def _min_ref(self, t: _Target) -> float:
        """Lowest reference score still accepted as this target.

        Besides the absolute floor, a match must not fall far below the scores
        this track has been getting: a sudden drop usually means a different,
        merely similar-looking vehicle."""
        floor = self.cfg.keep_threshold
        if t.score_ema is not None:
            floor = max(floor, t.score_ema - self.cfg.score_drop_tolerance)
        return floor

    @staticmethod
    def _car_shaped(box: Box, aspect: float) -> Box:
        x, y, w, h = box
        nh = max(1, int(round(w / aspect)))
        return (x, int(round(y + h / 2 - nh / 2)), w, nh)

    @staticmethod
    def _mostly_out(box: Box, shape, aspect: float) -> bool:
        """Box touches the top or a side edge and shows under half of a car of
        that size. The bottom edge is handled by the exit countdown."""
        x, y, w, h = box
        H, W = shape[:2]
        if y <= 1:
            return h * aspect < 0.5 * w
        if x + w >= W - 2 or x <= 1:
            return w < 0.5 * h * aspect
        return False

    @staticmethod
    def _outside_fraction(box: Box, shape) -> float:
        inside = clip_box(box, shape)
        area = max(box[2] * box[3], 1)
        return 1.0 - (inside[2] * inside[3] / area if inside else 0.0)

    # -------------------------------------------------------- template search
    def _template_search(self, t: _Target, frame, pred: Box,
                         verify: bool = True) -> tuple[Box, float] | None:
        cfg = self.cfg
        if t.template is None or t.missed > cfg.max_coast // 2:
            return None
        px, py, pw, ph = pred
        if pw < 12 or ph < 12:
            return None
        mx, my = pw // 2, ph // 2
        region = clip_box((px - mx, py - my, pw + 2 * mx, ph + 2 * my), frame.shape)
        if region is None:
            return None
        gray = cv2.cvtColor(crop(frame, region), cv2.COLOR_BGR2GRAY)
        # A few scales, so frames between full detections still measure growth.
        ncc, box = -1.0, None
        for s in (0.94, 1.0, 1.06):
            w, h = int(round(pw * s)), int(round(ph * s))
            if w >= gray.shape[1] or h >= gray.shape[0]:
                continue
            res = cv2.matchTemplate(gray, cv2.resize(t.template, (w, h), interpolation=cv2.INTER_LINEAR),
                                    cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(res)
            if v > ncc:
                ncc, box = v, (region[0] + loc[0], region[1] + loc[1], w, h)
        if box is None or ncc < cfg.template_threshold:
            return None
        if not verify:
            return box, ncc
        ref = t.model.score(crop(frame, box)).total
        if ref < self._min_ref(t):
            return None
        return box, ref

    # ----------------------------------------------------------- acquisition
    def _try_acquire(self, i: int, t: _Target, frame, boxes, ref_scores, used) -> FrameResult:
        cfg = self.cfg
        idle = Status.SEARCHING if t.status == Status.SEARCHING else t.status
        best_j, best_s = None, cfg.acquire_threshold
        for j in range(len(boxes)):
            if boxes[j][2] < cfg.acquire_min_width:
                continue
            if any(iou(boxes[j], u) > cfg.overlap_exclusion for u in used):
                continue
            s = float(ref_scores[i, j])
            others = [ref_scores[k, j] for k in range(len(self.targets)) if k != i]
            if others and s < max(others) + cfg.acquire_margin:
                continue
            if s >= best_s:
                best_j, best_s = j, s
        if best_j is None:
            t.pending.clear()
            return FrameResult(self.frame_idx, t.name, idle)

        box = boxes[best_j]
        if t.pending:
            _, last, _ = t.pending[-1]
            lcx, lcy = last[0] + last[2] / 2, last[1] + last[3] / 2
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            consistent = iou(last, box) > 0.2 or np.hypot(cx - lcx, cy - lcy) < 0.5 * max(last[2], last[3])
            if not consistent:
                t.pending.clear()
        t.pending.append((self.frame_idx, box, best_s))
        if len(t.pending) < cfg.confirm_frames:
            return FrameResult(self.frame_idx, t.name, idle)

        # Confirmed: initialise the filter from the pending detections and
        # back-fill those frames, which were genuine matched detections.
        t.kf = BoxKalman(t.pending[0][1])
        for _, b, _ in t.pending[1:]:
            t.kf.predict()
            t.kf.correct(b)
        for f, b, s in t.pending[:-1]:
            self.history[f][i] = FrameResult(f, t.name, Status.TRACKED, b, s)
        t.score_ema = float(np.mean([s for _, _, s in t.pending]))
        t.pending.clear()
        t.status = Status.TRACKED
        t.missed = 0
        t.low_score_run = 0
        patch = crop(frame, box)
        t.model.reset_online()
        t.model.update_online(patch, cfg.online_rate)
        t.template = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return FrameResult(self.frame_idx, t.name, Status.TRACKED, box, best_s)
