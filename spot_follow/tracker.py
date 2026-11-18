"""Single-target tracker in Spot's odometry frame.

Why track in odom (a world-fixed frame) instead of in the image: the target
moves between cameras as Spot turns, and the robot itself moves. In odom the
target's position is continuous no matter which camera saw it.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .detectors import appearance_similarity

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A detection lifted into 3D (odom frame x, y)."""
    pos: np.ndarray                 # (x, y) in odom
    score: float
    camera: str
    range_m: float
    range_source: str               # 'depth' | 'height' | 'height-truncated'
    identity: Optional[float] = None
    appearance: Optional[np.ndarray] = None
    detection: object = None


@dataclass
class Track:
    pos: np.ndarray
    vel: np.ndarray
    first_seen: float
    last_seen: float
    hits: List[float] = field(default_factory=list)
    confirmed: bool = False
    appearance: Optional[np.ndarray] = None
    face_rejects: int = 0
    best_identity: Optional[float] = None
    last_camera: str = ''

    def predict(self, t):
        dt = min(max(t - self.last_seen, 0.0), 1.0)   # don't extrapolate wildly
        return self.pos + self.vel * dt


def merge_candidates(cands: List[Candidate], merge_distance):
    """Fuse detections of the same object seen by two overlapping cameras."""
    merged = []
    for c in sorted(cands, key=lambda c: -c.score):
        for m in merged:
            if np.linalg.norm(m.pos - c.pos) < merge_distance:
                if c.identity is not None and (m.identity is None or c.identity > m.identity):
                    m.identity = c.identity
                if m.range_source != 'depth' and c.range_source == 'depth':
                    m.pos, m.range_m, m.range_source = c.pos, c.range_m, c.range_source
                break
        else:
            merged.append(c)
    return merged


class TargetTracker:

    def __init__(self, cfg):
        self.t = cfg.tracking
        self.mode = cfg.detector.mode
        self.face_cfg = cfg.face
        self.track: Optional[Track] = None

    # ------------------------------------------------------------------ helpers
    def reset(self):
        if self.track is not None:
            log.info('Target lock dropped')
        self.track = None

    def time_since_seen(self, now):
        return math.inf if self.track is None else now - self.track.last_seen

    def _is_target_identity(self, c: Candidate):
        if self.mode != 'face':
            return True
        return c.identity is not None and c.identity >= self.face_cfg.match_threshold

    def _acquire(self, cands, robot_xy):
        """Pick who to lock onto when we have no track."""
        eligible = [c for c in cands if self._is_target_identity(c)]
        if not eligible:
            return None
        if self.mode == 'face':
            return max(eligible, key=lambda c: c.identity)
        if self.mode == 'person':
            return min(eligible, key=lambda c: np.linalg.norm(c.pos - robot_xy))
        return max(eligible, key=lambda c: c.score)

    def _associate(self, cands, now, reacquiring):
        """Pick the candidate that best continues the current track."""
        pred = self.track.predict(now)
        gate = self.t.gate_distance_m * (2.0 if reacquiring else 1.0)
        best, best_cost = None, math.inf
        for c in cands:
            d = float(np.linalg.norm(c.pos - pred))
            if d > gate:
                continue
            if self.mode == 'face' and c.identity is not None \
                    and c.identity < self.face_cfg.reject_threshold and reacquiring:
                continue  # a clearly different face can't re-take a lost lock
            cost = d - self.t.appearance_weight * appearance_similarity(c.appearance, self.track.appearance)
            if self.mode == 'face' and c.identity is not None and c.identity >= self.face_cfg.match_threshold:
                cost -= 1.0  # a matching face is strong evidence
            if cost < best_cost:
                best, best_cost = c, cost
        return best

    # ------------------------------------------------------------------ update
    def update(self, cands: List[Candidate], now: float, robot_xy, reacquiring=False):
        """Feed this cycle's candidates. Returns the track if it is confirmed, else None."""
        cands = merge_candidates(cands, self.t.merge_distance_m)

        if self.track is None:
            c = self._acquire(cands, robot_xy)
            if c is not None:
                self.track = Track(pos=c.pos.copy(), vel=np.zeros(2), first_seen=now, last_seen=now,
                                   hits=[now], appearance=c.appearance, best_identity=c.identity,
                                   last_camera=c.camera)
                log.info('New candidate target at (%.2f, %.2f) via %s', c.pos[0], c.pos[1], c.camera)
            return self._confirmed_track(now)

        c = self._associate(cands, now, reacquiring)
        if c is None:
            # Unconfirmed tracks that go quiet are discarded quickly.
            if not self.track.confirmed and now - self.track.last_seen > self.t.confirm_window_s:
                self.reset()
            return self._confirmed_track(now)

        tr = self.track
        dt = max(now - tr.last_seen, 1e-3)
        a = self.t.smoothing_alpha
        new_pos = (1 - a) * tr.predict(now) + a * c.pos
        inst_vel = (new_pos - tr.pos) / dt
        tr.vel = 0.7 * tr.vel + 0.3 * np.clip(inst_vel, -2.5, 2.5)
        tr.pos = new_pos
        tr.last_seen = now
        tr.last_camera = c.camera
        tr.hits = [h for h in tr.hits if now - h < self.t.confirm_window_s] + [now]
        if c.appearance is not None:
            tr.appearance = c.appearance if tr.appearance is None else \
                0.8 * tr.appearance + 0.2 * c.appearance
        if c.identity is not None:
            tr.best_identity = max(tr.best_identity or -1.0, c.identity)
            if self.mode == 'face':
                if c.identity < self.face_cfg.reject_threshold:
                    tr.face_rejects += 1
                elif c.identity >= self.face_cfg.match_threshold:
                    tr.face_rejects = 0
                if tr.face_rejects >= self.face_cfg.reject_frames:
                    log.warning('Locked person shows a different face; dropping lock')
                    self.reset()
                    return None
        return self._confirmed_track(now)

    def _confirmed_track(self, now):
        tr = self.track
        if tr is None:
            return None
        if not tr.confirmed and len(tr.hits) >= self.t.confirm_frames:
            tr.confirmed = True
            log.info('Target CONFIRMED at (%.2f, %.2f)', tr.pos[0], tr.pos[1])
        return tr if tr.confirmed else None
