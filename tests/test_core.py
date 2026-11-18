"""Unit + closed-loop simulation tests. Run: python -m pytest -q"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spot_follow import follower as follower_mod  # noqa: E402
from spot_follow.cameras import (apply_affine, depth_in_box, pixel_to_normalized,  # noqa: E402
                                 rotation_matrices, range_from_height)
from spot_follow.config import load_config  # noqa: E402
from spot_follow.controller import clamp_to_leash, compute_goal  # noqa: E402
from spot_follow.tracker import Candidate, TargetTracker  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


# ----------------------------------------------------------------- geometry

@pytest.mark.parametrize('angle', [-102.0, -78.0, 0.0, 90.0, 180.0])
def test_rotation_roundtrip(angle):
    m, minv, (w, h) = rotation_matrices(640, 480, angle)
    for u, v in [(0, 0), (639, 479), (320, 240), (100, 400)]:
        uu, vv = apply_affine(m, u, v)
        assert -1 <= uu <= w + 1 and -1 <= vv <= h + 1
        back = apply_affine(minv, uu, vv)
        assert np.allclose(back, (u, v), atol=1e-6)


def test_kannala_brandt_center_ray():
    from bosdyn.api import image_pb2
    src = image_pb2.ImageSource()
    kb = src.kannala_brandt.intrinsics
    kb.pinhole_intrinsics.focal_length.x = kb.pinhole_intrinsics.focal_length.y = 300
    kb.pinhole_intrinsics.principal_point.x, kb.pinhole_intrinsics.principal_point.y = 320, 240
    kb.k1, kb.k2, kb.k3, kb.k4 = -0.01, 0.002, 0.0, 0.0
    assert np.allclose(pixel_to_normalized(src, 320, 240), (0, 0), atol=1e-6)
    x, _ = pixel_to_normalized(src, 470, 240)
    assert x > 0.45  # fisheye: further out than the pinhole value 0.5 would suggest after undistortion


def test_depth_in_box_prefers_target():
    d = np.full((480, 640), 6.0, np.float32)     # wall at 6 m
    d[150:350, 280:360] = 2.0                    # person at 2 m
    assert depth_in_box(d, (260, 100, 380, 450)) == pytest.approx(2.0)
    assert depth_in_box(None, (0, 0, 10, 10)) is None


def test_range_from_height():
    z, trunc = range_from_height((0, 100, 50, 270), 340.0, 1.7, 480)
    assert z == pytest.approx(340 * 1.7 / 170) and not trunc


# ----------------------------------------------------------------- controller

def test_controller_approach_face_and_leash(cfg):
    g = compute_goal((0, 0), 0.0, (5, 0), cfg.follow)
    assert g.action == 'approach' and g.x == pytest.approx(5 - cfg.follow.standoff_m) and g.heading == 0
    g = compute_goal((0, 0), 0.0, (0, 1.5), cfg.follow)
    assert g.action == 'face' and g.heading == pytest.approx(math.pi / 2) and (g.x, g.y) == (0, 0)
    g = compute_goal((0, 0), 0.0, (1.5, 0), cfg.follow)
    assert g.action == 'hold'
    g = compute_goal((0, 0), 0.0, (30, 0), cfg.follow, leash_center=(0, 0))
    assert g.leashed and math.hypot(g.x, g.y) == pytest.approx(cfg.follow.leash_radius_m)
    p, leashed = clamp_to_leash(np.array([3.0, 4.0]), np.zeros(2), 10)
    assert not leashed


# ----------------------------------------------------------------- tracker

def _c(x, y, ident=None, cam='frontleft'):
    return Candidate(np.array([x, y], float), 0.9, cam, 2.0, 'depth', identity=ident)


def test_tracker_confirms_and_follows(cfg):
    tr = TargetTracker(cfg)
    assert tr.update([_c(2, 0)], 0.0, np.zeros(2)) is None
    assert tr.update([_c(2.05, 0)], 0.2, np.zeros(2)) is None
    t = tr.update([_c(2.1, 0)], 0.4, np.zeros(2))
    assert t is not None and t.confirmed
    # a far-away second person doesn't steal the lock
    t = tr.update([_c(6, 3), _c(2.2, 0.05)], 0.6, np.zeros(2))
    assert t.pos[0] < 3


def test_tracker_face_mode_requires_identity(cfg):
    cfg.detector.mode = 'face'
    tr = TargetTracker(cfg)
    for i in range(5):
        assert tr.update([_c(2, 0, ident=0.1)], i * 0.2, np.zeros(2)) is None   # stranger
    tr.reset()
    for i in range(3):
        t = tr.update([_c(2, 0, ident=0.1), _c(3, 1, ident=0.6)], i * 0.2, np.zeros(2))
    assert t is not None and t.pos[0] > 2.5
    # back turned (no face) -> lock kept
    t = tr.update([_c(3.1, 1.0, ident=None)], 0.8, np.zeros(2))
    assert t is not None
    # a clearly different face on the track 3x -> lock dropped
    for k in range(3):
        t = tr.update([_c(3.1, 1.0, ident=0.05)], 1.0 + k * 0.2, np.zeros(2))
    assert t is None and tr.track is None


def test_merge_duplicate_camera_detections(cfg):
    tr = TargetTracker(cfg)
    for i in range(3):
        t = tr.update([_c(2, 0, cam='frontleft'), _c(2.2, 0.1, cam='frontright')], i * 0.2, np.zeros(2))
    assert t is not None and len(t.hits) == 3


# ----------------------------------------------------------------- closed loop

def _run_sim(cfg, target_fn, seconds, monkeypatch):
    from sim import FakeDetector, FakeSpot
    spot = FakeSpot(cfg, target_fn)
    det = FakeDetector()
    real_build = follower_mod.build_frames

    def build_and_record(responses, cameras):
        det.frames = real_build(responses, cameras)
        return det.frames
    monkeypatch.setattr(follower_mod, 'build_frames', build_and_record)
    clock = {'t': 1000.0}
    monkeypatch.setattr(follower_mod.time, 'time', lambda: clock['t'])
    f = follower_mod.Follower(cfg, spot, det)
    dt = 1.0 / cfg.loop_hz
    trace = []
    for _ in range(int(seconds / dt)):
        f.step()
        spot.advance(dt)
        clock['t'] += dt
        trace.append((f.state, spot.x, spot.y, *spot.target_world()))
    return f, spot, trace


def test_sim_finds_target_behind_and_follows(cfg, monkeypatch):
    # Target starts behind Spot (only the back camera sees it) and walks away along -x.
    f, spot, trace = _run_sim(cfg, lambda t: (-3.0 - 0.4 * t, 0.5), 15, monkeypatch)
    states = [s for s, *_ in trace]
    assert 'FOLLOW' in states
    _, rx, ry, tx, ty = trace[-1]
    dist = math.hypot(tx - rx, ty - ry)
    assert abs(dist - cfg.follow.standoff_m) < 0.8, dist
    assert rx < -3.0   # robot actually travelled after the target


def test_sim_searches_when_target_absent(cfg, monkeypatch):
    f, spot, trace = _run_sim(cfg, lambda t: (100.0, 100.0), 5, monkeypatch)
    assert all(s == 'SEARCH' for s, *_ in trace)
    assert any(c[0] == 'turn' for c in spot.commands)


def test_sim_reacquire_then_search(cfg, monkeypatch):
    # Visible for 4 s at 4 m, then teleports out of range -> REACQUIRE -> SEARCH
    f, spot, trace = _run_sim(cfg, lambda t: (4.0, 0.0) if t < 4 else (200.0, 0.0), 14, monkeypatch)
    states = [s for s, *_ in trace]
    assert 'FOLLOW' in states and 'REACQUIRE' in states and states[-1] == 'SEARCH'


def test_sim_leash_respected(cfg, monkeypatch):
    cfg.follow.leash_radius_m = 3.0
    f, spot, trace = _run_sim(cfg, lambda t: (2.0 + 0.6 * t, 0.0), 20, monkeypatch)
    assert max(math.hypot(x, y) for _, x, y, *_ in trace) <= 3.0 + 0.05
