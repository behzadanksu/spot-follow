"""Perceive -> track -> decide -> act loop with a SEARCH / FOLLOW / REACQUIRE state machine."""
import logging
import math
import time

import numpy as np

from .cameras import (CAMERA_SOURCES, CameraFrame, apply_affine, decode_image, depth_to_meters,
                      depth_in_box, focal_length_px, frame_transforms, make_upright,
                      pixel_to_camera_point, range_from_height, to_bgr)
from .controller import compute_goal, goal_changed
from .tracker import Candidate, TargetTracker

log = logging.getLogger(__name__)

SEARCH, FOLLOW, REACQUIRE, IDLE = 'SEARCH', 'FOLLOW', 'REACQUIRE', 'IDLE'


# --------------------------------------------------------------------------- perception

def build_frames(responses, cameras):
    """Pair visual + depth responses by camera and make everything upright."""
    by_source = {r.source.name: r for r in responses}
    frames = []
    for cam in cameras:
        vis_name, depth_name, angle = CAMERA_SOURCES[cam]
        vis = by_source.get(vis_name)
        if vis is None:
            continue
        image = to_bgr(decode_image(vis))
        depth = None
        d = by_source.get(depth_name)
        if d is not None:
            raw = decode_image(d)
            if raw.shape[:2] == image.shape[:2]:
                depth = depth_to_meters(raw, d.source.depth_scale)
        frame = CameraFrame(name=cam, response=vis, image=image, depth_m=depth)
        make_upright(frame, angle)
        frame.odom_T_cam, frame.odom_T_body = frame_transforms(vis)
        frame.focal_px = focal_length_px(vis.source)
        frames.append(frame)
    return frames


def detection_to_candidate(frame: CameraFrame, det, rcfg):
    """Lift one 2D detection into the odom frame."""
    z, source = None, 'depth'
    if rcfg.use_depth:
        z = depth_in_box(frame.upright_depth_m, det.box, rcfg.min_range_m, rcfg.max_range_m)
    if z is None:
        z, truncated = range_from_height(det.box, frame.focal_px, rcfg.assumed_target_height_m,
                                         frame.upright.shape[0])
        source = 'height-truncated' if truncated else 'height'
        if z > rcfg.max_range_m and not truncated:
            return None   # too far to be worth chasing / estimate unreliable
        z = float(np.clip(z, rcfg.min_range_m, rcfg.max_range_m))
    # Aim at the torso centre, mapped back to the camera's native pixel grid.
    x1, y1, x2, y2 = det.box
    u_up, v_up = 0.5 * (x1 + x2), y1 + 0.4 * (y2 - y1)
    u, v = apply_affine(frame.rot_inv, u_up, v_up)
    p_cam = pixel_to_camera_point(frame.response.source, u, v, z)
    p_odom = frame.odom_T_cam.transform_point(*p_cam)
    return Candidate(pos=np.array([p_odom[0], p_odom[1]]), score=det.score, camera=frame.name,
                     range_m=z, range_source=source, identity=det.identity,
                     appearance=det.appearance, detection=det)


# --------------------------------------------------------------------------- the loop

class Follower:

    def __init__(self, cfg, spot, detector, viz=None, dry_run=False):
        self.cfg, self.spot, self.detector, self.viz = cfg, spot, detector, viz
        self.dry_run = dry_run
        self.tracker = TargetTracker(cfg)
        self.state = SEARCH
        self.state_since = time.time()
        self.search_started = time.time()
        self.start_xy = None
        self.last_goal = None
        self.last_target_bearing = None
        self.requests = spot.image_requests(cfg.cameras, cfg.range.use_depth, cfg.image_quality)
        self.cycle = 0
        self.stop_requested = False   # set by the Ctrl-C handler

    def _set_state(self, new):
        if new != self.state:
            log.info('STATE %s -> %s', self.state, new)
            self.state, self.state_since = new, time.time()
            if new == SEARCH:
                self.search_started = time.time()

    def _act(self, kind, *args):
        """Send a command, or just log it in dry-run mode."""
        if self.dry_run:
            if self.cycle % 5 == 0:
                log.info('[dry-run] would %s %s', kind, ', '.join(f'{a:.2f}' for a in args))
            return
        if kind == 'walk_to':
            self.spot.walk_to(*args)
        elif kind == 'turn':
            self.spot.turn(*args)
        elif kind == 'stop':
            self.spot.stop()

    def perceive(self):
        responses = self.spot.get_images(self.requests)
        frames = build_frames(responses, self.cfg.cameras)
        cands, dets_by_cam = [], {}
        for fr in frames:
            dets = self.detector.detect(fr.upright)
            dets_by_cam[fr.name] = dets
            for det in dets:
                try:
                    cand = detection_to_candidate(fr, det, self.cfg.range)
                    if cand is not None:
                        cands.append(cand)
                except Exception as exc:  # noqa: BLE001 - one bad pixel shouldn't stop the loop
                    log.debug('Could not lift detection: %s', exc)
        return frames, dets_by_cam, cands

    def step(self):
        now = time.time()
        rx, ry, ryaw = self.spot.odom_pose()
        robot_xy = np.array([rx, ry])
        if self.start_xy is None:
            self.start_xy = robot_xy.copy()

        frames, dets_by_cam, cands = self.perceive()
        reacquiring = self.state == REACQUIRE
        track = self.tracker.update(cands, now, robot_xy, reacquiring=reacquiring)
        unseen = self.tracker.time_since_seen(now)
        tc = self.cfg.tracking

        # ---------------- state transitions
        if track is not None and unseen < tc.lost_timeout_s:
            self._set_state(FOLLOW)
        elif self.state == FOLLOW and unseen >= tc.lost_timeout_s:
            self._set_state(REACQUIRE)
        elif self.state == REACQUIRE and unseen >= tc.reacquire_timeout_s:
            self.tracker.reset()
            self._set_state(SEARCH)
        elif self.state == SEARCH and self.cfg.search.max_search_s and \
                now - self.search_started > self.cfg.search.max_search_s:
            log.warning('Search timed out; standing still. Ctrl-C to exit or step into view.')
            self._set_state(IDLE)
        elif self.state == IDLE and track is not None:
            self._set_state(FOLLOW)

        # ---------------- actions
        goal = None
        if self.state == FOLLOW:
            goal = compute_goal(robot_xy, ryaw, track.pos, self.cfg.follow, self.start_xy)
            self.last_target_bearing = math.atan2(track.pos[1] - ry, track.pos[0] - rx)
            if goal.action in ('approach', 'backoff', 'face'):
                self._act('walk_to', goal.x, goal.y, goal.heading)
            if goal_changed(self.last_goal, goal, self.cfg.follow):
                log.info('FOLLOW %s: target %.2fm away via %s%s', goal.action, goal.distance,
                         track.last_camera, ' (leash limit!)' if goal.leashed else '')
            self.last_goal = goal

        elif self.state == REACQUIRE and self.tracker.track is not None:
            # Head toward where the target was going; it is probably just around the corner.
            pred = self.tracker.track.predict(now)
            goal = compute_goal(robot_xy, ryaw, pred, self.cfg.follow, self.start_xy)
            self._act('walk_to', goal.x, goal.y, goal.heading)

        elif self.state == SEARCH:
            s = self.cfg.search
            period = s.rotate_step_s + s.pause_s
            phase = (now - self.state_since) % period
            if phase < s.rotate_step_s:
                direction = 1.0
                if self.last_target_bearing is not None:
                    err = math.atan2(math.sin(self.last_target_bearing - ryaw),
                                     math.cos(self.last_target_bearing - ryaw))
                    direction = 1.0 if err >= 0 else -1.0
                self._act('turn', direction * s.rotate_vel, min(0.6, s.rotate_step_s - phase + 0.1))
            # During the pause Spot stands still so the cameras get sharp frames.

        if self.viz is not None:
            self.viz.update(frames, dets_by_cam, self.state, track, goal, (rx, ry, ryaw), unseen)
        self.cycle += 1

    def run(self):
        period = 1.0 / float(self.cfg.loop_hz)
        log.info('Follower running (%s mode, %s). Ctrl-C to stop.', self.cfg.detector.mode,
                 'DRY RUN - no motion' if self.dry_run else 'LIVE')
        while not self.stop_requested:
            t0 = time.time()
            self.step()
            if self.viz is not None and self.viz.quit_requested:
                log.info('Quit requested from viewer')
                break
            dt = time.time() - t0
            if dt > 2 * period and self.cycle % 10 == 0:
                log.warning('Loop is slow: %.0f ms per cycle (target %.0f ms). Fewer cameras, '
                            'lower imgsz, or a GPU will help.', dt * 1000, period * 1000)
            time.sleep(max(0.0, period - dt))
