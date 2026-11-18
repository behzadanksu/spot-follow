"""Tiny kinematic simulator of Spot's camera API, for testing without a robot.

It fabricates real bosdyn ImageResponse protos (intrinsics, transform snapshots,
JPEG + depth payloads) so the production perception code runs unmodified.
"""
import math

import cv2
import numpy as np
from bosdyn.api import geometry_pb2, image_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME

from spot_follow.cameras import CAMERA_SOURCES
from spot_follow.detectors import Detection

W, H, F = 640, 480, 330.0

# Each camera: mounting yaw on the body (rad). Optical frame: z fwd, x right, y down.
MOUNT_YAW = {'frontleft': math.radians(20), 'frontright': math.radians(-20),
             'left': math.radians(90), 'right': math.radians(-90), 'back': math.pi}


def body_T_cam(cam):
    yaw = MOUNT_YAW[cam]
    # optical axes expressed in body frame
    z = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    x = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    R = np.stack([x, y, z], axis=1)
    rot = math_helpers.Quat.from_matrix(R)
    return math_helpers.SE3Pose(0.3 * math.cos(yaw), 0.3 * math.sin(yaw), 0.1, rot)


def _edge(parent, pose):
    return geometry_pb2.FrameTreeSnapshot.ParentEdge(parent_frame_name=parent,
                                                      parent_tform_child=pose.to_proto())


class FakeSpot:
    """Robot at (x, y, yaw) in odom; a target walks on a scripted path."""

    def __init__(self, cfg, target_fn):
        self.cfg = cfg
        self.x = self.y = self.yaw = 0.0
        self.t = 0.0
        self.target_fn = target_fn
        self.commands = []

    # --- interface used by Follower
    def image_requests(self, cameras, with_depth, quality):
        return list(cameras)

    def odom_pose(self):
        return self.x, self.y, self.yaw

    def target_world(self):
        return np.array(self.target_fn(self.t))

    def get_images(self, cameras):
        odom_T_body = math_helpers.SE3Pose(self.x, self.y, 0.5, math_helpers.Quat.from_yaw(self.yaw))
        out = []
        tw = self.target_world()
        for cam in cameras:
            vis_name, depth_name, angle = CAMERA_SOURCES[cam]
            sensor = f'{cam}_sensor'
            snap = geometry_pb2.FrameTreeSnapshot()
            snap.child_to_parent_edge_map[ODOM_FRAME_NAME].CopyFrom(
                geometry_pb2.FrameTreeSnapshot.ParentEdge(parent_frame_name=''))
            snap.child_to_parent_edge_map[BODY_FRAME_NAME].CopyFrom(_edge(ODOM_FRAME_NAME, odom_T_body))
            snap.child_to_parent_edge_map[sensor].CopyFrom(_edge(BODY_FRAME_NAME, body_T_cam(cam)))
            odom_T_cam = odom_T_body * body_T_cam(cam)
            p = odom_T_cam.inverse().transform_point(tw[0], tw[1], 1.0)  # torso height 1.0 m
            img = np.full((H, W), 90, np.uint8)
            depth = np.zeros((H, W), np.uint16)
            meta = None
            if 0.3 < p[2] < 15.0:   # a real detector can't see a person farther than this
                u, v = F * p[0] / p[2] + W / 2, F * p[1] / p[2] + H / 2
                if 0 <= u < W and 0 <= v < H:
                    half = int(F * 0.25 / p[2]) + 2
                    cv2.rectangle(depth, (int(u) - half, int(v) - half), (int(u) + half, int(v) + half),
                                  int(p[2] * 1000), -1)
                    meta = (u, v, p[2])
            for name, data, fmt, pix in (
                    (vis_name, cv2.imencode('.jpg', img)[1].tobytes(), image_pb2.Image.FORMAT_JPEG,
                     image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8),
                    (depth_name, depth.tobytes(), image_pb2.Image.FORMAT_RAW,
                     image_pb2.Image.PIXEL_FORMAT_DEPTH_U16)):
                r = image_pb2.ImageResponse()
                r.source.name = name
                r.source.rows, r.source.cols = H, W
                r.source.depth_scale = 1000.0
                r.source.pinhole.intrinsics.focal_length.x = F
                r.source.pinhole.intrinsics.focal_length.y = F
                r.source.pinhole.intrinsics.principal_point.x = W / 2
                r.source.pinhole.intrinsics.principal_point.y = H / 2
                r.shot.image.data = data
                r.shot.image.format = fmt
                r.shot.image.pixel_format = pix
                r.shot.image.rows, r.shot.image.cols = H, W
                r.shot.frame_name_image_sensor = sensor
                r.shot.transforms_snapshot.CopyFrom(snap)
                out.append(r)
            FakeDetector.truth[cam] = meta
        return out

    def walk_to(self, x, y, heading):
        self.commands.append(('walk_to', x, y, heading))
        self._goal = (x, y, heading)

    def turn(self, v_rot, duration=None):
        self.commands.append(('turn', v_rot))
        self._goal = ('turn', v_rot)

    def stop(self):
        self._goal = None

    def advance(self, dt):
        """Move toward the last goal at the configured speed limits."""
        g = getattr(self, '_goal', None)
        f = self.cfg.follow
        if g and g[0] == 'turn':
            self.yaw += g[1] * dt
        elif g:
            dx, dy = g[0] - self.x, g[1] - self.y
            d = math.hypot(dx, dy)
            step = min(d, f.max_lin_vel * dt)
            if d > 1e-6:
                self.x += dx / d * step
                self.y += dy / d * step
            e = math.atan2(math.sin(g[2] - self.yaw), math.cos(g[2] - self.yaw))
            self.yaw += max(-f.max_ang_vel * dt, min(f.max_ang_vel * dt, e))
        self._goal = None
        self.t += dt


class FakeDetector:
    """Returns a perfect box wherever the simulator projected the target.

    `frames` must be the list of CameraFrames currently being processed (the test
    wraps build_frames to fill it) so the box can be placed in upright coordinates.
    """
    truth = {}

    def __init__(self):
        self.frames = []

    def detect(self, upright):
        fr = next(f for f in self.frames if f.upright is upright)
        meta = FakeDetector.truth.get(fr.name)
        if meta is None:
            return []
        u, v, z = meta
        uu, vv = fr.rot[0] @ [u, v, 1], fr.rot[1] @ [u, v, 1]
        hb = F * 1.7 / z      # box height for a 1.7 m person
        # torso centre sits 40% down the box
        return [Detection((uu - hb / 4, vv - 0.4 * hb, uu + hb / 4, vv + 0.6 * hb), 0.9, 'person',
                          appearance=np.ones(32) / math.sqrt(32))]
