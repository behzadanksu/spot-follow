"""Spot body cameras: decoding, making images upright, and pixel -> 3D geometry.

Spot's five body cameras are mounted at odd angles (the two front cameras are
turned roughly sideways, the right camera is upside down). Detectors are trained
on upright images, so every frame is rotated upright before detection and the
results are mapped back to the original pixel grid for the 3D math.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_a_tform_b, get_odom_tform_body

# name -> (visual source, depth-aligned-to-visual source, degrees to rotate upright)
# Rotation angles match Boston Dynamics' own get_image example (counter-clockwise positive,
# same convention for scipy.ndimage.rotate and cv2.getRotationMatrix2D).
CAMERA_SOURCES = {
    'frontleft': ('frontleft_fisheye_image', 'frontleft_depth_in_visual_frame', -78.0),
    'frontright': ('frontright_fisheye_image', 'frontright_depth_in_visual_frame', -102.0),
    'left': ('left_fisheye_image', 'left_depth_in_visual_frame', 0.0),
    'right': ('right_fisheye_image', 'right_depth_in_visual_frame', 180.0),
    'back': ('back_fisheye_image', 'back_depth_in_visual_frame', 0.0),
}


@dataclass
class CameraFrame:
    """One camera's visual image (+ optional aligned depth) at one instant."""
    name: str
    response: object                     # image_pb2.ImageResponse (visual)
    image: np.ndarray                    # original orientation, BGR uint8
    depth_m: Optional[np.ndarray]        # original orientation, float32 metres (0 = invalid)
    upright: np.ndarray = None           # rotated upright, BGR
    upright_depth_m: Optional[np.ndarray] = None
    rot: np.ndarray = None               # 2x3 affine: original -> upright
    rot_inv: np.ndarray = None           # 2x3 affine: upright -> original
    odom_T_cam: object = None            # math_helpers.SE3Pose
    odom_T_body: object = None
    focal_px: float = 500.0
    extras: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- decoding

def decode_image(response):
    """ImageResponse -> numpy array. Visual images become BGR uint8, depth stays uint16."""
    img = response.shot.image
    if img.format == image_pb2.Image.FORMAT_JPEG:
        arr = cv2.imdecode(np.frombuffer(img.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        if img.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype, channels = np.uint16, 1
        elif img.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            dtype, channels = np.uint8, 3
        elif img.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            dtype, channels = np.uint8, 4
        else:
            dtype, channels = np.uint8, 1
        arr = np.frombuffer(img.data, dtype=dtype)
        if channels == 1:
            arr = arr.reshape(img.rows, img.cols)
        else:
            arr = arr.reshape(img.rows, img.cols, channels)
            if channels == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if arr is None:
        raise ValueError(f'Could not decode image from {response.source.name}')
    return arr


def to_bgr(arr):
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


def depth_to_meters(depth_raw, depth_scale):
    scale = depth_scale if depth_scale and depth_scale > 0 else 1000.0
    d = depth_raw.astype(np.float32) / float(scale)
    d[depth_raw == 0] = 0.0
    return d


# --------------------------------------------------------------------------- rotation

def rotation_matrices(width, height, angle_deg):
    """Affine (original->upright) on an expanded canvas, its inverse, and the output size."""
    cx, cy = width / 2.0, height / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    out_w = int(round(height * sin + width * cos))
    out_h = int(round(height * cos + width * sin))
    m[0, 2] += out_w / 2.0 - cx
    m[1, 2] += out_h / 2.0 - cy
    return m, cv2.invertAffineTransform(m), (out_w, out_h)


def apply_affine(m, u, v):
    return (m[0, 0] * u + m[0, 1] * v + m[0, 2], m[1, 0] * u + m[1, 1] * v + m[1, 2])


def make_upright(frame: CameraFrame, angle_deg: float):
    h, w = frame.image.shape[:2]
    frame.rot, frame.rot_inv, size = rotation_matrices(w, h, angle_deg)
    frame.upright = cv2.warpAffine(frame.image, frame.rot, size, flags=cv2.INTER_LINEAR)
    if frame.depth_m is not None:
        frame.upright_depth_m = cv2.warpAffine(frame.depth_m, frame.rot, size,
                                               flags=cv2.INTER_NEAREST)
    return frame


# --------------------------------------------------------------------------- intrinsics

def _intrinsics(source):
    """Return (K 3x3, distortion model name, distortion coeffs) from an ImageSource."""
    which = source.WhichOneof('camera_models')
    if which == 'pinhole':
        intr = source.pinhole.intrinsics
        dist = ('none', None)
    elif which == 'pinhole_brown_conrady':
        bc = source.pinhole_brown_conrady.intrinsics
        intr = bc.pinhole_intrinsics
        dist = ('brown_conrady', np.array([bc.k1, bc.k2, bc.p1, bc.p2, bc.k3], np.float64))
    elif which == 'kannala_brandt':
        kb = source.kannala_brandt.intrinsics
        intr = kb.pinhole_intrinsics
        dist = ('kannala_brandt', np.array([kb.k1, kb.k2, kb.k3, kb.k4], np.float64))
    else:
        raise ValueError(f'Image source {source.name} has no camera model')
    k = np.array([[intr.focal_length.x, intr.skew.x, intr.principal_point.x],
                  [0.0, intr.focal_length.y, intr.principal_point.y],
                  [0.0, 0.0, 1.0]], np.float64)
    return k, dist[0], dist[1]


def focal_length_px(source):
    k, _, _ = _intrinsics(source)
    return float((k[0, 0] + k[1, 1]) / 2.0)


def pixel_to_normalized(source, u, v):
    """Pixel in the ORIGINAL image -> (x/z, y/z) in the camera frame."""
    k, model, dist = _intrinsics(source)
    pts = np.array([[[float(u), float(v)]]], np.float64)
    if model == 'brown_conrady':
        und = cv2.undistortPoints(pts, k, dist)
    elif model == 'kannala_brandt':
        und = cv2.fisheye.undistortPoints(pts, k, dist)
    else:
        und = cv2.undistortPoints(pts, k, None)
    return float(und[0, 0, 0]), float(und[0, 0, 1])


def pixel_to_camera_point(source, u, v, z_depth):
    """Pixel + depth along the optical axis (what Spot's depth images store) -> (x, y, z)."""
    xn, yn = pixel_to_normalized(source, u, v)
    return np.array([xn * z_depth, yn * z_depth, z_depth], np.float64)


# --------------------------------------------------------------------------- range

def depth_in_box(depth_m, box, min_r=0.3, max_r=8.0):
    """Robust depth of a target from an (upright) depth image and an (upright) box.

    Samples the central torso region and returns a low percentile of the valid values,
    which favours the target over the background visible around it. None if no data.
    """
    if depth_m is None:
        return None
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rx1, rx2 = int(x1 + 0.3 * w), int(x2 - 0.3 * w)
    ry1, ry2 = int(y1 + 0.2 * h), int(y1 + 0.6 * h)
    H, W = depth_m.shape[:2]
    rx1, rx2 = max(0, rx1), min(W, max(rx2, rx1 + 1))
    ry1, ry2 = max(0, ry1), min(H, max(ry2, ry1 + 1))
    patch = depth_m[ry1:ry2, rx1:rx2]
    vals = patch[(patch > min_r) & (patch < max_r)]
    if vals.size < 5:
        return None
    return float(np.percentile(vals, 30))


def range_from_height(box, focal_px, assumed_height_m, image_height):
    """Pinhole estimate z = f * H / h. Flags the estimate when the box is cut off."""
    h = max(1.0, box[3] - box[1])
    truncated = box[1] <= 2 or box[3] >= image_height - 2
    return focal_px * assumed_height_m / h, truncated


# --------------------------------------------------------------------------- frames

def frame_transforms(response):
    snap = response.shot.transforms_snapshot
    odom_T_cam = get_a_tform_b(snap, ODOM_FRAME_NAME, response.shot.frame_name_image_sensor)
    odom_T_body = get_odom_tform_body(snap)
    return odom_T_cam, odom_T_body


def yaw_of(se3):
    """Yaw of an SE3Pose rotation (rad)."""
    q = se3.rotation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
