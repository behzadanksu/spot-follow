"""Object detectors and face identity.

All detectors take an upright BGR image and return Detection objects in that
image's pixel coordinates. Heavy imports (ultralytics/torch) happen lazily so
the rest of the package works without them (e.g. for unit tests).
"""
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    box: tuple                 # (x1, y1, x2, y2) in upright pixels
    score: float
    label: str
    identity: Optional[float] = None   # face similarity to the enrolled target, if a face was seen
    appearance: Optional[np.ndarray] = None
    extras: dict = field(default_factory=dict)

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


# --------------------------------------------------------------------------- YOLO family

class YoloDetector:
    """Ultralytics YOLO. Used for 'person', 'face' (person boxes) and 'custom' modes."""

    def __init__(self, model_path, target_classes, confidence=0.45, imgsz=640, device=''):
        from ultralytics import YOLO  # lazy import
        self.model = YOLO(model_path)
        self.names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        wanted = {c.lower() for c in target_classes}
        self.class_ids = [i for i, n in self.names.items() if n.lower() in wanted]
        if not self.class_ids:
            raise ValueError(f'None of {target_classes} are classes of {model_path}; '
                             f'it has: {sorted(self.names.values())}')
        self.confidence, self.imgsz, self.device = confidence, imgsz, device
        log.info('Detector %s targeting classes %s', model_path, [self.names[i] for i in self.class_ids])

    def detect(self, image) -> List[Detection]:
        kwargs = dict(conf=self.confidence, imgsz=self.imgsz, classes=self.class_ids, verbose=False)
        if self.device:
            kwargs['device'] = self.device
        result = self.model.predict(image, **kwargs)[0]
        out = []
        for b in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(Detection((x1, y1, x2, y2), float(b.conf[0]), self.names[int(b.cls[0])]))
        return out


class OpenVocabDetector(YoloDetector):
    """YOLO-World: detect objects described in plain text, no training needed."""

    def __init__(self, model_path, prompts, confidence=0.25, imgsz=640, device=''):
        from ultralytics import YOLOWorld  # lazy import
        self.model = YOLOWorld(model_path)
        self.model.set_classes(list(prompts))
        self.names = dict(enumerate(prompts))
        self.class_ids = list(self.names.keys())
        self.confidence, self.imgsz, self.device = confidence, imgsz, device
        log.info('Open-vocabulary detector prompts: %s', prompts)


# --------------------------------------------------------------------------- appearance

def appearance_descriptor(image, box):
    """Small, normalized histogram of the torso region; cheap re-identification cue.

    Spot's body cameras are greyscale, so this uses intensity only. It cannot tell
    two people apart reliably, but it does help pick the right one when two
    candidates are equally close to the predicted position.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1, y2 - y1
    H, W = image.shape[:2]
    rx1, rx2 = max(0, x1 + w // 5), min(W, x2 - w // 5)
    ry1, ry2 = max(0, y1 + h // 5), min(H, y1 + (3 * h) // 5)
    if rx2 - rx1 < 4 or ry2 - ry1 < 4:
        return None
    patch = image[ry1:ry2, rx1:rx2]
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch = cv2.equalizeHist(patch)
    hist = cv2.calcHist([patch], [0], None, [32], [0, 256]).flatten()
    return hist / (np.linalg.norm(hist) + 1e-9)


def appearance_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


# --------------------------------------------------------------------------- face identity

class FaceIdentifier:
    """OpenCV YuNet (face detection) + SFace (face embedding). CPU-friendly, no torch."""

    def __init__(self, detector_model, recognizer_model, target_embedding=None, head_upscale=2.0):
        for p in (detector_model, recognizer_model):
            if not os.path.exists(p):
                raise FileNotFoundError(f'{p} missing - run: python scripts/download_models.py')
        self.det = cv2.FaceDetectorYN.create(detector_model, '', (320, 320), 0.6, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(recognizer_model, '')
        self.head_upscale = head_upscale
        self.targets = None
        if target_embedding:
            if not os.path.exists(target_embedding):
                raise FileNotFoundError(f'{target_embedding} missing - run scripts/enroll_target.py first')
            self.targets = np.load(target_embedding).astype(np.float32)
            if self.targets.ndim == 1:
                self.targets = self.targets[None, :]

    def faces(self, image):
        """All faces in an image: array of YuNet rows (x, y, w, h, 10 landmarks, score)."""
        img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.det.detect(img)
        return img, (faces if faces is not None else np.zeros((0, 15), np.float32))

    def embed(self, image, face_row):
        aligned = self.rec.alignCrop(image, face_row)
        feat = self.rec.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9)

    def largest_face_embedding(self, image):
        img, faces = self.faces(image)
        if len(faces) == 0:
            return None
        row = max(faces, key=lambda f: f[2] * f[3])
        return self.embed(img, row)

    def similarity(self, feat):
        if self.targets is None or feat is None:
            return None
        return float(np.max(self.targets @ feat))

    def identify_in_box(self, image, box):
        """Similarity of the face inside a person box to the enrolled target (None if no face)."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        H, W = image.shape[:2]
        # Head region: top ~40% of the person box, widened a little.
        hx1, hx2 = int(max(0, x1 - 0.1 * w)), int(min(W, x2 + 0.1 * w))
        hy1, hy2 = int(max(0, y1 - 0.05 * h)), int(min(H, y1 + 0.45 * h))
        if hx2 - hx1 < 8 or hy2 - hy1 < 8:
            return None
        crop = image[hy1:hy2, hx1:hx2]
        if self.head_upscale and self.head_upscale != 1.0:
            crop = cv2.resize(crop, None, fx=self.head_upscale, fy=self.head_upscale,
                              interpolation=cv2.INTER_CUBIC)
        return self.similarity(self.largest_face_embedding(crop))


# --------------------------------------------------------------------------- factory

class TargetDetector:
    """Wraps the configured detector (+ face identity) behind one detect() call."""

    def __init__(self, cfg):
        d = cfg.detector
        self.mode = d.mode
        self.face = None
        if self.mode in ('person', 'face'):
            self.det = YoloDetector(d.model, [d.person_class], d.confidence, d.imgsz, d.device)
        elif self.mode == 'custom':
            self.det = YoloDetector(d.custom_model, d.custom_class_names, d.confidence, d.imgsz, d.device)
        elif self.mode == 'open_vocab':
            self.det = OpenVocabDetector(d.open_vocab_model, d.open_vocab_prompts,
                                         min(d.confidence, 0.25), d.imgsz, d.device)
        else:
            raise ValueError(f'Unknown detector.mode {self.mode!r}')
        if self.mode == 'face':
            f = cfg.face
            self.face = FaceIdentifier(f.detector_model, f.recognizer_model,
                                       f.target_embedding, f.head_upscale)

    def detect(self, image) -> List[Detection]:
        dets = self.det.detect(image)
        for det in dets:
            det.appearance = appearance_descriptor(image, det.box)
            if self.face is not None:
                det.identity = self.face.identify_in_box(image, det.box)
        return dets
