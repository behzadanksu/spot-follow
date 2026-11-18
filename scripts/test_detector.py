#!/usr/bin/env python3
"""Try the detector WITHOUT a robot: laptop webcam, a video, or a folder of images.

  python scripts/test_detector.py --webcam
  python scripts/test_detector.py --webcam --mode face        # after enroll_target.py
  python scripts/test_detector.py --images runs/check/*_upright.jpg
  python scripts/test_detector.py --video hallway.mp4 --mode open_vocab --prompts "horse mascot costume"
  python scripts/test_detector.py --webcam --grey             # see it the way Spot's cameras do
"""
import argparse
import glob
import time

import _common  # noqa: F401
from spot_follow.config import load_config


def draw(img, dets, cv2):
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        ok = d.identity is not None and d.identity >= 0.36
        color = (0, 255, 0) if ok else (255, 160, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'{d.label} {d.score:.2f}' + (f' id={d.identity:.2f}' if d.identity is not None else '')
        cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--webcam', action='store_true')
    p.add_argument('--webcam-index', type=int, default=0)
    p.add_argument('--video')
    p.add_argument('--images', nargs='*', default=[])
    p.add_argument('--mode', choices=['person', 'face', 'custom', 'open_vocab'])
    p.add_argument('--custom-model')
    p.add_argument('--prompts', nargs='+')
    p.add_argument('--grey', action='store_true', help='convert to greyscale like Spot body cameras')
    p.add_argument('--config')
    p.add_argument('--out', default='runs/detector_test')
    args = p.parse_args()
    _common.setup_logging()
    cfg = load_config(args.config, {'detector.mode': args.mode,
                                    'detector.custom_model': args.custom_model,
                                    'detector.open_vocab_prompts': args.prompts})

    import os
    import cv2
    from spot_follow.detectors import TargetDetector
    det = TargetDetector(cfg)

    def prep(img):
        if args.grey:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return img

    paths = [q for pat in args.images for q in sorted(glob.glob(pat))]
    if paths:
        os.makedirs(args.out, exist_ok=True)
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                continue
            img = prep(img)
            t = time.time()
            dets = det.detect(img)
            ms = (time.time() - t) * 1000
            print(f'{path}: {len(dets)} detections in {ms:.0f} ms ' +
                  ' '.join(f'[{d.label} {d.score:.2f}' + (f' id={d.identity:.2f}]' if d.identity is not None else ']')
                           for d in dets))
            cv2.imwrite(os.path.join(args.out, os.path.basename(path)), draw(img, dets, cv2))
        print(f'Annotated images in {args.out}/')
        return

    cap = cv2.VideoCapture(args.video if args.video else args.webcam_index)
    if not cap.isOpened():
        raise SystemExit('Could not open video source')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = prep(frame)
        t = time.time()
        dets = det.detect(frame)
        fps = 1.0 / max(time.time() - t, 1e-3)
        draw(frame, dets, cv2)
        cv2.putText(frame, f'{fps:.1f} FPS  mode={cfg.detector.mode}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('detector test (q = quit)', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
