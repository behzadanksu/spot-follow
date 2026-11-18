#!/usr/bin/env python3
"""Download all model weights ahead of time (do this on a network with internet,
because Spot's own WiFi usually has none).

  python scripts/download_models.py            # YOLO person model + face models
  python scripts/download_models.py --open-vocab   # also YOLO-World for Charlie zero-shot
"""
import argparse
import os
import urllib.request

import _common  # noqa: F401

FACE_MODELS = {
    'models/face_detection_yunet_2023mar.onnx': [
        'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/'
        'face_detection_yunet_2023mar.onnx',
        'https://huggingface.co/opencv/face_detection_yunet/resolve/main/'
        'face_detection_yunet_2023mar.onnx',
    ],
    'models/face_recognition_sface_2021dec.onnx': [
        'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/'
        'face_recognition_sface_2021dec.onnx',
        'https://huggingface.co/opencv/face_recognition_sface/resolve/main/'
        'face_recognition_sface_2021dec.onnx',
    ],
}


def fetch(path, urls):
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        print(f'[ok ] {path} already present')
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for url in urls:
        print(f'[ .. ] downloading {path} from {url.split("/")[2]}')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'spot-follow/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, 'wb') as f:
                f.write(r.read())
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            print(f'[ ! ] {exc}')
            continue
        size = os.path.getsize(path)
        if size > 100_000:
            print(f'[ok ] {path} ({size / 1e6:.1f} MB)')
            return
        print(f'[ ! ] only {size} bytes (Git LFS pointer?), trying next mirror')
    raise SystemExit(f'Could not download {path}. Download it manually from:\n  ' + '\n  '.join(urls))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--open-vocab', action='store_true')
    args = p.parse_args()

    for path, url in FACE_MODELS.items():
        fetch(path, url)

    from ultralytics import YOLO
    YOLO('yolov8n.pt')   # person detector, cached by ultralytics in the working dir
    print('[ok ] yolov8n.pt')
    if args.open_vocab:
        from ultralytics import YOLOWorld
        m = YOLOWorld('yolov8s-worldv2.pt')
        m.set_classes(['horse mascot costume'])   # also pulls the CLIP text encoder
        print('[ok ] yolov8s-worldv2.pt + CLIP text encoder')


if __name__ == '__main__':
    main()
