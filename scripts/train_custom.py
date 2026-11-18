#!/usr/bin/env python3
"""Fine-tune a YOLO detector on your labelled Charlie images.

Expects a YOLO-format dataset (what Roboflow/CVAT/Label Studio export), e.g.
  data/charlie/
    data.yaml          # names: ['charlie'], train: images/train, val: images/val
    images/train/*.jpg   labels/train/*.txt
    images/val/*.jpg     labels/val/*.txt

  python scripts/train_custom.py --data data/charlie/data.yaml --epochs 60
  -> models/charlie.pt, then:
  python scripts/run_follow.py --mode custom --custom-model models/charlie.pt --show

A GPU (or Google Colab) makes this minutes instead of hours; 150-300 labelled
images is usually enough for a single distinctive object like a mascot.
"""
import argparse
import shutil

import _common  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--base', default='yolov8n.pt', help='starting weights (yolov8s.pt = more accurate)')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='')
    p.add_argument('--out', default='models/charlie.pt')
    args = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.base)
    kwargs = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, project='runs/train',
                  name='custom', exist_ok=True,
                  hsv_h=0.0, hsv_s=0.0,      # Spot body cameras are greyscale: colour jitter is useless
                  degrees=10.0, fliplr=0.5)
    if args.device:
        kwargs['device'] = args.device
    model.train(**kwargs)
    best = model.trainer.best
    shutil.copy(best, args.out)
    print(f'Best weights copied to {args.out}')
    metrics = YOLO(args.out).val(data=args.data, imgsz=args.imgsz, verbose=False)
    print(f'Validation mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}')


if __name__ == '__main__':
    main()
