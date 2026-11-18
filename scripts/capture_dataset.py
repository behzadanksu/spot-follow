#!/usr/bin/env python3
"""Collect training images of Charlie (or anything) from Spot's own cameras.

Images are saved upright, exactly as the follower's detector will see them, which
is the single biggest factor in how well a custom model works on the robot.

  python scripts/capture_dataset.py --out data/charlie_raw --seconds 120 --interval 0.5
Move Charlie around Spot at 1-6 m, different angles, lighting, partly occluded.
Then label the images (Roboflow, CVAT or Label Studio, YOLO format) and run
scripts/train_custom.py. Nothing moves.
"""
import argparse
import os
import time

import _common  # noqa: F401
from spot_follow.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/charlie_raw')
    p.add_argument('--seconds', type=float, default=120)
    p.add_argument('--interval', type=float, default=0.5)
    p.add_argument('--cameras', nargs='+')
    p.add_argument('--hostname')
    p.add_argument('--config')
    args = p.parse_args()
    _common.setup_logging()
    cfg = load_config(args.config, {'robot.hostname': args.hostname, 'cameras': args.cameras})

    import cv2
    from spot_follow.follower import build_frames
    from spot_follow.robot import SpotInterface

    os.makedirs(args.out, exist_ok=True)
    spot = SpotInterface(cfg).connect()
    reqs = spot.image_requests(cfg.cameras, with_depth=False, quality=90)
    t_end, n = time.time() + args.seconds, 0
    while time.time() < t_end:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        for fr in build_frames(spot.get_images(reqs), cfg.cameras):
            cv2.imwrite(os.path.join(args.out, f'{stamp}_{n:05d}_{fr.name}.jpg'), fr.upright)
        n += 1
        time.sleep(args.interval)
    print(f'Saved {n} rounds x {len(cfg.cameras)} cameras to {args.out}/')
    print('Tip: delete frames where Charlie is not visible before labelling (keep ~10% as negatives).')


if __name__ == '__main__':
    main()
