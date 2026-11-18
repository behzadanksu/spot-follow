#!/usr/bin/env python3
"""Pre-flight check: connectivity, login, E-Stop, lease, battery, cameras.

Saves one upright snapshot per camera to runs/check/ so you can confirm that
people appear upright (the detector needs that). Nothing moves.

  python scripts/check_robot.py
  python scripts/check_robot.py --hostname 192.168.80.3
"""
import argparse
import os

import _common  # noqa: F401
from spot_follow.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config')
    p.add_argument('--hostname')
    args = p.parse_args()
    _common.setup_logging()
    cfg = load_config(args.config, {'robot.hostname': args.hostname})

    import cv2
    from bosdyn.api import estop_pb2
    from spot_follow.cameras import CAMERA_SOURCES
    from spot_follow.follower import build_frames
    from spot_follow.robot import SpotInterface

    ok = True
    spot = SpotInterface(cfg).connect()
    print(f'[ok ] connected and authenticated to {cfg.robot.hostname}')

    batt = spot.battery_percent()
    print(f'[{"ok " if batt is None or batt > 25 else "!! "}] battery: {batt}%')

    level, n = spot.estop_status()
    if n == 0:
        print('[!! ] no E-Stop endpoint registered (keep the tablet open, or use --own-estop)')
        ok = False
    else:
        print(f'[{"ok " if level == estop_pb2.ESTOP_LEVEL_NONE else "!! "}] E-Stop endpoints: {n}, '
              f'level: {estop_pb2.EstopStopLevel.Name(level)}')

    for res in spot.lease_client.list_leases():
        owner = res.lease_owner.client_name or '(nobody)'
        if res.resource == 'body':
            print(f'[ok ] body lease held by: {owner}  '
                  f'(release control on the tablet, or use --take-lease)')

    sources = set(spot.list_image_sources())
    for cam in cfg.cameras:
        vis, depth, _ = CAMERA_SOURCES[cam]
        print(f'[{"ok " if vis in sources else "!! "}] {vis}   depth: {"yes" if depth in sources else "NO"}')
        ok &= vis in sources

    os.makedirs('runs/check', exist_ok=True)
    frames = build_frames(spot.get_images(spot.image_requests(cfg.cameras, True, 90)), cfg.cameras)
    for fr in frames:
        path = f'runs/check/{fr.name}_upright.jpg'
        cv2.imwrite(path, fr.upright)
        if fr.upright_depth_m is not None:
            vis = cv2.applyColorMap(cv2.convertScaleAbs(fr.upright_depth_m, alpha=255 / 5.0),
                                    cv2.COLORMAP_JET)
            cv2.imwrite(f'runs/check/{fr.name}_depth.jpg', vis)
        model = fr.response.source.WhichOneof('camera_models')
        print(f'      saved {path}  ({fr.upright.shape[1]}x{fr.upright.shape[0]}, model={model}, '
              f'f={fr.focal_px:.0f}px)')
    print('\nOpen runs/check/*_upright.jpg - people and walls must look upright.')
    print('ALL GOOD' if ok else 'FIX THE !! ITEMS ABOVE BEFORE RUNNING')


if __name__ == '__main__':
    main()
