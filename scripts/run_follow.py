#!/usr/bin/env python3
"""Spot looks around for a target and follows it.

Examples
  # 1) Safe first run: robot can be sitting, nothing moves, watch what it sees
  python scripts/run_follow.py --dry-run --show

  # 2) Follow the nearest person
  python scripts/run_follow.py --show

  # 3) Follow only one enrolled person (after scripts/enroll_target.py)
  python scripts/run_follow.py --mode face --show

  # 4) Charlie the Charger, zero-shot or with a trained model
  python scripts/run_follow.py --mode open_vocab --show
  python scripts/run_follow.py --mode custom --custom-model models/charlie.pt --show
"""
import argparse
import logging
import signal
import sys

import _common  # noqa: F401  (sets sys.path / cwd)
from spot_follow.config import load_config

log = logging.getLogger('run_follow')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=None, help='YAML file layered over config/default.yaml')
    p.add_argument('--hostname', help='robot IP (default from config: 192.168.80.3)')
    p.add_argument('--mode', choices=['person', 'face', 'custom', 'open_vocab'])
    p.add_argument('--dry-run', action='store_true', help='perceive and plan but never move the robot')
    p.add_argument('--show', action='store_true', help='live camera mosaic window (press q to quit)')
    p.add_argument('--save-dir', help='save annotated frames to this folder')
    p.add_argument('--cameras', nargs='+', choices=['frontleft', 'frontright', 'left', 'right', 'back'])
    p.add_argument('--standoff', type=float, help='following distance in metres')
    p.add_argument('--max-speed', type=float, help='max walking speed m/s')
    p.add_argument('--leash', type=float, help='max distance from start position in metres')
    p.add_argument('--custom-model', help='path to a trained YOLO .pt (custom mode)')
    p.add_argument('--custom-classes', nargs='+', help='class names in the custom model to follow')
    p.add_argument('--prompts', nargs='+', help='text prompts for open_vocab mode')
    p.add_argument('--device', help='cpu | cuda:0 | mps')
    p.add_argument('--take-lease', action='store_true', help='take the lease from the tablet forcibly')
    p.add_argument('--own-estop', action='store_true', help='register this program as the E-Stop')
    p.add_argument('--stay-standing', action='store_true', help="don't sit/power off at exit")
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    _common.setup_logging(args.verbose)
    cfg = load_config(args.config, {
        'robot.hostname': args.hostname,
        'robot.take_lease': True if args.take_lease else None,
        'robot.own_estop': True if args.own_estop else None,
        'robot.sit_and_power_off_at_exit': False if args.stay_standing else None,
        'detector.mode': args.mode,
        'detector.custom_model': args.custom_model,
        'detector.custom_class_names': args.custom_classes,
        'detector.open_vocab_prompts': args.prompts,
        'detector.device': args.device,
        'cameras': args.cameras,
        'follow.standoff_m': args.standoff,
        'follow.max_lin_vel': args.max_speed,
        'follow.leash_radius_m': args.leash,
        'viz.show': True if args.show else None,
        'viz.save_dir': args.save_dir,
    })

    # Heavy imports after arg parsing so --help is instant.
    from spot_follow.detectors import TargetDetector
    from spot_follow.follower import Follower
    from spot_follow.robot import EstopNotReady, SpotInterface
    from spot_follow.viz import Viewer
    from spot_follow.cameras import CAMERA_SOURCES

    log.info('Loading detector (%s mode)...', cfg.detector.mode)
    detector = TargetDetector(cfg)

    spot = SpotInterface(cfg).connect()
    available = set(spot.list_image_sources())
    missing = [c for c in cfg.cameras if CAMERA_SOURCES[c][0] not in available]
    if missing:
        log.error('Camera sources not found on this robot: %s', missing)
        sys.exit(2)
    if cfg.range.use_depth:
        no_depth = [c for c in cfg.cameras if CAMERA_SOURCES[c][1] not in available]
        if no_depth:
            log.warning('No depth for %s; range will be estimated from box height', no_depth)
            cfg.range.use_depth = False if len(no_depth) == len(cfg.cameras) else True
    batt = spot.battery_percent()
    log.info('Battery: %s%%', f'{batt:.0f}' if batt is not None else '?')

    viewer = Viewer(cfg.viz.show, cfg.viz.save_dir, cfg.viz.save_every_n) \
        if (cfg.viz.show or cfg.viz.save_dir) else None
    follower = Follower(cfg, spot, detector, viewer, dry_run=args.dry_run)

    # First Ctrl-C: finish the current cycle, stop, sit, power off.
    # Second Ctrl-C (e.g. while it is sitting down): cut motor power immediately.
    def on_sigint(*_):
        if not follower.stop_requested:
            follower.stop_requested = True
            log.info('Ctrl-C: stopping and sitting down (Ctrl-C again = cut power)')
        else:
            log.error('Second Ctrl-C: cutting motor power NOW')
            if not args.dry_run:
                spot.cut_power_now()
            sys.exit(1)
    signal.signal(signal.SIGINT, on_sigint)

    try:
        if args.dry_run:
            follower.run()
        else:
            with spot.motion_session():
                follower.run()
    except EstopNotReady as exc:
        log.error('%s', exc)
        sys.exit(3)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == '__main__':
    main()
