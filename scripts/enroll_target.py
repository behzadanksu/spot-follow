#!/usr/bin/env python3
"""Enroll the person Spot should follow in `face` mode.

Give it 3-10 photos of the person (different angles/lighting, one face per photo),
or capture from a laptop webcam, or straight from Spot's cameras.

  python scripts/enroll_target.py --images data/enroll/*.jpg
  python scripts/enroll_target.py --webcam                 # SPACE = capture, q = finish
  python scripts/enroll_target.py --spot --camera frontleft  # stand ~1.5 m in front of Spot

Writes data/target_face.npy (one normalized embedding per photo).
"""
import argparse
import glob
import os
import time

import _common  # noqa: F401
from spot_follow.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', nargs='*', default=[])
    p.add_argument('--webcam', action='store_true')
    p.add_argument('--webcam-index', type=int, default=0)
    p.add_argument('--spot', action='store_true', help='capture from Spot (5 shots, 1 s apart)')
    p.add_argument('--camera', default='frontleft')
    p.add_argument('--hostname')
    p.add_argument('--config')
    p.add_argument('--out')
    args = p.parse_args()
    _common.setup_logging()
    cfg = load_config(args.config, {'robot.hostname': args.hostname})

    import cv2
    import numpy as np
    from spot_follow.detectors import FaceIdentifier

    f = cfg.face
    fid = FaceIdentifier(f.detector_model, f.recognizer_model, None, f.head_upscale)
    embeddings = []

    def add(image, tag):
        _, faces = fid.faces(image)
        if len(faces) == 0:
            # Small faces in wide shots: try again upscaled.
            image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, faces = fid.faces(image)
        if len(faces) == 0:
            print(f'[!! ] {tag}: no face found')
            return
        if len(faces) > 1:
            print(f'[ ! ] {tag}: {len(faces)} faces, using the largest')
        emb = fid.largest_face_embedding(image)
        embeddings.append(emb)
        print(f'[ok ] {tag}: face enrolled ({len(embeddings)} total)')

    paths = []
    for pattern in args.images:
        paths.extend(sorted(glob.glob(pattern)))
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f'[!! ] could not read {path}')
            continue
        add(img, path)

    if args.webcam:
        cap = cv2.VideoCapture(args.webcam_index)
        print('SPACE = capture, q = finish')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow('enroll (SPACE capture, q quit)', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                add(frame, f'webcam #{len(embeddings) + 1}')
            elif k == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    if args.spot:
        from spot_follow.follower import build_frames
        from spot_follow.robot import SpotInterface
        spot = SpotInterface(cfg).connect()
        reqs = spot.image_requests([args.camera], with_depth=False, quality=95)
        for i in range(5):
            fr = build_frames(spot.get_images(reqs), [args.camera])[0]
            add(fr.upright, f'spot {args.camera} #{i + 1}')
            time.sleep(1.0)

    if not embeddings:
        raise SystemExit('No faces enrolled - nothing written.')
    E = np.stack(embeddings)
    if len(E) > 1:
        sims = E @ E.T
        off = sims[~np.eye(len(E), dtype=bool)]
        print(f'Self-similarity between enrolled photos: min {off.min():.2f}, mean {off.mean():.2f} '
              f'(threshold {f.match_threshold}). Low values mean a bad photo.')
    out = args.out or f.target_embedding
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.save(out, E)
    print(f'Wrote {out} with {len(E)} embeddings')


if __name__ == '__main__':
    main()
