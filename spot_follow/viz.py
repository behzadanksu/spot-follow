"""Debug view: every camera upright in one mosaic, with boxes and the robot's state."""
import math
import os
import time

import cv2
import numpy as np

STATE_COLORS = {'SEARCH': (0, 200, 255), 'FOLLOW': (0, 220, 0), 'REACQUIRE': (0, 140, 255),
                'IDLE': (160, 160, 160)}


def _tile(img, h):
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h))


class Viewer:

    def __init__(self, show=False, save_dir='', save_every_n=5, tile_height=360):
        self.show, self.save_dir, self.every = show, save_dir, max(1, int(save_every_n))
        self.tile_h = tile_height
        self.quit_requested = False
        self.n = 0
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def render(self, frames, dets_by_cam, state, track, goal, pose, unseen):
        tiles = []
        for fr in frames:
            img = fr.upright.copy()
            for det in dets_by_cam.get(fr.name, []):
                x1, y1, x2, y2 = (int(v) for v in det.box)
                is_target = track is not None and track.last_camera == fr.name and unseen < 0.5
                color = (0, 255, 0) if is_target else (255, 160, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f'{det.label} {det.score:.2f}'
                if det.identity is not None:
                    label += f' id={det.identity:.2f}'
                cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, fr.name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(_tile(img, self.tile_h))
        if not tiles:
            return None
        row = np.hstack(tiles)
        bar = np.zeros((70, row.shape[1], 3), np.uint8)
        cv2.putText(bar, state, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, STATE_COLORS.get(state, (255, 255, 255)), 3)
        info = f'robot ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f} deg)'
        if track is not None:
            info += f' | target ({track.pos[0]:.2f}, {track.pos[1]:.2f}) unseen {unseen:.1f}s'
        if goal is not None:
            info += f' | {goal.action} dist {goal.distance:.2f}m'
        cv2.putText(bar, info, (260, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        return np.vstack([bar, row])

    def update(self, *args):
        self.n += 1
        if not self.show and not (self.save_dir and self.n % self.every == 0):
            return
        canvas = self.render(*args)
        if canvas is None:
            return
        if self.save_dir and self.n % self.every == 0:
            cv2.imwrite(os.path.join(self.save_dir, f'{time.strftime("%H%M%S")}_{self.n:05d}.jpg'), canvas)
        if self.show:
            cv2.imshow('spot-follow (q = quit)', canvas)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                self.quit_requested = True

    def close(self):
        if self.show:
            cv2.destroyAllWindows()
