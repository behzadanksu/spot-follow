"""Pure-math follow controller: target position -> walking goal for Spot.

Kept free of any robot or SDK calls so it can be unit tested on a laptop.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Goal:
    x: float
    y: float
    heading: float
    action: str            # 'approach' | 'backoff' | 'face' | 'hold'
    distance: float        # current robot -> target distance
    leashed: bool = False


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clamp_to_leash(point, center, radius):
    """Pull `point` back onto the circle of `radius` around `center` if it is outside."""
    if radius is None or radius <= 0:
        return point, False
    d = point - center
    n = float(np.linalg.norm(d))
    if n <= radius:
        return point, False
    return center + d * (radius / n), True


def compute_goal(robot_xy, robot_yaw, target_xy, fcfg, leash_center=None):
    """Where should Spot stand, and which way should it face?

    Spot aims for a point `standoff_m` short of the target along the line
    between them, facing the target. Inside the deadband it only turns.
    """
    robot_xy = np.asarray(robot_xy, float)
    target_xy = np.asarray(target_xy, float)
    d = target_xy - robot_xy
    dist = float(np.linalg.norm(d))
    heading = math.atan2(d[1], d[0]) if dist > 1e-6 else robot_yaw
    standoff, band = fcfg.standoff_m, fcfg.deadband_m

    if dist > standoff + band:
        goal, action = target_xy - d / dist * standoff, 'approach'
    elif dist < standoff - band and fcfg.allow_backoff:
        goal, action = target_xy - d / max(dist, 1e-6) * standoff, 'backoff'
    else:
        goal = robot_xy.copy()
        action = 'face' if abs(wrap_angle(heading - robot_yaw)) > fcfg.min_heading_change_rad else 'hold'

    leashed = False
    if leash_center is not None:
        goal, leashed = clamp_to_leash(goal, np.asarray(leash_center, float), fcfg.leash_radius_m)
    return Goal(float(goal[0]), float(goal[1]), float(heading), action, dist, leashed)


def goal_changed(prev: Goal, new: Goal, fcfg):
    if prev is None or prev.action != new.action:
        return True
    moved = math.hypot(new.x - prev.x, new.y - prev.y) > fcfg.min_goal_change_m
    turned = abs(wrap_angle(new.heading - prev.heading)) > fcfg.min_heading_change_rad
    return moved or turned
