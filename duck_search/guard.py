"""Depth-only braking layer shared by both physical benchmark controllers."""

import numpy as np

from .env import FOV


def guarded_action(requested: int, ranges) -> int:
    """Replace unsafe advance with a turn; never change a turn or confirmation.

    Endpoints inside a 36 cm wide, 32 cm long forward corridor imply insufficient
    clearance for the robot plus its next control interval. No object position,
    simulator contact or evaluator ground truth is used here.
    """
    if requested not in (0, 3, 4):
        return requested
    ranges = np.asarray(ranges)
    angles = np.linspace(-FOV / 2, FOV / 2, len(ranges))
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    blocked = np.any((x < 0.32) & (np.abs(y) < 0.18))
    if not blocked:
        return requested
    # Positive bearing = left. Depth rays run from right to left.
    return (
        1
        if ranges[len(ranges) // 2 :].mean() >= ranges[: len(ranges) // 2].mean()
        else 2
    )
