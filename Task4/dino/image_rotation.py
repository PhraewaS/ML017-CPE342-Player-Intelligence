"""Rotation helpers for Task 4 scene fingerprints and canonical image views."""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def padding_angle(image):
    # A straight threshold crossing is expected where the black padding meets the screenshot.
    gray = gaussian_filter(np.asarray(image).mean(axis=2), 1.5)
    corners = [gray[:5, :5], gray[:5, -5:], gray[-5:, :5], gray[-5:, -5:]]
    top_left, top_right, bottom_left, bottom_right = [np.median(corner) for corner in corners]
    padding = min((top_left + bottom_right) / 2, (top_right + bottom_left) / 2)
    if padding > 30:
        return 0.0
    black = gray <= padding + 7
    angles, side_ids = [], []
    offsets = np.arange(13, dtype=float) - 6
    for side, view in enumerate((black, black[:, ::-1], black.T, black.T[:, ::-1])):
        depth = (~view).argmax(axis=1)
        windows = np.lib.stride_tricks.sliding_window_view(depth, len(offsets))[::2]
        slope = (windows * offsets).sum(axis=1) / (offsets @ offsets)
        fitted = windows.mean(axis=1, keepdims=True) + slope[:, None] * offsets
        good = (
            (windows.min(axis=1) >= 4)
            & (windows.max(axis=1) <= 0.55 * view.shape[1])
            & (np.mean(np.abs(windows - fitted), axis=1) <= 0.8)
            & (np.abs(slope) >= 0.10)
        )
        # Convert all four edge orientations into the same x-right, y-down frame.
        sign = -1 if side in (1, 3) else 1
        slope = slope[good] * sign
        bearing = np.degrees(np.arctan(slope)) if side >= 2 else np.degrees(np.arctan2(1, slope))
        a = (bearing + 45) % 90 - 45
        angles.extend(a)
        side_ids.extend([side] * len(a))
    if not angles:
        return 0.0
    angles, side_ids = np.asarray(angles), np.asarray(side_ids)
    centers = np.arange(-44, 45, 2)
    votes = np.abs((angles[:, None] - centers + 45) % 90 - 45) <= 2.5
    best = votes[:, votes.sum(axis=0).argmax()]
    if best.sum() < 7 or len(np.unique(side_ids[best])) < 2:
        return 0.0
    center = centers[votes.sum(axis=0).argmax()]
    median = float((np.median((angles[best] - center + 45) % 90 - 45) + center + 45) % 90 - 45)
    return median if 4 <= abs(median) <= 43 else 0.0


def rotate(image, angle):
    if angle == 0:
        return image
    return image.rotate(angle, resample=Image.Resampling.BILINEAR)
