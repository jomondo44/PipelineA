"""Face close-up: find the head from body landmarks, keep the crop steady, cut it at full resolution.

In a full-body shot the face is only a few dozen pixels wide in the working-size
video, which is too small for expressions. The source video is usually 1080p or
4K, so cropping the head from it gives the face tracker 3-5x more pixels.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .landmarks import IDX

MIN_VIS = 0.3


def head_box(lm: np.ndarray, W: int, H: int) -> tuple[float, float, float] | None:
    """Square head box (center x, center y, side) in full-res pixels from one frame's
    normalized pose landmarks (33, 4: x, y, z, visibility). None if the head isn't visible."""
    if lm is None or not np.isfinite(lm[:, :2]).all():
        return None

    def pt(name):
        p = lm[IDX[name]]
        return np.array([p[0] * W, p[1] * H]) if p[3] >= MIN_VIS else None

    def dist(a, b):
        return float(np.linalg.norm(a - b)) if a is not None and b is not None else 0.0

    nose, le, re = pt("nose"), pt("l_ear"), pt("r_ear")
    leo, reo = pt("l_eye_outer"), pt("r_eye_outer")
    ls, rs = pt("l_shoulder"), pt("r_shoulder")
    face = [p for p in (nose, le, re, leo, reo, pt("l_eye"), pt("r_eye")) if p is not None]
    if len(face) < 2:
        return None
    center = 0.5 * (le + re) if le is not None and re is not None else np.mean(face, axis=0)
    sho_mid = 0.5 * (ls + rs) if ls is not None and rs is not None else None
    size = max(2.4 * dist(le, re), 3.2 * dist(leo, reo), 1.3 * dist(nose, sho_mid), 48.0)
    return float(center[0]), float(center[1] - 0.08 * size), float(size)


class BoxSmoother:
    """Keeps the crop steady: jittery crops make the face appear to shake."""

    def __init__(self, a_center: float = 0.5, a_size: float = 0.25, hold: int = 30):
        self.a_c, self.a_s, self.hold = a_center, a_size, hold
        self.box = None
        self.missed = 0

    def update(self, box):
        if box is None:
            self.missed += 1
            if self.missed > self.hold:
                self.box = None
            return self.box
        self.missed = 0
        if self.box is None:
            self.box = box
        else:
            cx, cy, s = self.box
            self.box = (cx + self.a_c * (box[0] - cx), cy + self.a_c * (box[1] - cy), s + self.a_s * (box[2] - s))
        return self.box


def extract(frame: np.ndarray, box, out: int) -> np.ndarray:
    """Cut the square box from frame (padding past the edges by repeating them) and
    resize to out x out."""
    H, W = frame.shape[:2]
    cx, cy, s = box
    half = s / 2.0
    x0, y0 = int(math.floor(cx - half)), int(math.floor(cy - half))
    x1, y1 = int(math.ceil(cx + half)), int(math.ceil(cy + half))
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - W), max(0, y1 - H)
    crop = frame[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if crop.size == 0:
        return np.zeros((out, out, 3), np.uint8)
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    interp = cv2.INTER_AREA if crop.shape[0] > out else cv2.INTER_CUBIC
    return np.ascontiguousarray(cv2.resize(crop, (out, out), interpolation=interp))
