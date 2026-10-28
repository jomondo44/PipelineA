"""Synthetic performer: a known 3D skeleton moving in front of a known camera.

Used to test the solve (we know the true root position) and to run the whole pipeline
end to end without needing MediaPipe model downloads.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from avatarpipe.landmarks import ARKIT, IDX, POSE

# Rest skeleton in MediaPipe world axes: hip-centered meters, x = image right
# (performer's left when facing camera), y down, z away from camera.
REST = {
    "nose": (0, -0.70, -0.09), "l_eye_inner": (0.015, -0.73, -0.08), "l_eye": (0.03, -0.73, -0.075),
    "l_eye_outer": (0.045, -0.73, -0.07), "r_eye_inner": (-0.015, -0.73, -0.08), "r_eye": (-0.03, -0.73, -0.075),
    "r_eye_outer": (-0.045, -0.73, -0.07), "l_ear": (0.075, -0.71, 0.0), "r_ear": (-0.075, -0.71, 0.0),
    "mouth_l": (0.025, -0.66, -0.075), "mouth_r": (-0.025, -0.66, -0.075),
    "l_shoulder": (0.18, -0.50, 0.0), "r_shoulder": (-0.18, -0.50, 0.0),
    "l_elbow": (0.20, -0.24, 0.0), "r_elbow": (-0.20, -0.24, 0.0),
    "l_wrist": (0.22, 0.01, 0.0), "r_wrist": (-0.22, 0.01, 0.0),
    "l_pinky": (0.23, 0.09, 0.01), "r_pinky": (-0.23, 0.09, 0.01),
    "l_index": (0.225, 0.10, -0.02), "r_index": (-0.225, 0.10, -0.02),
    "l_thumb": (0.21, 0.06, -0.03), "r_thumb": (-0.21, 0.06, -0.03),
    "l_hip": (0.10, 0.0, 0.0), "r_hip": (-0.10, 0.0, 0.0),
    "l_knee": (0.10, 0.45, 0.0), "r_knee": (-0.10, 0.45, 0.0),
    "l_ankle": (0.10, 0.88, 0.02), "r_ankle": (-0.10, 0.88, 0.02),
    "l_heel": (0.10, 0.93, 0.05), "r_heel": (-0.10, 0.93, 0.05),
    "l_foot_index": (0.10, 0.94, -0.12), "r_foot_index": (-0.10, 0.94, -0.12),
}
ARM = {"l": ["l_elbow", "l_wrist", "l_pinky", "l_index", "l_thumb"],
       "r": ["r_elbow", "r_wrist", "r_pinky", "r_index", "r_thumb"]}


def skeleton(t: float) -> np.ndarray:
    """(33,3) world landmarks at time t: arms swing out to the side, one waves."""
    p = np.array([REST[n] for n in POSE], dtype=np.float64)
    for side, sign in (("l", 1), ("r", -1)):
        sh = p[IDX[f"{side}_shoulder"]].copy()
        ang = (0.6 + 0.5 * math.sin(2 * math.pi * 0.5 * t + (0 if side == "l" else 1.0))) * sign
        c, s = math.cos(ang), math.sin(ang)
        for n in ARM[side]:
            v = p[IDX[n]] - sh
            # rotate in the image (x,y) plane about the shoulder: arm raises sideways
            p[IDX[n]] = sh + np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
    return p


def root(t: float) -> np.ndarray:
    """True hip position in OpenCV camera space (x right, y down, z forward)."""
    return np.array([0.35 * math.sin(2 * math.pi * 0.2 * t), 0.25, 3.2 + 0.4 * math.sin(2 * math.pi * 0.1 * t)])


def project(P: np.ndarray, f: float, w: int, h: int) -> np.ndarray:
    return np.stack([f * P[:, 0] / P[:, 2] + w / 2, f * P[:, 1] / P[:, 2] + h / 2], -1)


def tracking(n: int, fps: float, w: int, h: int, focal_mm: float, noise_px: float = 0.0, seed: int = 0):
    """Return (pose_world, pose_image, face_bs, roots) shaped like track.py output."""
    rng = np.random.default_rng(seed)
    f = focal_mm / 36.0 * max(w, h)
    world = np.zeros((n, 33, 4), np.float32)
    image = np.zeros((n, 33, 4), np.float32)
    roots = np.zeros((n, 3))
    for i in range(n):
        t = i / fps
        P = skeleton(t)
        R = root(t)
        uv = project(P + R, f, w, h) + rng.normal(0, noise_px, (33, 2))
        world[i, :, :3], world[i, :, 3] = P, 0.99
        image[i, :, 0], image[i, :, 1] = uv[:, 0] / w, uv[:, 1] / h
        image[i, :, 3] = 0.99
        roots[i] = R
    bs = np.zeros((n, len(ARKIT)), np.float32)
    ts = np.arange(n) / fps
    bs[:, ARKIT.index("jawOpen")] = 0.5 + 0.5 * np.sin(2 * np.pi * 1.0 * ts)
    bs[:, ARKIT.index("eyeBlinkLeft")] = (np.sin(2 * np.pi * 0.7 * ts) > 0.95).astype(np.float32)
    bs[:, ARKIT.index("eyeBlinkRight")] = bs[:, ARKIT.index("eyeBlinkLeft")]
    return world, image, bs, roots


BONES = [("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
         ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
         ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
         ("l_ankle", "l_foot_index"), ("r_ankle", "r_foot_index")]


def background(w: int, h: int) -> np.ndarray:
    y = np.linspace(0, 1, h)[:, None]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (90 + 80 * y).astype(np.uint8)
    img[..., 1] = (120 + 60 * y).astype(np.uint8)
    img[..., 2] = (160 - 40 * y).astype(np.uint8)
    img[int(h * 0.72):] = (95, 85, 70)  # floor
    cv2.rectangle(img, (int(w * 0.08), int(h * 0.25)), (int(w * 0.25), int(h * 0.72)), (60, 50, 40), -1)  # door
    return img


def draw_person(img: np.ndarray, image_lm: np.ndarray, thick: int, color=(230, 200, 170)) -> None:
    h, w = img.shape[:2]
    pt = lambda n: (int(image_lm[IDX[n], 0] * w), int(image_lm[IDX[n], 1] * h))  # noqa: E731
    for a, b in BONES:
        cv2.line(img, pt(a), pt(b), color, thick, cv2.LINE_AA)
    cv2.circle(img, pt("nose"), thick, color, -1, cv2.LINE_AA)
