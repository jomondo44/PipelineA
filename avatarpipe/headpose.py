"""Head rotation from MediaPipe's face transform, as directions in Blender camera space.

The face transform maps MediaPipe's canonical face (x = the person's left, y = up,
z = out of the face) into a camera space with x right, y up, looking down -z,
which is the same axis convention as a Blender camera's local space.

Body landmarks only give a rough head direction (ears and nose). The face tracker
measures the head's turn, nod and tilt, so this drives the head bone when a face
is visible and fades back to the body estimate when it isn't.
"""
from __future__ import annotations

import logging

import numpy as np

from .filters import fill_gaps, one_euro_bidirectional

log = logging.getLogger("avatarpipe")


def rotation_from_matrix(m: np.ndarray) -> np.ndarray | None:
    """3x3 rotation from a 4x4 face transform (either storage order), or None."""
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        return None
    if np.abs(m[3, :3]).sum() > np.abs(m[:3, 3]).sum():  # translation in the bottom row: transposed
        m = m.T
    u, _, vt = np.linalg.svd(m[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def crop_correction(cxn: float, cyn: float, w: int, h: int, f: float) -> np.ndarray:
    """Rotation from a crop's virtual camera to the real camera.

    The face tracker treats the crop's center as straight ahead. A head at the edge
    of a wide shot is actually seen off-axis, so its true rotation is this matrix
    times the measured one. (cxn, cyn) = crop center as 0-1 of the frame."""
    d = np.array([(cxn - 0.5) * w / f, -(cyn - 0.5) * h / f, -1.0])
    d /= np.linalg.norm(d)
    a = np.array([0.0, 0.0, -1.0])
    v = np.cross(a, d)
    s, c = np.linalg.norm(v), float(np.dot(a, d))
    if s < 1e-9:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)


def _to_quat(r: np.ndarray) -> np.ndarray:
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def head_directions(face_mat: np.ndarray, face_crop: np.ndarray | None, w: int, h: int, f: float,
                    fps: float, body_left: np.ndarray, body_up: np.ndarray):
    """Per-frame head 'up' and 'left' unit vectors in Blender camera space, plus a 0-1
    confidence (1 where the face was tracked, fading over gaps).

    body_left / body_up: (T,3) rough directions from the ears and shoulders, used to
    check the face axes point the right way (and flip them if a convention differs)."""
    T = face_mat.shape[0]
    quats = np.full((T, 4), np.nan)
    for t in range(T):
        r = rotation_from_matrix(face_mat[t])
        if r is None:
            continue
        if face_crop is not None and np.isfinite(face_crop[t]).all():
            r = crop_correction(float(face_crop[t, 0]), float(face_crop[t, 1]), w, h, f) @ r
        quats[t] = _to_quat(r)
    valid = np.isfinite(quats).all(1)
    if valid.sum() < 3:
        return None
    prev = None
    for t in range(T):  # keep consecutive quaternions in the same hemisphere so smoothing works
        if valid[t]:
            if prev is not None and np.dot(quats[t], prev) < 0:
                quats[t] = -quats[t]
            prev = quats[t]
    quats = one_euro_bidirectional(fill_gaps(quats), fps, min_cutoff=1.5, beta=0.3)
    rots = np.stack([_from_quat(q) for q in quats])
    up, left = rots[:, :, 1], rots[:, :, 0]

    def agree(face_vecs, body_vecs):
        b = body_vecs[valid]
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
        return float(np.nanmean(np.sum(face_vecs[valid] * b, axis=1)))

    if agree(left, body_left) < 0:
        left = -left
        log.info("  head: flipped face side axis to match your ears")
    if agree(up, body_up) < 0:
        up = -up
        log.info("  head: flipped face up axis to match your shoulders")
    k = 9  # ~0.3 s fade between face-driven and body-driven head
    conf = np.convolve(valid.astype(float), np.ones(k) / k, mode="same")
    return up, left, conf
