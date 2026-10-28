"""Stage 3: turn raw tracking into clean, camera-space motion the Blender stage can consume.

  1. fill detection gaps, smooth with a bidirectional One Euro filter
  2. place the body in 3D: least-squares solve for the root translation T so that
     (T + world landmarks) projects onto the 2D image landmarks through a pinhole
     camera with the configured focal length (MediaPipe world landmarks are
     already camera-aligned, so this is linear in T)
  3. convert to Blender camera-local axes (X right, Y up, -Z forward)
"""
from __future__ import annotations

import logging

import numpy as np

from ..filters import fill_gaps, one_euro_bidirectional
from ..landmarks import ARKIT, IDX, SOLVE_POINTS, derived, limb_lengths
from ..pipeline import Context

log = logging.getLogger("avatarpipe")
S = "solve"
SENSOR_MM = 36.0


def inputs(ctx: Context) -> list:
    return [ctx.workdir / "track" / "tracking.npz"]


def focal_px(focal_mm: float, w: int, h: int) -> float:
    """35mm-equivalent focal -> pixels, fitted to the long side (matches Blender sensor_fit=AUTO)."""
    return focal_mm / SENSOR_MM * max(w, h)


def solve_root(world: np.ndarray, image: np.ndarray, w: int, h: int, f: float,
               min_vis: float = 0.3) -> np.ndarray:
    """Per-frame root translation in OpenCV camera space (X right, Y down, Z forward).

    world: (T,33,4) hip-centered meters + visibility; image: (T,33,4) normalized.
    Returns (T,3); NaN rows where the solve is not possible.
    """
    T = world.shape[0]
    out = np.full((T, 3), np.nan)
    cx, cy = w / 2.0, h / 2.0
    ids = [IDX[n] for n in SOLVE_POINTS]
    for t in range(T):
        P = world[t, ids, :3]
        uv = image[t, ids, :2] * np.array([w, h])
        vis = np.minimum(world[t, ids, 3], image[t, ids, 3])
        ok = np.isfinite(P).all(1) & np.isfinite(uv).all(1) & (vis >= min_vis)
        if ok.sum() < 4:
            continue
        P, uv, wt = P[ok], uv[ok], np.sqrt(np.clip(vis[ok], 0.05, 1.0))
        du, dv = uv[:, 0] - cx, uv[:, 1] - cy
        # f*X - du*Z = du*Pz - f*Px   ;   f*Y - dv*Z = dv*Pz - f*Py
        A = np.zeros((2 * len(P), 3))
        b = np.zeros(2 * len(P))
        A[0::2, 0], A[0::2, 2], b[0::2] = f, -du, du * P[:, 2] - f * P[:, 0]
        A[1::2, 1], A[1::2, 2], b[1::2] = f, -dv, dv * P[:, 2] - f * P[:, 1]
        A *= np.repeat(wt, 2)[:, None]
        b *= np.repeat(wt, 2)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        if sol[2] > 0.2:  # must be in front of the camera
            out[t] = sol
    return out


def to_blender_cam(p: np.ndarray) -> np.ndarray:
    """OpenCV camera axes (x right, y down, z fwd) -> Blender camera-local (x right, y up, -z fwd)."""
    return p * np.array([1.0, -1.0, -1.0])


def run(ctx: Context) -> list:
    d = np.load(ctx.workdir / "track" / "tracking.npz")
    world, image = d["pose_world"].astype(np.float64), d["pose_image"].astype(np.float64)
    fps = float(d["fps"])
    w, h = (int(v) for v in d["size"])
    scfg, sm = ctx.cfg["solve"], ctx.cfg["track"]["smoothing"]
    T = world.shape[0]
    valid = np.isfinite(world[:, :, :3]).all(axis=(1, 2))

    xyz = fill_gaps(world[:, :, :3])
    xyz = one_euro_bidirectional(xyz, fps, min_cutoff=sm["min_cutoff"], beta=sm["beta"], d_cutoff=sm["d_cutoff"])

    f = focal_px(scfg["focal_mm"], w, h)
    root = solve_root(world, image, w, h, f)
    ok = np.isfinite(root).all(1)
    if not ok.any():
        raise RuntimeError("could not place the body in 3D in any frame (too few visible joints)")
    root = fill_gaps(root)
    # depth (Z) is the noisiest axis from a single camera: smooth it harder
    root[:, :2] = one_euro_bidirectional(root[:, :2], fps, min_cutoff=sm["min_cutoff"], beta=sm["beta"])
    root[:, 2:] = one_euro_bidirectional(root[:, 2:], fps, min_cutoff=sm["min_cutoff"] * 0.3, beta=sm["beta"] * 0.5)

    lengths = limb_lengths(world[:, :, :3])
    if scfg["torso_length_m"]:
        # user-supplied torso length rescales the metric skeleton and the depth together
        k = float(scfg["torso_length_m"]) / lengths["torso_m"]
        xyz *= k
        root *= k
        lengths = {n: v * k for n, v in lengths.items()}

    cam = to_blender_cam(root[:, None, :] + xyz)          # (T,33,3)
    pts = derived(cam)

    bs = d["face_bs"].astype(np.float64)
    shapes = {}
    if np.isfinite(bs).any():
        bs = fill_gaps(bs)
        bs = np.clip(one_euro_bidirectional(bs, fps, min_cutoff=2.0, beta=0.5), 0.0, 1.0)
        shapes = {n: np.round(bs[:, i], 3).tolist() for i, n in enumerate(ARKIT)}

    motion = {
        "fps": fps, "width": w, "height": h, "frames": T,
        "camera": {"focal_mm": scfg["focal_mm"], "sensor_mm": SENSOR_MM,
                   "height_m": scfg["camera_height_m"], "tilt_deg": scfg["camera_tilt_deg"]},
        "performer": lengths,
        "valid": valid.astype(int).tolist(),
        "points": {n: np.round(v, 4).tolist() for n, v in pts.items()},
        "shapes": shapes,
    }
    log.info("  performer: legs %.2fm torso %.2fm; distance from camera %.1f-%.1fm",
             lengths["leg_m"], lengths["torso_m"], np.nanmin(root[:, 2]), np.nanmax(root[:, 2]))
    return [ctx.write_json(S, "motion.json", motion)]
