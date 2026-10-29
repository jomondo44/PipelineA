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

from .. import headpose
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
    feet = feet_in_frame(image)
    lengths["feet_in_frame"] = round(feet, 3)
    lengths["full_body"] = bool(feet >= 0.6)

    cam = to_blender_cam(root[:, None, :] + xyz)          # (T,33,3)
    pts = derived(cam)
    face_crop = d["face_crop"] if "face_crop" in d.files else None
    head_info = apply_head(pts, d["face_mat"], face_crop, w, h, f, fps)

    bs = d["face_bs"].astype(np.float64)
    shapes = {}
    if np.isfinite(bs).any():
        bs = fill_gaps(bs)
        bs = np.clip(one_euro_bidirectional(bs, fps, min_cutoff=2.0, beta=0.5), 0.0, 1.0)
        shapes = {n: bs[:, i] for i, n in enumerate(ARKIT)}
    lip_path = ctx.workdir / "lipsync" / "visemes.json"
    if ctx.cfg["lipsync"]["enabled"] and lip_path.exists():
        lip = ctx.read_json("lipsync", "visemes.json").get("shapes") or {}
        if lip:
            shapes = merge_lipsync(shapes, lip, T, ctx.cfg["lipsync"]["mode"], ctx.cfg["lipsync"]["strength"])
            log.info("  lip sync merged into the mouth (%s)", ctx.cfg["lipsync"]["mode"])
    shapes = {n: np.round(np.asarray(v, dtype=float), 3).tolist() for n, v in shapes.items()}

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
    log.info("  feet in frame on %.0f%% of frames: %s", 100 * feet,
             "full-body shot" if lengths["full_body"] else
             "closer shot, so the avatar is sized by your torso and no floor is used")
    log.info("  head: %s", head_info)
    return [ctx.write_json(S, "motion.json", motion)]


def feet_in_frame(image: np.ndarray, min_vis: float = 0.5) -> float:
    """Fraction of frames where both ankles are confidently seen inside the picture.
    Without the feet, leg length and the floor are only guesses."""
    ids = [IDX["l_ankle"], IDX["r_ankle"]]
    p = image[:, ids, :]
    inside = (p[..., 0] >= 0) & (p[..., 0] <= 1) & (p[..., 1] >= 0) & (p[..., 1] <= 1)
    ok = np.isfinite(p).all(-1) & inside & (p[..., 3] >= min_vis)
    rows = np.isfinite(p[:, :, :2]).all(axis=(1, 2))
    return float(ok.all(1)[rows].mean()) if rows.any() else 0.0


def apply_head(pts: dict, face_mat: np.ndarray, face_crop, w: int, h: int, f: float, fps: float) -> str:
    """Set the head targets: 'head_top' (where the top of the head points) and 'head_side'
    (the head's left, for its turn). Face-tracked rotation where available, faded into
    the body-landmark estimate where the face was lost."""
    body_top = pts["head_top"]
    body_side = pts["l_ear"]
    pts["head_side"] = body_side
    body_up = pts["ear_mid"] - pts["shoulder_mid"]
    body_left = pts["l_ear"] - pts["r_ear"]
    res = headpose.head_directions(face_mat, face_crop, w, h, f, fps, body_left, body_up)
    if res is None:
        return "from body landmarks (face not tracked)"
    up, left, conf = res
    face_top = pts["neck"] + 0.3 * up
    face_side = face_top + 0.2 * left
    c = conf[:, None]
    pts["head_top"] = c * face_top + (1 - c) * body_top
    pts["head_side"] = c * face_side + (1 - c) * body_side
    return f"from the face tracker on {int((conf > 0.5).sum())}/{len(conf)} frames"


def merge_lipsync(shapes: dict, lip: dict, T: int, mode: str, strength: float) -> dict:
    """Combine camera-tracked mouth values with audio-driven ones.
    max: per frame, whichever opens the mouth more · audio: mouth from audio only."""
    out = dict(shapes)

    def fit(a):
        a = np.asarray(a, dtype=float)[:T]
        return np.pad(a, (0, T - len(a)), mode="edge") if 0 < len(a) < T else (a if len(a) else np.zeros(T))

    for name, vals in lip.items():
        audio = np.clip(fit(vals) * float(strength), 0.0, 1.0)
        cam_vals = np.asarray(out.get(name, np.zeros(T)), dtype=float)
        out[name] = audio if mode == "audio" else np.maximum(cam_vals, audio)
    return out
