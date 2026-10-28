"""Diagnostic video: raw 2D landmarks (green) vs the solved 3D skeleton reprojected (orange).

If orange sits on green, the camera/focal settings and depth solve are right and the
avatar will land where you were standing. Orange drifting off = adjust solve.focal_mm.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .. import media
from ..landmarks import IDX
from ..pipeline import Context
from .solve import focal_px

log = logging.getLogger("avatarpipe")

BONES = [("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
         ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
         ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
         ("l_ankle", "l_foot_index"), ("r_ankle", "r_foot_index")]


def project(p_blender: np.ndarray, f: float, w: int, h: int) -> np.ndarray:
    X, Y, Z = p_blender[..., 0], -p_blender[..., 1], -p_blender[..., 2]
    Z = np.where(Z > 1e-3, Z, np.nan)
    return np.stack([f * X / Z + w / 2, f * Y / Z + h / 2], -1)


def run(ctx: Context) -> str:
    meta = ctx.read_json("ingest", "meta.json")
    w, h = meta["width"], meta["height"]
    trk = np.load(ctx.workdir / "track" / "tracking.npz")
    image = trk["pose_image"]
    mpath = ctx.workdir / "solve" / "motion.json"
    solved = None
    if mpath.exists():
        motion = ctx.read_json("solve", "motion.json")
        f = focal_px(motion["camera"]["focal_mm"], w, h)
        solved = {n: project(np.asarray(motion["points"][n]), f, w, h) for n in IDX}
    out = ctx.path("preview", "tracking_preview.mp4")

    def draw(img, pts, color, thick):
        for a, b in BONES:
            pa, pb = pts(a), pts(b)
            if pa is not None and pb is not None:
                cv2.line(img, pa, pb, color, thick, cv2.LINE_AA)

    with media.FrameWriter(out, w, h, meta["fps_str"], crf=23) as wr:
        for t, frame in enumerate(media.read_frames(ctx.workdir / "ingest" / "performance.mp4", w, h)):
            img = np.ascontiguousarray(frame.copy())
            if t < len(image):
                def raw(n, t=t):
                    p = image[t, IDX[n]]
                    return None if not np.isfinite(p[:2]).all() else (int(p[0] * w), int(p[1] * h))
                draw(img, raw, (60, 220, 90), 3)
            if solved is not None:
                def sol(n, t=t):
                    if t >= len(solved[n]) or not np.isfinite(solved[n][t]).all():
                        return None
                    return tuple(int(v) for v in solved[n][t])
                draw(img, sol, (255, 140, 0), 1)
            cv2.putText(img, f"{t + 1}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            wr.write(img)
    log.info("preview: %s", out)
    return str(out)
