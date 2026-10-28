"""Stage 2: MediaPipe Pose (+ Face blendshapes) on every frame, CPU only.

Writes raw, unsmoothed results to track/tracking.npz:
  pose_world (T,33,4)  metric, hip-centered, camera-aligned: x,y,z,visibility
  pose_image (T,33,4)  normalized image coords: x,y,z,visibility
  face_bs    (T,52)    ARKit-style blendshape scores (NaN when no face)
  face_mat   (T,4,4)   face transform (NaN when no face)
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .. import media, models
from ..landmarks import ARKIT
from ..pipeline import Context

log = logging.getLogger("avatarpipe")
S = "track"


def inputs(ctx: Context) -> list:
    return [ctx.workdir / "ingest" / "performance.mp4"]


def _vision():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("mediapipe is not installed: pip install mediapipe") from e
    return mp, mpt, vision


def run(ctx: Context) -> list:
    mp, mpt, vision = _vision()
    tcfg = ctx.cfg["track"]
    meta = ctx.read_json("ingest", "meta.json")
    w, h, fps = meta["width"], meta["height"], meta["fps"]
    conf = float(tcfg["min_confidence"])
    cpu = mpt.BaseOptions.Delegate.CPU

    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=models.get(f"pose_{tcfg['pose_model']}"), delegate=cpu),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=conf, min_pose_presence_confidence=conf,
        min_tracking_confidence=conf, output_segmentation_masks=False)
    face_lm = None
    if tcfg["face"]:
        face_opts = vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=models.get("face"), delegate=cpu),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=conf, min_face_presence_confidence=conf,
            min_tracking_confidence=conf,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        face_lm = vision.FaceLandmarker.create_from_options(face_opts)

    world, image, bs, mats = [], [], [], []
    nan_pose = np.full((33, 4), np.nan, np.float32)
    total = max(meta["frames"], 1)
    t0, last = time.time(), 0.0
    hits = faces = 0
    with vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm:
        for i, frame in enumerate(media.read_frames(ctx.workdir / "ingest" / "performance.mp4", w, h)):
            ts = int(round(i * 1000.0 / fps))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            r = pose_lm.detect_for_video(img, ts)
            if r.pose_world_landmarks:
                world.append([[p.x, p.y, p.z, p.visibility or 0.0] for p in r.pose_world_landmarks[0]])
                image.append([[p.x, p.y, p.z, p.visibility or 0.0] for p in r.pose_landmarks[0]])
                hits += 1
            else:
                world.append(nan_pose)
                image.append(nan_pose)

            row = np.full(len(ARKIT), np.nan, np.float32)
            mat = np.full((4, 4), np.nan, np.float32)
            if face_lm is not None:
                fr = face_lm.detect_for_video(img, ts)
                if fr.face_blendshapes:
                    scores = {c.category_name: c.score for c in fr.face_blendshapes[0]}
                    row = np.array([scores.get(n, 0.0) for n in ARKIT], np.float32)
                    faces += 1
                if fr.facial_transformation_matrixes:
                    mat = np.asarray(fr.facial_transformation_matrixes[0], np.float32).reshape(4, 4)
            bs.append(row)
            mats.append(mat)

            if time.time() - last > 5:
                last = time.time()
                rate = (i + 1) / (last - t0)
                log.info("  frame %d/%d  (%.1f fps, ~%.0fs left)", i + 1, total, rate, (total - i - 1) / max(rate, 1e-6))
    if face_lm is not None:
        face_lm.close()

    n = len(world)
    if n == 0:
        raise RuntimeError("no frames decoded from performance video")
    if hits == 0:
        raise RuntimeError("no body detected in any frame. Is your full body (or at least torso) in frame and well lit?")
    log.info("  body found in %d/%d frames (%.0f%%)%s", hits, n, 100 * hits / n,
             f", face in {faces}/{n}" if face_lm is not None else "")
    if hits / n < 0.6:
        log.warning("  low body detection rate; gaps will be interpolated, expect drift")

    out = ctx.path(S, "tracking.npz")
    np.savez_compressed(out, pose_world=np.asarray(world, np.float32), pose_image=np.asarray(image, np.float32),
                        face_bs=np.asarray(bs, np.float32), face_mat=np.asarray(mats, np.float32),
                        fps=np.float32(fps), size=np.array([w, h], np.int32))
    return [out]
