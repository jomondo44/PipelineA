"""Stage 2: MediaPipe Pose (+ Face blendshapes) on every frame, CPU only.

Writes raw, unsmoothed results to track/tracking.npz:
  pose_world (T,33,4)  metric, hip-centered, camera-aligned: x,y,z,visibility
  pose_image (T,33,4)  normalized image coords: x,y,z,visibility
  face_bs    (T,52)    ARKit-style blendshape scores (NaN when no face)
  face_mat   (T,4,4)   face transform (NaN when no face)
  face_crop  (T,3)     close-up crop used for the face: center x, y (0-1 of the frame) and
                       side (fraction of the frame's long side); NaN = whole frame was used

With track.face_crop on, the face is tracked on a head crop cut from the ORIGINAL
video at full resolution, which gives the face tracker several times more pixels.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .. import facecrop, media, models
from ..landmarks import ARKIT
from ..pipeline import Context

log = logging.getLogger("avatarpipe")
S = "track"


def inputs(ctx: Context) -> list:
    return [ctx.workdir / "ingest" / "performance.mp4", ctx.cfg["input"]["performance"]]


def _vision():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("mediapipe is not installed: pip install mediapipe") from e
    return mp, mpt, vision


def full_res_frames(ctx: Context, meta: dict):
    """The original performance at full resolution, cut and timed exactly like ingest did.
    Returns (frame iterator, width, height) or None if it can't be read."""
    src = ctx.cfg["input"]["performance"]
    try:
        info = media.probe(src)
    except media.MediaError as e:
        log.warning("  face close-up off: can't read %s (%s)", src, e)
        return None
    W, H = media.even(info["width"]), media.even(info["height"])
    trim = meta.get("trim")
    if trim is None:  # ingest from an older version: rebuild the same trim
        i = ctx.cfg["input"]
        trim = (["-ss", str(i["start"])] if i["start"] else []) + (["-to", str(i["end"])] if i["end"] is not None else [])
    fps = meta.get("fps_filter") or meta["fps_str"]
    return media.read_frames(src, W, H, pre=trim, vf=f"fps={fps}"), W, H


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
    full = None
    if tcfg["face"]:
        face_opts = vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=models.get("face"), delegate=cpu),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=conf, min_face_presence_confidence=conf,
            min_tracking_confidence=conf,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        face_lm = vision.FaceLandmarker.create_from_options(face_opts)
        if tcfg["face_crop"]:
            full = full_res_frames(ctx, meta)
            if full is not None:
                log.info("  face close-up from the original %dx%d video", full[1], full[2])
    full_iter, FW, FH = full if full is not None else (None, 0, 0)
    out_px = int(tcfg["face_crop_size"])
    smoother = facecrop.BoxSmoother()

    world, image, bs, mats, crops = [], [], [], [], []
    nan_pose = np.full((33, 4), np.nan, np.float32)
    total = max(meta["frames"], 1)
    t0, last = time.time(), 0.0
    hits = faces = cropped = 0
    with vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm:
        for i, frame in enumerate(media.read_frames(ctx.workdir / "ingest" / "performance.mp4", w, h)):
            ts = int(round(i * 1000.0 / fps))
            big = next(full_iter, None) if full_iter is not None else None
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            r = pose_lm.detect_for_video(img, ts)
            lm = None
            if r.pose_world_landmarks:
                world.append([[p.x, p.y, p.z, p.visibility or 0.0] for p in r.pose_world_landmarks[0]])
                lm = np.array([[p.x, p.y, p.z, p.visibility or 0.0] for p in r.pose_landmarks[0]], np.float32)
                image.append(lm)
                hits += 1
            else:
                world.append(nan_pose)
                image.append(nan_pose)

            row = np.full(len(ARKIT), np.nan, np.float32)
            mat = np.full((4, 4), np.nan, np.float32)
            crop_rec = np.full(3, np.nan, np.float32)
            if face_lm is not None:
                face_img = img
                if big is not None:
                    box = smoother.update(facecrop.head_box(lm, FW, FH) if lm is not None else None)
                    if box is not None:
                        patch = facecrop.extract(big, box, out_px)
                        face_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=patch)
                        crop_rec = np.array([box[0] / FW, box[1] / FH, box[2] / max(FW, FH)], np.float32)
                        cropped += 1
                fr = face_lm.detect_for_video(face_img, ts)
                if fr.face_blendshapes:
                    scores = {c.category_name: c.score for c in fr.face_blendshapes[0]}
                    row = np.array([scores.get(n, 0.0) for n in ARKIT], np.float32)
                    faces += 1
                if fr.facial_transformation_matrixes:
                    mat = np.asarray(fr.facial_transformation_matrixes[0], np.float32).reshape(4, 4)
            bs.append(row)
            mats.append(mat)
            crops.append(crop_rec)

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
    log.info("  body found in %d/%d frames (%.0f%%)", hits, n, 100 * hits / n)
    if face_lm is not None:
        log.info("  face found in %d/%d frames (%.0f%%)%s", faces, n, 100 * faces / n,
                 f", close-up crop on {cropped}" if full is not None else "")
        if faces / n < 0.5:
            log.warning("  face often lost: face the camera more, light your face, or shoot at a higher resolution")
    if hits / n < 0.6:
        log.warning("  low body detection rate; gaps will be interpolated, expect drift")

    out = ctx.path(S, "tracking.npz")
    np.savez_compressed(out, pose_world=np.asarray(world, np.float32), pose_image=np.asarray(image, np.float32),
                        face_bs=np.asarray(bs, np.float32), face_mat=np.asarray(mats, np.float32),
                        face_crop=np.asarray(crops, np.float32),
                        fps=np.float32(fps), size=np.array([w, h], np.int32))
    return [out]
