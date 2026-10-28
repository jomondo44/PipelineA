"""Stage 4: per-frame person matte (CPU segmentation), grown + feathered for paint-out.

Output: matte/matte.mkv, lossless grayscale; white = you.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from .. import media, models
from ..pipeline import Context

log = logging.getLogger("avatarpipe")
S = "matte"


def inputs(ctx: Context) -> list:
    return [ctx.workdir / "ingest" / "performance.mp4"]


def enabled(cfg: dict) -> bool:
    return bool(cfg["matte"]["enabled"]) and cfg["composite"]["mode"] == "replace"


def person_prob(masks: list[np.ndarray]) -> np.ndarray:
    """Multiclass model: index 0 is background -> person = 1 - bg. Single-mask model: the mask
    is the person probability; guard against an inverted mask using the frame border."""
    if len(masks) > 1:
        return 1.0 - masks[0]
    m = masks[0]
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean()
    return 1.0 - m if border > 0.5 else m


def refine(prob: np.ndarray, threshold: float, dilate_px: int, feather_px: int) -> np.ndarray:
    hard = (prob >= threshold).astype(np.uint8) * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        hard = cv2.dilate(hard, k)
    if feather_px > 0:
        hard = cv2.GaussianBlur(hard, (0, 0), sigmaX=feather_px)
    return hard


def run(ctx: Context) -> list:
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    mcfg = ctx.cfg["matte"]
    meta = ctx.read_json("ingest", "meta.json")
    w, h, fps = meta["width"], meta["height"], meta["fps"]
    opts = vision.ImageSegmenterOptions(
        base_options=mpt.BaseOptions(model_asset_path=models.get(mcfg["model"]),
                                     delegate=mpt.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True, output_category_mask=False)
    out = ctx.path(S, "matte.mkv")
    t0, last, n = time.time(), 0.0, 0
    with vision.ImageSegmenter.create_from_options(opts) as seg, \
            media.FrameWriter(out, w, h, meta["fps_str"], gray=True) as wr:
        for i, frame in enumerate(media.read_frames(ctx.workdir / "ingest" / "performance.mp4", w, h)):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            r = seg.segment_for_video(img, int(round(i * 1000.0 / fps)))
            masks = [m.numpy_view().astype(np.float32).reshape(h, w) for m in r.confidence_masks]
            wr.write(refine(person_prob(masks), mcfg["threshold"], mcfg["dilate_px"], mcfg["feather_px"]))
            n += 1
            if time.time() - last > 5:
                last = time.time()
                log.info("  matte %d/%d (%.1f fps)", n, meta["frames"], n / (last - t0))
    return [out]
