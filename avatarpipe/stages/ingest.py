"""Stage 1: normalize all footage to one size/fps; build the clean plate; prep the background."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .. import media
from ..pipeline import Context

log = logging.getLogger("avatarpipe")
S = "ingest"


def inputs(ctx: Context) -> list:
    i = ctx.cfg["input"]
    return [i["performance"], i["clean_plate"], i["background"]]


def fit_filter(w: int, h: int, how: str) -> str:
    if how == "fit":
        return (f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    return (f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={w}:{h},setsar=1")


def run(ctx: Context) -> list:
    cfg = ctx.cfg
    inp = cfg["input"]
    src = media.probe(inp["performance"])
    w, h = media.target_size(src["width"], src["height"], inp["max_height"])
    fps_str = str(inp["fps"]) if inp["fps"] else src["fps_str"]
    log.info("  performance %dx%d @ %.3f fps -> %dx%d @ %s", src["width"], src["height"], src["fps"], w, h, fps_str)

    perf = ctx.path(S, "performance.mp4")
    trim = ["-ss", str(inp["start"])] if inp["start"] else []
    if inp["end"] is not None:
        trim += ["-to", str(inp["end"])]
    args = ["ffmpeg", "-v", "error", "-y", *trim, "-i", inp["performance"],
            "-vf", f"scale={w}:{h}:flags=lanczos,setsar=1,fps={fps_str}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p"]
    args += ["-c:a", "aac", "-b:a", "192k"] if src["has_audio"] else ["-an"]
    media.run(args + [str(perf)], "ingest performance")

    out = media.probe(perf)
    meta = {"width": w, "height": h, "fps": out["fps"], "fps_str": out["fps_str"],
            "frames": out["frames"], "duration": out["duration"], "has_audio": out["has_audio"],
            "source": inp["performance"], "source_size": [src["width"], src["height"]],
            # exactly how the performance was cut, so other stages can re-read the source in step
            "trim": trim, "fps_filter": fps_str}
    outputs = [perf]

    if inp["clean_plate"]:
        plate = ctx.path(S, "clean_plate.png")
        build_clean_plate(inp["clean_plate"], w, h, plate)
        meta["clean_plate"] = str(plate)
        outputs.append(plate)

    if inp["background"]:
        how = cfg["composite"]["background_scale"]
        if media.is_image(inp["background"]):
            bg = ctx.path(S, "background.png")
            media.run(["ffmpeg", "-v", "error", "-y", "-i", inp["background"],
                       "-vf", fit_filter(w, h, how), "-frames:v", "1", str(bg)], "ingest background")
        else:
            bg = ctx.path(S, "background.mp4")
            media.run(["ffmpeg", "-v", "error", "-y", "-i", inp["background"],
                       "-vf", f"{fit_filter(w, h, how)},fps={fps_str}", "-an",
                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
                       str(bg)], "ingest background")
        meta["background"] = str(bg)
        outputs.append(bg)

    outputs.append(ctx.write_json(S, "meta.json", meta))
    log.info("  %d frames, %.1fs", meta["frames"], meta["duration"])
    return outputs


def build_clean_plate(path: str, w: int, h: int, dest: Path, samples: int = 31) -> None:
    """Image: just resize. Video: per-pixel temporal median of evenly spaced frames.

    The median removes anything that moves through the empty-scene clip (people,
    cars, leaves) and averages out sensor noise.
    """
    if media.is_image(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise media.MediaError(f"could not read clean plate image {path}")
        cv2.imwrite(str(dest), cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
        return
    info = media.probe(path)
    n = max(info["frames"], 1)
    step = max(n // samples, 1)
    frames = [f for i, f in enumerate(media.read_frames(path, w, h)) if i % step == 0][:samples]
    if not frames:
        raise media.MediaError(f"no frames decoded from clean plate {path}")
    med = np.median(np.stack(frames), axis=0).astype(np.uint8)
    cv2.imwrite(str(dest), cv2.cvtColor(med, cv2.COLOR_RGB2BGR))
    log.info("  clean plate: median of %d frames", len(frames))
