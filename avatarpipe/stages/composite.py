"""Stage 6: layer it all with one ffmpeg filter graph.

  replace    you painted out with the clean plate (through the matte), avatar on top
  new_scene  avatar over a different background video or image
  overlay    avatar over the original footage (useful for checking alignment)
"""
from __future__ import annotations

import logging
from pathlib import Path

from .. import media
from ..pipeline import Context

log = logging.getLogger("avatarpipe")
S = "composite"


def inputs(ctx: Context) -> list:
    # upstream outputs are tracked through stage keys (deps), not by hashing frames
    return []


def build_graph(mode: str, *, has_matte: bool, bg_is_image: bool, grade: dict, grain: int,
                offset: tuple[int, int], frames: int, fps: str = "30") -> tuple[str, list[str]]:
    """Return (filter_complex, extra input specs). Input 0 = performance, 1 = avatar frames,
    then 2 = plate/background, 3 = matte (replace mode)."""
    g = grade
    eq = (f"eq=brightness={g['brightness']}:contrast={g['contrast']}:"
          f"saturation={g['saturation']}:gamma={g['gamma']}")

    def src(i: int) -> str:
        # Re-stamp every input as frame N at exactly the project rate. Containers round
        # timestamps differently (mp4 vs mkv vs image loops); without this, frame-sync
        # filters see "extra" frames and the output comes out short.
        return f"[{i}:v]trim=end_frame={frames},settb=AVTB,setpts=N/({fps})/TB"

    parts = [f"{src(1)},format=yuva444p,{eq},format=rgba[av]"]
    if mode == "overlay":
        parts.append(f"{src(0)},format=rgb24[base]")
    elif mode == "new_scene":
        parts.append(f"{src(2)},format=rgb24[base]")
    elif mode == "replace" and has_matte:
        parts += [
            f"{src(0)},format=gbrp[live]",
            f"{src(2)},format=gbrp[plate]",
            f"{src(3)},format=gbrp[mask]",
            "[live][plate][mask]maskedmerge,format=rgb24[base]",
        ]
    else:  # replace without matte: the clean plate is the whole background
        parts.append(f"{src(2)},format=rgb24[base]")
    x, y = offset
    tail = f"[base][av]overlay=x={x}:y={y}:format=auto:eof_action=pass,trim=end_frame={frames}"
    if grain:
        tail += f",noise=alls={int(grain)}:allf=t"
    parts.append(tail + ",format=yuv420p[out]")
    return ";".join(parts), []


def run(ctx: Context) -> list:
    cfg = ctx.cfg
    c = cfg["composite"]
    meta = ctx.read_json("ingest", "meta.json")
    mode = c["mode"]
    frames_dir = ctx.workdir / "render" / "frames"
    if not any(frames_dir.glob("avatar_*.png")):
        raise RuntimeError("no rendered avatar frames; run the render stage first (needs avatar.path)")
    fps = meta["fps_str"]
    perf = ctx.workdir / "ingest" / "performance.mp4"
    matte = ctx.workdir / "matte" / "matte.mkv"
    has_matte = mode == "replace" and cfg["matte"]["enabled"] and matte.exists()

    args = ["ffmpeg", "-v", "error", "-y", "-i", str(perf),
            "-framerate", fps, "-start_number", "1", "-i", str(frames_dir / "avatar_%04d.png")]
    bg_is_image = False
    if mode == "replace":
        args += ["-loop", "1", "-framerate", fps, "-i", meta["clean_plate"]]
        bg_is_image = True
        if has_matte:
            args += ["-i", str(matte)]
    elif mode == "new_scene":
        bg = meta["background"]
        bg_is_image = media.is_image(bg)
        args += (["-loop", "1", "-framerate", fps, "-i", bg] if bg_is_image
                 else ["-stream_loop", "-1", "-i", bg])

    graph, _ = build_graph(mode, has_matte=has_matte, bg_is_image=bg_is_image, grade=c["grade"],
                           grain=c["grain"], offset=tuple(c["avatar_offset_px"]), frames=meta["frames"],
                           fps=fps)
    out = Path(c["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    args += ["-filter_complex", graph, "-map", "[out]"]
    if c["audio"] and meta["has_audio"]:
        args += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    args += ["-frames:v", str(meta["frames"]), "-r", fps,
             "-c:v", "libx264", "-preset", c["preset"], "-crf", str(c["crf"]),
             "-movflags", "+faststart", str(out)]
    log.info("  mode %s%s -> %s", mode, " (matte paint-out)" if has_matte else "", out)
    media.run(args, "composite")
    return [out]
