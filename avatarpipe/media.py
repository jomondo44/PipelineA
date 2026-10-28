"""Thin ffmpeg/ffprobe wrappers: probing, frame pipes in and out."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

log = logging.getLogger("avatarpipe")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".exr", ".webp"}


class MediaError(RuntimeError):
    pass


def require(tool: str) -> str:
    exe = shutil.which(tool)
    if not exe:
        raise MediaError(f"'{tool}' not found on PATH. Install FFmpeg (https://ffmpeg.org/download.html).")
    return exe


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXT


def run(args: list[str], what: str = "ffmpeg") -> None:
    cmd = [require(args[0])] + [str(a) for a in args[1:]]
    log.debug("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MediaError(f"{what} failed (exit {proc.returncode}):\n{tail}")


def probe(path: str | Path) -> dict:
    """Return width, height, fps (float), fps_str, frames, duration, has_audio, rotation."""
    if not Path(path).exists():
        raise MediaError(f"file not found: {path}")
    out = subprocess.run(
        [require("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if out.returncode != 0:
        raise MediaError(f"ffprobe could not read {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if v is None:
        raise MediaError(f"no video stream in {path}")
    a = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    fps_str = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(fps_str)) if fps_str != "0/0" else 0.0
    except (ZeroDivisionError, ValueError):
        fps = 0.0
    dur = float(v.get("duration") or data.get("format", {}).get("duration") or 0.0)
    frames = int(v.get("nb_frames") or 0) or (int(round(dur * fps)) if fps else 0)
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(v.get("tags", {}).get("rotate", rot))
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) in (90, 270):
        w, h = h, w
    return {"width": w, "height": h, "fps": fps, "fps_str": fps_str, "frames": frames,
            "duration": dur, "has_audio": a, "rotation": rot}


def read_frames(path: str | Path, width: int, height: int, gray: bool = False,
                pre: list | None = None, vf: str | None = None):
    """Yield frames (H,W,3 RGB uint8, or H,W gray) decoded by ffmpeg at the given size.

    pre: extra input options (e.g. a trim: -ss/-to); vf: filters to run before scaling."""
    pix = "gray" if gray else "rgb24"
    ch = 1 if gray else 3
    chain = (vf + "," if vf else "") + f"scale={width}:{height}:flags=area"
    cmd = [require("ffmpeg"), "-v", "error", *[str(a) for a in (pre or [])], "-i", str(path),
           "-vf", chain, "-f", "rawvideo", "-pix_fmt", pix, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
    n = width * height * ch
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            arr = np.frombuffer(buf, np.uint8)
            yield arr.reshape(height, width) if gray else arr.reshape(height, width, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
        if proc.returncode not in (0, None, -13) and err.strip():
            log.debug("ffmpeg reader stderr: %s", err.strip())


class FrameWriter:
    """Pipe raw frames into ffmpeg. gray=True writes a lossless-ish grayscale matte."""

    def __init__(self, path, width, height, fps_str, gray=False, crf=12, preset="veryfast"):
        self.path = Path(path)
        pix_in = "gray" if gray else "rgb24"
        cmd = [require("ffmpeg"), "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{width}x{height}", "-r", fps_str, "-i", "-"]
        if gray:
            # mattes are stored lossless (FFV1 in .mkv)
            cmd += ["-c:v", "ffv1", "-pix_fmt", "gray", str(self.path)]
        else:
            cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.shape = (height, width) if gray else (height, width, 3)
        self.count = 0

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self.shape:
            raise MediaError(f"frame shape {frame.shape} != {self.shape}")
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.count += 1

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise MediaError(f"ffmpeg writer failed for {self.path}:\n{err[-1500:]}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def even(n: float) -> int:
    n = int(round(n))
    return n - (n % 2)


def target_size(w: int, h: int, max_height: int | None) -> tuple[int, int]:
    if max_height and h > max_height:
        return even(w * max_height / h), even(max_height)
    return even(w), even(h)
