"""Lip sync from the performance audio with Rhubarb Lip Sync (MIT, CPU, offline).

Rhubarb turns speech into mouth shapes (A-H, X). Each shape becomes a set of
ARKit-style mouth values, which the solve stage merges with what the camera saw,
so the mouth reads well even when the face is tiny in a full-body shot.

Writes lipsync/visemes.json: {"fps", "frames", "shapes": {arkit name: [per frame]}, "note"}.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .. import media
from ..filters import one_euro_bidirectional
from ..pipeline import Context

log = logging.getLogger("avatarpipe")
S = "lipsync"
INSTALL_HINT = ("download Rhubarb Lip Sync from https://github.com/DanielSWolf/rhubarb-lip-sync/releases, "
                "unzip the whole folder (keep its 'res' folder) into a 'tools' folder in your project, "
                "or set lipsync.executable to the full path of rhubarb.exe")

# Rhubarb mouth shape -> ARKit-style mouth values
SHAPES = {
    "X": {},                                                     # rest
    "A": {"mouthClose": 0.3},                                    # M B P: lips pressed
    "B": {"jawOpen": 0.12, "mouthStretchLeft": 0.45, "mouthStretchRight": 0.45},   # K S T, "ee"
    "C": {"jawOpen": 0.45, "mouthLowerDownLeft": 0.35, "mouthLowerDownRight": 0.35},  # "eh", "a" in bat
    "D": {"jawOpen": 0.9},                                       # "aa" in father
    "E": {"jawOpen": 0.3, "mouthFunnel": 0.6},                   # "o" in off, "er"
    "F": {"mouthPucker": 0.9, "mouthFunnel": 0.3},               # "oo", "ow", "w"
    "G": {"jawOpen": 0.08, "mouthStretchLeft": 0.25, "mouthStretchRight": 0.25, "mouthRollLower": 0.5},  # F V
    "H": {"jawOpen": 0.5, "mouthLowerDownLeft": 0.2, "mouthLowerDownRight": 0.2},  # long L
}
MOUTH = sorted({k for v in SHAPES.values() for k in v})


def enabled(cfg: dict) -> bool:
    return bool(cfg["lipsync"]["enabled"])


def find_rhubarb(cfg: dict) -> str | None:
    exe = os.environ.get("AVATARPIPE_RHUBARB") or cfg["lipsync"]["executable"] or "rhubarb"
    base = Path(cfg.get("_base_dir", "."))
    if any(sep in exe for sep in ("/", "\\")):
        p = Path(exe).expanduser()
        p = p if p.is_absolute() else (base / p)
        return str(p.resolve()) if p.exists() else None
    found = shutil.which(exe)
    if found:
        return found
    # an unzipped release in a tools/ folder next to the shot, the project, or where you ran from
    names = ("rhubarb.exe", "rhubarb") if os.name == "nt" else ("rhubarb",)
    for root in (base, base.parent, Path.cwd()):
        for name in names:
            hits = sorted(glob.glob(str(root / "tools" / "*" / name)) + glob.glob(str(root / "tools" / name)))
            if hits:
                return str(Path(hits[-1]).resolve())
    return None


def inputs(ctx: Context) -> list:
    return [ctx.workdir / "ingest" / "performance.mp4", find_rhubarb(ctx.cfg)]


def cues_to_frames(cues: list[dict], fps: float, frames: int, smooth: bool = True) -> dict[str, list]:
    """Rhubarb's timed mouth cues -> per-frame mouth values (smoothed so shapes blend)."""
    vals = {k: np.zeros(frames) for k in MOUTH}
    t = np.arange(frames) / fps
    for c in cues:
        shape = SHAPES.get(str(c.get("value", "X")).upper(), {})
        sel = (t >= float(c["start"])) & (t < float(c["end"]))
        for k, v in shape.items():
            vals[k][sel] = v
    out = {}
    for k, v in vals.items():
        if smooth and frames > 2:
            v = one_euro_bidirectional(v, fps, min_cutoff=4.0, beta=0.0)
        out[k] = np.round(np.clip(v, 0.0, 1.0), 3).tolist()
    return out


def _write(ctx: Context, meta: dict, shapes: dict, note: str) -> Path:
    return ctx.write_json(S, "visemes.json", {"fps": meta["fps"], "frames": meta["frames"],
                                              "shapes": shapes, "note": note})


def run(ctx: Context) -> list:
    meta = ctx.read_json("ingest", "meta.json")
    lcfg = ctx.cfg["lipsync"]
    if not meta.get("has_audio"):
        log.warning("  no audio in the performance video: lip sync skipped")
        return [_write(ctx, meta, {}, "no audio")]
    exe = find_rhubarb(ctx.cfg)
    if not exe:
        log.warning("  Rhubarb Lip Sync not found, so lip sync is skipped: %s", INSTALL_HINT)
        return [_write(ctx, meta, {}, "rhubarb not found")]

    wav = ctx.path(S, "audio.wav")
    media.run(["ffmpeg", "-v", "error", "-y", "-i", ctx.workdir / "ingest" / "performance.mp4",
               "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], "extract audio")
    cues_path = ctx.path(S, "cues.json")
    cmd = [exe, "-q", "-r", lcfg["recognizer"], "-f", "json", "-o", str(cues_path), str(wav)]
    log.info("  rhubarb (%s) on %.0fs of audio", lcfg["recognizer"], meta["duration"])
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                          cwd=str(Path(exe).parent))  # Rhubarb looks for its res/ folder next to itself
    if proc.returncode != 0 or not cues_path.exists():
        # a lip sync problem shouldn't stop the whole shot: carry on with the camera's mouth
        log.warning("  Rhubarb failed (exit %s), so lip sync is skipped. After fixing it, use "
                    "'Rerun stage: lipsync'. Rhubarb said: %s", proc.returncode, proc.stderr.strip()[-600:])
        return [_write(ctx, meta, {}, f"rhubarb failed (exit {proc.returncode})")]
    cues = json.loads(cues_path.read_text(encoding="utf-8")).get("mouthCues", [])
    talking = sum(1 for c in cues if c.get("value") not in ("X", "A"))
    log.info("  %d mouth cues (%d speaking)", len(cues), talking)
    shapes = cues_to_frames(cues, meta["fps"], meta["frames"])
    return [_write(ctx, meta, shapes, f"{len(cues)} cues"), cues_path]
