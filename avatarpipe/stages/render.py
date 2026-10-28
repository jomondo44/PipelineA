"""Stage 5: drive the avatar in headless Blender and render an RGBA PNG sequence."""
from __future__ import annotations

import glob
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .. import presets
from ..pipeline import Context

log = logging.getLogger("avatarpipe")
S = "render"
SCRIPT = Path(__file__).resolve().parent.parent / "blender" / "drive_avatar.py"


def inputs(ctx: Context) -> list:
    a = ctx.cfg["avatar"]
    return [ctx.workdir / "solve" / "motion.json", a["path"], a["bone_map"], a["face_map"], SCRIPT]


def enabled(cfg: dict) -> bool:
    return bool(cfg["avatar"]["path"])


def find_blender(configured: str) -> str:
    cand = [os.environ.get("AVATARPIPE_BLENDER"), configured]
    for c in cand:
        if c and (shutil.which(c) or Path(c).exists()):
            return shutil.which(c) or c
    sysname = platform.system()
    globs = []
    if sysname == "Darwin":
        globs = ["/Applications/Blender*.app/Contents/MacOS/Blender"]
    elif sysname == "Windows":
        globs = [r"C:\Program Files\Blender Foundation\Blender*\blender.exe"]
    else:
        globs = ["/opt/blender*/blender", str(Path.home() / "blender*" / "blender"), "/snap/bin/blender"]
    for g in globs:
        hits = sorted(glob.glob(g))
        if hits:
            return hits[-1]
    raise RuntimeError("Blender not found. Install it from https://www.blender.org/download/ and set "
                       "blender.executable in your config (or the AVATARPIPE_BLENDER env var).")


def build_job(ctx: Context, frames_dir: Path) -> dict:
    a, b = ctx.cfg["avatar"], ctx.cfg["blender"]

    def load_map(p):
        if not p:
            return {}
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{p}: expected a mapping")
        return data

    if not Path(a["path"]).exists():
        raise FileNotFoundError(f"avatar not found: {a['path']}")
    return {
        "motion_path": str(ctx.workdir / "solve" / "motion.json"),
        "frames_dir": str(frames_dir),
        "blend_out": str(ctx.path(S, "scene.blend")) if b["save_blend"] else None,
        "avatar": {k: a[k] for k in ("path", "rig_preset", "face_preset", "twist_axis", "scale", "armature",
                                     "face_gain")},
        "blender": b,
        "drive": presets.DRIVE,
        "rigs": presets.RIGS,
        "faces": presets.FACES,
        "custom_bone_map": load_map(a["bone_map"]),
        "custom_face_map": load_map(a["face_map"]),
    }


def command(exe: str, job_path: Path, extra: list) -> list:
    name = Path(exe).name.lower()
    if name.startswith("python"):
        # bpy-as-a-module install (pip install bpy): run the script with that interpreter
        return [exe, str(SCRIPT), "--", str(job_path)]
    return [exe, "-b", "--python-exit-code", "3", *extra, "--python", str(SCRIPT), "--", str(job_path)]


def run(ctx: Context) -> list:
    meta = ctx.read_json("solve", "motion.json")
    frames_dir = ctx.dir(S) / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    job = build_job(ctx, frames_dir)
    job_path = ctx.write_json(S, "job.json", job)

    exe = find_blender(ctx.cfg["blender"]["executable"])
    cmd = command(exe, job_path, list(ctx.cfg["blender"]["extra_args"] or []))
    log.info("  blender: %s", exe)
    log_path = ctx.path(S, "blender.log")
    total = meta["frames"]
    t0, last = time.time(), time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        timeout = ctx.cfg["blender"]["timeout_s"]
        while proc.poll() is None:
            time.sleep(1.0)
            if time.time() - last > 10:
                last = time.time()
                done = len(list(frames_dir.glob("avatar_*.png")))
                if done:
                    rate = done / (last - t0)
                    log.info("  rendered %d/%d (~%.0fs left)", done, total, (total - done) / max(rate, 1e-6))
            if timeout and time.time() - t0 > timeout:
                proc.kill()
                raise RuntimeError(f"Blender exceeded {timeout}s; see {log_path}")
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if line.startswith("[avatarpipe]"):
            log.info("  %s", line[len("[avatarpipe] "):])
    if proc.returncode != 0:
        tail = "\n".join(Path(log_path).read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"Blender failed (exit {proc.returncode}). Last lines of {log_path}:\n{tail}")

    pngs = sorted(frames_dir.glob("avatar_*.png"))
    if len(pngs) != total:
        raise RuntimeError(f"expected {total} rendered frames, found {len(pngs)}; see {log_path}")
    outs = [frames_dir, job_path, log_path]
    if job["blend_out"]:
        outs.append(Path(job["blend_out"]))
    return outs
