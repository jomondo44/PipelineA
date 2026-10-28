"""End-to-end run on synthetic footage: real ingest, solve, render (Blender), composite.

Track and matte are replaced by fakes that emit what MediaPipe would (from the known
synthetic skeleton), so this runs offline with no model downloads.

  python tests/e2e_synthetic.py --blender /path/to/blender --avatar test_avatar.blend --out /tmp/e2e
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avatarpipe import config, media  # noqa: E402
from avatarpipe.pipeline import Context  # noqa: E402
from avatarpipe.stages import build  # noqa: E402
from synthetic import background, draw_person, tracking  # noqa: E402

W, H, FPS, N, FOCAL = 640, 360, 30, 48, 28.0


def make_footage(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    world, image, bs, _ = tracking(N, FPS, W, H, FOCAL)
    bg = background(W, H)
    rng = np.random.default_rng(0)
    with media.FrameWriter(d / "me.mp4", W, H, str(FPS), crf=16) as wr:
        for i in range(N):
            f = bg.copy()
            draw_person(f, image[i], thick=10)
            wr.write(np.clip(f + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8))
    with media.FrameWriter(d / "empty.mp4", W, H, str(FPS), crf=16) as wr:
        for i in range(20):
            wr.write(np.clip(bg + rng.normal(0, 2, bg.shape), 0, 255).astype(np.uint8))
    return world, image, bs


def face_mats(n: int, fps: float) -> np.ndarray:
    """Head turning +-40 degrees (positive = face turns toward image right)."""
    out = np.zeros((n, 4, 4), np.float32)
    for i in range(n):
        a = np.radians(40 * np.sin(2 * np.pi * 0.5 * i / fps))
        m = np.eye(4)
        m[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        m[:3, 3] = [0, 0, -60]
        out[i] = m.T  # MediaPipe-style layout, translation in the bottom row
    return out


def fake_track(world, image, bs, face=False):
    def run(ctx: Context):
        meta = ctx.read_json("ingest", "meta.json")
        n = meta["frames"]
        out = ctx.path("track", "tracking.npz")
        np.savez_compressed(out, pose_world=world[:n], pose_image=image[:n], face_bs=bs[:n],
                            face_mat=face_mats(n, meta["fps"]) if face else np.full((n, 4, 4), np.nan, np.float32),
                            fps=np.float32(meta["fps"]), size=np.array([meta["width"], meta["height"]], np.int32))
        return [out]
    return run


def fake_matte(image):
    from avatarpipe.stages.matte import refine

    def run(ctx: Context):
        meta = ctx.read_json("ingest", "meta.json")
        w, h = meta["width"], meta["height"]
        out = ctx.path("matte", "matte.mkv")
        with media.FrameWriter(out, w, h, meta["fps_str"], gray=True) as wr:
            for i in range(meta["frames"]):
                m = np.zeros((h, w, 3), np.uint8)
                draw_person(m, image[i], thick=10, color=(255, 255, 255))
                wr.write(refine(m[..., 0] / 255.0, 0.5, 6, 3))
        return [out]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", required=True)
    ap.add_argument("--avatar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="CYCLES")
    ap.add_argument("--mode", default="replace")
    ap.add_argument("--face", action="store_true", help="add a face-tracked head turn")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = Path(a.out).resolve()
    world, image, bs = make_footage(root / "footage")
    shot = {
        "project": "e2e",
        "input": {"performance": "footage/me.mp4", "clean_plate": "footage/empty.mp4",
                  "background": "footage/empty.mp4" if a.mode == "new_scene" else None},
        "solve": {"focal_mm": FOCAL, "camera_height_m": 1.2},
        "avatar": {"path": str(Path(a.avatar).resolve())},
        "blender": {"executable": a.blender, "engine": a.engine, "samples": 4},
        "composite": {"mode": a.mode, "grain": 3},
    }
    cfg_path = root / "shot.yaml"
    cfg_path.write_text(yaml.safe_dump(shot, sort_keys=False))
    cfg = config.load(cfg_path)
    pipe = build()
    pipe.by_name["track"].run = fake_track(world, image, bs, face=a.face)
    pipe.by_name["matte"].run = fake_matte(image)
    pipe.run(Context(cfg=cfg, workdir=Path(cfg["workdir"])))

    final = Path(cfg["composite"]["output"])
    info = media.probe(final)
    print(f"final: {final} {info['width']}x{info['height']} {info['frames']} frames")
    assert info["frames"] == N, info

    # avatar must be visible where the performer's hips are, in a mid frame
    frames = list(media.read_frames(final, W, H))
    plate = cv2.cvtColor(cv2.imread(str(Path(cfg["workdir"]) / "ingest" / "clean_plate.png")), cv2.COLOR_BGR2RGB)
    k = N // 2
    hip = 0.5 * (image[k, 23, :2] + image[k, 24, :2]) * [W, H]
    x, y = int(hip[0]), int(hip[1])
    patch = frames[k][y - 6:y + 6, x - 6:x + 6].astype(int)
    diff = np.abs(patch - plate[y - 6:y + 6, x - 6:x + 6].astype(int)).mean()
    print(f"avatar-vs-plate difference at hips: {diff:.1f}")
    assert diff > 20, "avatar not found at the performer's hip position"
    cv2.imwrite(str(root / "frame_mid.png"), cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR))
    print("E2E OK")


if __name__ == "__main__":
    main()
