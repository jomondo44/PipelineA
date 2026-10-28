"""Face close-up crop, head rotation, lip sync."""
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest

from avatarpipe import config, facecrop, headpose, media
from avatarpipe.landmarks import ARKIT, IDX
from avatarpipe.pipeline import Context
from avatarpipe.stages import ingest, lipsync, solve, track

from synthetic import tracking


# ---------------------------------------------------------------- crop
def _lm_frame():
    _, image, _, _ = tracking(1, 30, 1280, 720, 28)
    return image[0]


def test_head_box_centers_on_head():
    lm = _lm_frame()
    W, H = 3840, 2160
    cx, cy, s = facecrop.head_box(lm, W, H)
    ears = 0.5 * (lm[IDX["l_ear"], :2] + lm[IDX["r_ear"], :2]) * [W, H]
    assert abs(cx - ears[0]) < 1 and abs(cy - ears[1]) < 0.15 * s
    ear_px = np.linalg.norm((lm[IDX["l_ear"], :2] - lm[IDX["r_ear"], :2]) * [W, H])
    assert 2.0 * ear_px < s < 6 * ear_px


def test_head_box_none_when_head_hidden():
    lm = _lm_frame()
    for n in ("nose", "l_ear", "r_ear", "l_eye_outer", "r_eye_outer", "l_eye", "r_eye"):
        lm[IDX[n], 3] = 0.0
    assert facecrop.head_box(lm, 100, 100) is None
    assert facecrop.head_box(None, 100, 100) is None


def test_extract_pads_at_edges():
    frame = np.full((100, 200, 3), 50, np.uint8)
    patch = facecrop.extract(frame, (5, 5, 40), 64)
    assert patch.shape == (64, 64, 3) and patch.flags["C_CONTIGUOUS"]


def test_box_smoother_holds_then_drops():
    sm = facecrop.BoxSmoother(hold=2)
    sm.update((100, 100, 50))
    b = sm.update((110, 100, 50))
    assert 100 < b[0] < 110
    assert sm.update(None) is not None and sm.update(None) is not None
    assert sm.update(None) is None


# ---------------------------------------------------------------- head rotation
def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def as_mat(r, transposed=False):
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = [2, 1, -60]
    return m.T if transposed else m


def test_rotation_from_matrix_handles_both_layouts():
    r = rot_y(30)
    np.testing.assert_allclose(headpose.rotation_from_matrix(as_mat(r)), r, atol=1e-9)
    np.testing.assert_allclose(headpose.rotation_from_matrix(as_mat(r, True)), r, atol=1e-9)
    assert headpose.rotation_from_matrix(np.full((4, 4), np.nan)) is None


def test_crop_correction():
    np.testing.assert_allclose(headpose.crop_correction(0.5, 0.5, 640, 360, 500), np.eye(3), atol=1e-12)
    c = headpose.crop_correction(0.9, 0.2, 640, 360, 500)
    d = np.array([0.4 * 640 / 500, 0.3 * 360 / 500, -1.0])
    np.testing.assert_allclose(c @ [0, 0, -1], d / np.linalg.norm(d), atol=1e-9)
    # a face seen square-on in a crop at the frame's right really faces back at the camera
    np.testing.assert_allclose(c @ [0, 0, 1], -d / np.linalg.norm(d), atol=1e-9)


def test_head_directions_track_turn_and_self_check():
    T = 60
    yaws = np.linspace(-40, 40, T)
    mats = np.stack([as_mat(rot_y(a), transposed=True) for a in yaws])
    mats[10:14] = np.nan  # face lost briefly
    left_true = np.stack([rot_y(a)[:, 0] for a in yaws])
    up_true = np.tile([0.0, 1.0, 0.0], (T, 1))
    up, left, conf = headpose.head_directions(mats, None, 640, 360, 500, 30, left_true, up_true)
    assert np.degrees(np.arccos(np.clip(np.sum(left * left_true, 1), -1, 1)))[20:].max() < 3
    assert conf[30] > 0.99 and conf[12] < 1
    # if the face axes disagree with the ears, they're flipped to agree
    up2, left2, _ = headpose.head_directions(mats, None, 640, 360, 500, 30, -left_true, up_true)
    np.testing.assert_allclose(left2, -left, atol=1e-9)


def test_apply_head_falls_back_to_body():
    world, *_ = tracking(10, 30, 640, 360, 28)
    from avatarpipe.landmarks import derived
    pts = derived(world[:, :, :3].astype(float) * [1, -1, -1] + [0, 0, -3])
    note = solve.apply_head(pts, np.full((10, 4, 4), np.nan), None, 640, 360, 500, 30)
    assert "body" in note
    np.testing.assert_allclose(pts["head_side"], pts["l_ear"])


# ---------------------------------------------------------------- lip sync
def test_cues_to_frames_places_shapes():
    cues = [{"start": 0.0, "end": 0.5, "value": "X"}, {"start": 0.5, "end": 1.0, "value": "D"},
            {"start": 1.0, "end": 2.0, "value": "F"}]
    out = lipsync.cues_to_frames(cues, 10, 20, smooth=False)
    assert out["jawOpen"][2] == 0 and out["jawOpen"][7] == 0.9
    assert out["mouthPucker"][15] == 0.9 and out["jawOpen"][15] == 0
    sm = lipsync.cues_to_frames(cues, 10, 20)
    assert 0 < sm["jawOpen"][5] < 0.9  # smoothed edge


def test_merge_lipsync_modes():
    shapes = {"jawOpen": np.array([0.5, 0.1, 0.0]), "eyeBlinkLeft": np.array([1.0, 0, 0])}
    lip = {"jawOpen": [0.2, 0.8], "mouthPucker": [0, 1, 1]}
    m = solve.merge_lipsync(shapes, lip, 3, "max", 1.0)
    np.testing.assert_allclose(m["jawOpen"], [0.5, 0.8, 0.8])
    np.testing.assert_allclose(m["mouthPucker"], [0, 1, 1])
    np.testing.assert_allclose(m["eyeBlinkLeft"], [1, 0, 0])
    a = solve.merge_lipsync(shapes, lip, 3, "audio", 0.5)
    np.testing.assert_allclose(a["jawOpen"], [0.1, 0.4, 0.4])


def test_find_rhubarb_in_tools_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("AVATARPIPE_RHUBARB", raising=False)
    monkeypatch.setattr(lipsync.shutil, "which", lambda name: None)
    shot = tmp_path / "shots"
    shot.mkdir()
    cfg = {"lipsync": {"executable": "rhubarb"}, "_base_dir": str(shot)}
    assert lipsync.find_rhubarb(cfg) is None
    exe = tmp_path / "tools" / "Rhubarb-Lip-Sync-1.14.0-Windows" / ("rhubarb.exe" if lipsync.os.name == "nt" else "rhubarb")
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert lipsync.find_rhubarb(cfg) == str(exe.resolve())
    cfg["lipsync"]["executable"] = str(tmp_path / "nope" / "rhubarb.exe")
    assert lipsync.find_rhubarb(cfg) is None


# ---------------------------------------------------------------- track stage with stand-in MediaPipe
class _P:
    def __init__(self, x, y, z, v):
        self.x, self.y, self.z, self.visibility = x, y, z, v


def fake_vision(image_lm, world_lm, seen):
    """Stand-in for mediapipe's API, returning the synthetic skeleton and a face."""
    class Image:
        def __init__(self, image_format, data):
            self.data = data

    class Pose:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def detect_for_video(self, img, ts):
            i = min(int(round(ts * 30 / 1000)), len(image_lm) - 1)
            seen["pose_shapes"].add(img.data.shape)
            return types.SimpleNamespace(
                pose_world_landmarks=[[_P(*p) for p in world_lm[i]]],
                pose_landmarks=[[_P(*p) for p in image_lm[i]]])

    class Face:
        def detect_for_video(self, img, ts):
            seen["face_shapes"].add(img.data.shape)
            seen["face_px"].append(float(img.data.mean()))
            cats = [types.SimpleNamespace(category_name=n, score=0.25) for n in ARKIT]
            return types.SimpleNamespace(face_blendshapes=[cats], facial_transformation_matrixes=[as_mat(rot_y(10))])
        def close(self): pass

    class Factory:
        def __init__(self, cls): self.cls = cls
        def create_from_options(self, opts): return self.cls()

    opts = lambda **k: types.SimpleNamespace(**k)  # noqa: E731
    vision = types.SimpleNamespace(
        PoseLandmarkerOptions=opts, FaceLandmarkerOptions=opts, PoseLandmarker=Factory(Pose),
        FaceLandmarker=Factory(Face), RunningMode=types.SimpleNamespace(VIDEO="video"))
    mpt = types.SimpleNamespace(BaseOptions=lambda **k: k)
    mpt.BaseOptions.Delegate = types.SimpleNamespace(CPU="cpu")
    mp = types.SimpleNamespace(Image=Image, ImageFormat=types.SimpleNamespace(SRGB="srgb"))
    return mp, mpt, vision


def test_track_uses_full_res_face_crop(tmp_path, monkeypatch):
    W, H, N = 1280, 720, 12
    world, image, _, _ = tracking(N, 30, W, H, 28)
    # source video: dark frame with a bright square where the head is
    with media.FrameWriter(tmp_path / "me.mp4", W, H, "30") as wr:
        for i in range(N):
            f = np.zeros((H, W, 3), np.uint8)
            hx, hy = (image[i, IDX["nose"], :2] * [W, H]).astype(int)
            f[hy - 40:hy + 40, hx - 40:hx + 40] = 255
            wr.write(f)
    (tmp_path / "shot.yaml").write_text(
        "input: {performance: me.mp4, clean_plate: me.mp4, max_height: 360}\nlipsync: {enabled: false}\n")
    cfg = config.load(tmp_path / "shot.yaml")
    ctx = Context(cfg=cfg, workdir=Path(cfg["workdir"]))
    ingest.run(ctx)
    seen = {"pose_shapes": set(), "face_shapes": set(), "face_px": []}
    monkeypatch.setattr(track, "_vision", lambda: fake_vision(image[:, :, :], world, seen))
    monkeypatch.setattr(track.models, "get", lambda name: name)
    track.run(ctx)
    d = np.load(ctx.workdir / "track" / "tracking.npz")
    assert seen["pose_shapes"] == {(360, 640, 3)}            # body on the working-size frame
    assert seen["face_shapes"] == {(256, 256, 3)}            # face on the close-up
    assert min(seen["face_px"]) > 40                          # the crop really landed on the head
    assert np.isfinite(d["face_crop"]).all()
    np.testing.assert_allclose(d["face_crop"][0, 0], image[0, IDX["nose"], 0], atol=0.03)
    assert np.isfinite(d["face_bs"]).all() and np.isfinite(d["face_mat"]).all()
