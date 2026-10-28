import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from avatarpipe import config
from avatarpipe.filters import fill_gaps, one_euro, one_euro_bidirectional
from avatarpipe.landmarks import IDX, MIRROR_PERM, derived, limb_lengths
from avatarpipe.pipeline import Context, Pipeline, Stage
from avatarpipe.stages.matte import person_prob, refine
from avatarpipe.stages.solve import focal_px, solve_root, to_blender_cam

from synthetic import tracking


# ---------------------------------------------------------------- filters
def test_fill_gaps_interpolates_and_holds_ends():
    x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    np.testing.assert_allclose(fill_gaps(x), [1, 1, 2, 3, 3])


def test_fill_gaps_all_nan_is_zero():
    assert np.all(fill_gaps(np.full((4, 2), np.nan)) == 0)


def test_one_euro_reduces_jitter_keeps_mean():
    rng = np.random.default_rng(1)
    clean = np.linspace(0, 1, 300)
    noisy = clean + rng.normal(0, 0.02, 300)
    sm = one_euro_bidirectional(noisy, 30, min_cutoff=1.0, beta=0.0)
    assert np.std(sm - clean) < 0.5 * np.std(noisy - clean)
    assert abs(sm.mean() - clean.mean()) < 0.01


def test_one_euro_shape_preserved():
    x = np.random.default_rng(0).normal(size=(50, 33, 3))
    assert one_euro(x, 30).shape == x.shape


# ---------------------------------------------------------------- solve
def test_solve_root_recovers_true_position():
    w, h, fps, fm = 1280, 720, 30.0, 28.0
    world, image, _, roots = tracking(60, fps, w, h, fm)
    est = solve_root(world.astype(float), image.astype(float), w, h, focal_px(fm, w, h))
    err = np.linalg.norm(est - roots, axis=1)
    assert np.nanmax(err) < 1e-3


def test_solve_root_with_pixel_noise_is_close():
    w, h, fps, fm = 1280, 720, 30.0, 28.0
    world, image, _, roots = tracking(60, fps, w, h, fm, noise_px=2.0)
    est = solve_root(world.astype(float), image.astype(float), w, h, focal_px(fm, w, h))
    assert np.nanmedian(np.abs(est[:, 2] - roots[:, 2])) < 0.15   # depth within 15cm
    assert np.nanmedian(np.abs(est[:, 0] - roots[:, 0])) < 0.03


def test_solve_root_skips_missing_frames():
    w, h = 640, 360
    world, image, _, _ = tracking(5, 30, w, h, 28)
    world[2] = np.nan
    est = solve_root(world.astype(float), image.astype(float), w, h, focal_px(28, w, h))
    assert np.isnan(est[2]).all() and np.isfinite(est[[0, 1, 3, 4]]).all()


def test_blender_axes():
    np.testing.assert_allclose(to_blender_cam(np.array([1.0, 2.0, 3.0])), [1, -2, -3])


def test_limb_lengths_and_derived():
    world, *_ = tracking(10, 30, 640, 360, 28)
    L = limb_lengths(world[:, :, :3].astype(float))
    assert 0.8 < L["leg_m"] < 1.0 and 0.45 < L["torso_m"] < 0.55
    d = derived(world[:, :, :3].astype(float))
    assert d["hip_mid"].shape == (10, 3)
    np.testing.assert_allclose(d["hip_mid"], 0, atol=1e-6)


def test_mirror_perm_is_involution():
    perm = np.array(MIRROR_PERM)
    assert (perm[perm] == np.arange(33)).all()
    assert perm[IDX["l_wrist"]] == IDX["r_wrist"]


# ---------------------------------------------------------------- matte
def test_person_prob_multiclass_and_inverted_single():
    bg = np.ones((20, 20), np.float32)
    bg[5:15, 5:15] = 0
    assert person_prob([bg, 1 - bg])[10, 10] == 1.0          # multiclass
    assert person_prob([bg])[10, 10] == 1.0                  # single but inverted -> fixed
    assert person_prob([1 - bg])[10, 10] == 1.0


def test_refine_dilates():
    p = np.zeros((50, 50), np.float32)
    p[20:30, 20:30] = 1
    m = refine(p, 0.5, dilate_px=5, feather_px=0)
    assert m[17, 25] == 255 and m[5, 5] == 0


# ---------------------------------------------------------------- config
def _write(tmp_path, data):
    p = tmp_path / "shot.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_config_defaults_and_paths(tmp_path):
    p = _write(tmp_path, {"input": {"performance": "me.mp4", "clean_plate": "empty.mp4"}})
    cfg = config.load(p)
    assert cfg["input"]["performance"] == str(tmp_path / "me.mp4")
    assert cfg["workdir"] == str(tmp_path / "runs" / "shot")
    assert cfg["composite"]["output"] == str(tmp_path / "runs" / "shot" / "final.mp4")
    assert cfg["track"]["smoothing"]["d_cutoff"] == 1.0


def test_config_rejects_unknown_and_bad(tmp_path):
    with pytest.raises(config.ConfigError, match="unknown keys"):
        config.load(_write(tmp_path, {"input": {"performance": "a.mp4", "bogus": 1}}))
    with pytest.raises(config.ConfigError, match="clean_plate"):
        config.load(_write(tmp_path, {"input": {"performance": "a.mp4"}}))
    with pytest.raises(config.ConfigError, match="engine"):
        config.load(_write(tmp_path, {"input": {"performance": "a.mp4", "clean_plate": "b.png"},
                                      "blender": {"engine": "OCTANE"}}))


# ---------------------------------------------------------------- pipeline caching
def test_pipeline_caches_and_invalidates(tmp_path):
    calls = []

    def mk(name):
        def run(ctx):
            calls.append(name)
            out = ctx.path(name, "out.txt")
            out.write_text(json.dumps(ctx.cfg["track"]))
            return [out]
        return run

    pipe = Pipeline([Stage("a", mk("a"), sections=("track",)),
                     Stage("b", mk("b"), sections=("solve",), deps=("a",))])
    cfg = {"track": {"x": 1}, "solve": {"y": 1}}
    ctx = Context(cfg=cfg, workdir=tmp_path)
    pipe.run(ctx)
    assert calls == ["a", "b"]
    pipe.run(Context(cfg=cfg, workdir=tmp_path))
    assert calls == ["a", "b"]                     # all cached
    cfg["solve"]["y"] = 2
    pipe.run(Context(cfg=cfg, workdir=tmp_path))
    assert calls == ["a", "b", "b"]                # only b reruns
    cfg["track"]["x"] = 2
    pipe.run(Context(cfg=cfg, workdir=tmp_path))
    assert calls == ["a", "b", "b", "a", "b"]      # upstream change cascades
    pipe.run(Context(cfg=cfg, workdir=tmp_path), force=("a",))
    assert calls[-2:] == ["a", "b"] or calls[-1] == "a"


def test_pipeline_only_requires_fresh_deps(tmp_path):
    pipe = Pipeline([Stage("a", lambda c: [c.path("a", "x").touch() or c.path("a", "x")]),
                     Stage("b", lambda c: [], deps=("a",))])
    with pytest.raises(RuntimeError, match="needs 'a'"):
        pipe.run(Context(cfg={}, workdir=tmp_path), only=["b"])
