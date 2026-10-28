"""Config loading: YAML file deep-merged over defaults, paths resolved, validated."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "project": "shot",
    "workdir": "./runs/{project}",
    "input": {
        "performance": None,      # video of you (required)
        "clean_plate": None,      # same camera, empty scene (video or image); needed for mode=replace
        "background": None,       # new scene (video or image); needed for mode=new_scene
        "fps": None,              # None = keep source fps
        "max_height": 720,        # downscale for CPU speed; final render uses this size
        "start": 0.0,             # seconds
        "end": None,              # seconds, None = to the end
    },
    "track": {
        "pose_model": "full",     # lite | full | heavy (heavy is slow on CPU)
        "face": True,             # ARKit-style blendshapes for facial expression
        "face_crop": True,        # track the face on a full-resolution close-up crop (much more detail)
        "face_crop_size": 256,    # pixels the crop is resized to before tracking
        "min_confidence": 0.5,
        "smoothing": {"min_cutoff": 1.2, "beta": 0.02, "d_cutoff": 1.0},
    },
    "solve": {
        "focal_mm": 28.0,         # real lens focal length (35mm-equivalent); phones ~26-28
        "camera_height_m": 1.2,   # height of your camera lens off the floor
        "camera_tilt_deg": 0.0,   # +down; level camera = 0
        "torso_length_m": None,   # None = measured from tracking
    },
    "matte": {
        "enabled": True,
        "model": "selfie_multiclass",   # selfie_multiclass | selfie
        "threshold": 0.5,
        "dilate_px": 12,          # grow the matte so paint-out covers edges/hair
        "feather_px": 6,
    },
    "avatar": {
        "path": None,             # .vrm | .glb/.gltf | .fbx | .blend (required for render)
        "rig_preset": "auto",     # auto | vrm | vroid | mixamo | custom
        "bone_map": None,         # yaml file for rig_preset=custom (humanoid name -> bone name)
        "face_preset": "auto",    # auto | arkit | vroid | none | custom
        "face_map": None,         # yaml for face_preset=custom
        "face_gain": 1.4,         # expression strength; webcam-style trackers read faces a bit flat
        "twist_axis": "X",        # bone axis that should point to character's left; flip to -X if limbs twist backwards
        "scale": "auto",          # "auto" matches avatar leg length to yours, or a number
        "armature": None,         # object name, if the file has several armatures
    },
    "lipsync": {
        "enabled": True,          # mouth shapes from your voice (needs Rhubarb Lip Sync installed)
        "executable": "rhubarb",  # name on PATH, or full path to rhubarb / rhubarb.exe
        "recognizer": "pocketSphinx",   # pocketSphinx (English, best) | phonetic (any language)
        "mode": "max",            # max: whichever opens the mouth more, camera or audio | audio: audio only
        "strength": 1.0,
    },
    "blender": {
        "executable": "blender",
        "engine": "EEVEE",        # EEVEE | CYCLES | WORKBENCH
        "samples": 16,
        "shadow_catcher": True,   # Cycles only: soft shadows onto the real floor
        "sun_strength": 3.0,
        "sun_angle_deg": [50, 0, 30],
        "world_strength": 0.8,
        "extra_args": [],
        "save_blend": True,       # keep a .blend so you can open and tweak the shot
        "timeout_s": None,
    },
    "composite": {
        "mode": "replace",        # replace | new_scene | overlay
        "background_scale": "fill",   # fill | fit
        "avatar_offset_px": [0, 0],
        "grade": {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "gamma": 1.0},
        "grain": 0,               # 0-20, ties CG into camera noise
        "crf": 18,
        "preset": "veryfast",
        "audio": True,            # carry audio from the performance
        "output": "final.mp4",    # relative to workdir unless absolute
    },
}

VALID = {
    ("track", "pose_model"): {"lite", "full", "heavy"},
    ("matte", "model"): {"selfie_multiclass", "selfie"},
    ("avatar", "rig_preset"): {"auto", "vrm", "vroid", "mixamo", "custom"},
    ("avatar", "face_preset"): {"auto", "arkit", "vroid", "none", "custom"},
    ("avatar", "twist_axis"): {"X", "-X", "Z", "-Z"},
    ("blender", "engine"): {"EEVEE", "CYCLES", "WORKBENCH"},
    ("composite", "mode"): {"replace", "new_scene", "overlay"},
    ("composite", "background_scale"): {"fill", "fit"},
    ("lipsync", "recognizer"): {"pocketSphinx", "phonetic"},
    ("lipsync", "mode"): {"max", "audio"},
}

PATH_KEYS = [
    ("input", "performance"), ("input", "clean_plate"), ("input", "background"),
    ("avatar", "path"), ("avatar", "bone_map"), ("avatar", "face_map"),
]


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | Path) -> dict:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    if not isinstance(user, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"unknown top-level keys: {sorted(unknown)}")
    for section, val in user.items():
        if isinstance(DEFAULTS[section], dict) and isinstance(val, dict):
            bad = set(val) - set(DEFAULTS[section])
            if bad:
                raise ConfigError(f"unknown keys in '{section}': {sorted(bad)}")
    cfg = deep_merge(DEFAULTS, user)
    return finalize(cfg, base_dir=path.parent)


def finalize(cfg: dict, base_dir: Path) -> dict:
    base_dir = Path(base_dir)

    def res(p):
        if p is None:
            return None
        p = Path(str(p)).expanduser()
        return str(p if p.is_absolute() else (base_dir / p).resolve())

    for sec, key in PATH_KEYS:
        cfg[sec][key] = res(cfg[sec][key])

    cfg["workdir"] = res(str(cfg["workdir"]).format(project=cfg["project"]))
    out = Path(str(cfg["composite"]["output"]))
    cfg["composite"]["output"] = str(out if out.is_absolute() else Path(cfg["workdir"]) / out)
    cfg["_base_dir"] = str(base_dir)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    errs = []
    for (sec, key), allowed in VALID.items():
        if cfg[sec][key] not in allowed:
            errs.append(f"{sec}.{key} = {cfg[sec][key]!r}; expected one of {sorted(allowed)}")
    if not cfg["input"]["performance"]:
        errs.append("input.performance is required (video of you)")
    mode = cfg["composite"]["mode"]
    if mode == "replace" and not cfg["input"]["clean_plate"]:
        errs.append("composite.mode=replace needs input.clean_plate (record the empty scene)")
    if mode == "new_scene" and not cfg["input"]["background"]:
        errs.append("composite.mode=new_scene needs input.background")
    if cfg["avatar"]["rig_preset"] == "custom" and not cfg["avatar"]["bone_map"]:
        errs.append("avatar.rig_preset=custom needs avatar.bone_map")
    if cfg["avatar"]["face_preset"] == "custom" and not cfg["avatar"]["face_map"]:
        errs.append("avatar.face_preset=custom needs avatar.face_map")
    sc = cfg["avatar"]["scale"]
    if sc != "auto" and not isinstance(sc, (int, float)):
        errs.append("avatar.scale must be 'auto' or a number")
    if not 0 <= int(cfg["composite"]["grain"]) <= 20:
        errs.append("composite.grain must be 0-20")
    if errs:
        raise ConfigError("config problems:\n  - " + "\n  - ".join(errs))


def section_view(cfg: dict, *sections: str) -> dict:
    """The parts of the config a stage depends on (used for cache keys)."""
    return {s: cfg[s] for s in sections}
