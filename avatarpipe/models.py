"""MediaPipe model files: downloaded once to a cache dir (all run on CPU)."""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger("avatarpipe")

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "pose_lite": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_full": f"{BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_heavy": f"{BASE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "face": f"{BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "selfie": f"{BASE}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "selfie_multiclass": f"{BASE}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}


def cache_dir() -> Path:
    d = Path(os.environ.get("AVATARPIPE_MODELS") or Path.home() / ".cache" / "avatarpipe" / "models")
    d.mkdir(parents=True, exist_ok=True)
    return d


def get(name: str) -> str:
    """Path to a model file, downloading it on first use.

    Offline? Put the file in the cache dir yourself (or set AVATARPIPE_MODELS).
    """
    url = MODELS[name]
    dest = cache_dir() / url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    log.info("downloading model %s -> %s", name, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download {name} model from {url}: {e}\n"
            f"Download it manually and place it at {dest}") from e
    return str(dest)
