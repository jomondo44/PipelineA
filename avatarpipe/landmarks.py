"""MediaPipe Pose landmark layout and the derived points the avatar is driven by."""
from __future__ import annotations

import numpy as np

POSE = [
    "nose", "l_eye_inner", "l_eye", "l_eye_outer", "r_eye_inner", "r_eye", "r_eye_outer",
    "l_ear", "r_ear", "mouth_l", "mouth_r", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_pinky", "r_pinky", "l_index", "r_index", "l_thumb", "r_thumb",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_heel", "r_heel",
    "l_foot_index", "r_foot_index",
]
IDX = {n: i for i, n in enumerate(POSE)}

# For mirrored (selfie) footage: swap left/right landmark slots
MIRROR_PERM = [IDX[n.replace("l_", "#").replace("r_", "l_").replace("#", "r_")] if n[:2] in ("l_", "r_") else i
               for i, n in enumerate(POSE)]

# Points used for least-squares root placement (stable, usually visible)
SOLVE_POINTS = ["l_shoulder", "r_shoulder", "l_hip", "r_hip", "l_elbow", "r_elbow",
                "l_knee", "r_knee", "l_ankle", "r_ankle", "l_wrist", "r_wrist", "nose", "l_ear", "r_ear"]

# The 52 ARKit-style blendshape names FaceLandmarker emits (index 0 is _neutral)
ARKIT = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
ARKIT_MIRROR = {n: (n.replace("Left", "#").replace("Right", "Left").replace("#", "Right")) for n in ARKIT}


def derived(p: np.ndarray) -> dict[str, np.ndarray]:
    """p: (T, 33, 3) camera-space positions. Returns named (T, 3) tracks for driving bones."""
    g = lambda n: p[:, IDX[n]]  # noqa: E731
    hip = 0.5 * (g("l_hip") + g("r_hip"))
    sho = 0.5 * (g("l_shoulder") + g("r_shoulder"))
    ear = 0.5 * (g("l_ear") + g("r_ear"))
    up = ear - sho
    out = {
        "hip_mid": hip,
        "shoulder_mid": sho,
        "spine_mid": hip + 0.5 * (sho - hip),
        "neck": sho + 0.35 * up,
        "head_top": ear + 0.6 * up,
        "ear_mid": ear,
        "l_hand": 0.5 * (g("l_index") + g("l_pinky")),
        "r_hand": 0.5 * (g("r_index") + g("r_pinky")),
    }
    for n in POSE:
        out[n] = g(n)
    return out


def limb_lengths(world: np.ndarray) -> dict[str, float]:
    """Median performer segment lengths (meters) from MediaPipe world landmarks (T, 33, 3)."""
    def seg(a, b):
        d = np.linalg.norm(world[:, IDX[a]] - world[:, IDX[b]], axis=-1)
        d = d[np.isfinite(d)]
        return float(np.median(d)) if d.size else float("nan")

    upper = np.nanmean([seg("l_hip", "l_knee"), seg("r_hip", "r_knee")])
    lower = np.nanmean([seg("l_knee", "l_ankle"), seg("r_knee", "r_ankle")])
    hip = 0.5 * (world[:, IDX["l_hip"]] + world[:, IDX["r_hip"]])
    sho = 0.5 * (world[:, IDX["l_shoulder"]] + world[:, IDX["r_shoulder"]])
    torso = np.linalg.norm(sho - hip, axis=-1)
    torso = torso[np.isfinite(torso)]
    return {
        "leg_m": float(upper + lower),
        "torso_m": float(np.median(torso)) if torso.size else float("nan"),
        "arm_m": float(np.nanmean([seg("l_shoulder", "l_elbow") + seg("l_elbow", "l_wrist"),
                                   seg("r_shoulder", "r_elbow") + seg("r_elbow", "r_wrist")])),
    }
