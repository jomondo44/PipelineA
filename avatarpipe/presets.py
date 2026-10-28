"""Rig and face presets. Everything is keyed on VRM humanoid bone names so one drive spec
works for any rig; a preset just maps humanoid name -> the rig's actual bone name."""
from __future__ import annotations

# How each humanoid bone is driven by the solved points (see landmarks.derived()).
#   track: point the bone's Y axis at this point (DAMPED_TRACK)
#   twist: then roll the bone so its twist axis faces this point (LOCKED_TRACK, Y locked)
#   copy_location: pin the bone's head to this point (hips only: carries root motion)
DRIVE = {
    "hips":          {"copy_location": "hip_mid", "track": "spine_mid", "twist": "l_hip"},
    "spine":         {"track": "shoulder_mid", "twist": "l_shoulder"},
    "neck":          {"track": "ear_mid"},
    "head":          {"track": "head_top", "twist": "head_side"},
    "leftUpperArm":  {"track": "l_elbow"},
    "leftLowerArm":  {"track": "l_wrist"},
    "leftHand":      {"track": "l_hand"},
    "rightUpperArm": {"track": "r_elbow"},
    "rightLowerArm": {"track": "r_wrist"},
    "rightHand":     {"track": "r_hand"},
    "leftUpperLeg":  {"track": "l_knee"},
    "leftLowerLeg":  {"track": "l_ankle"},
    "leftFoot":      {"track": "l_foot_index"},
    "rightUpperLeg": {"track": "r_knee"},
    "rightLowerLeg": {"track": "r_ankle"},
    "rightFoot":     {"track": "r_foot_index"},
}

# Bones used to measure the avatar's leg length for auto-scaling
LEG_CHAIN = [("leftUpperLeg", "leftLowerLeg"), ("leftLowerLeg", "leftFoot")]

RIGS = {
    # VRoid Studio exports (also what the VRM add-on keeps as bone names)
    "vroid": {
        "hips": "J_Bip_C_Hips", "spine": "J_Bip_C_Spine", "chest": "J_Bip_C_Chest",
        "upperChest": "J_Bip_C_UpperChest", "neck": "J_Bip_C_Neck", "head": "J_Bip_C_Head",
        "leftUpperArm": "J_Bip_L_UpperArm", "leftLowerArm": "J_Bip_L_LowerArm", "leftHand": "J_Bip_L_Hand",
        "rightUpperArm": "J_Bip_R_UpperArm", "rightLowerArm": "J_Bip_R_LowerArm", "rightHand": "J_Bip_R_Hand",
        "leftUpperLeg": "J_Bip_L_UpperLeg", "leftLowerLeg": "J_Bip_L_LowerLeg", "leftFoot": "J_Bip_L_Foot",
        "rightUpperLeg": "J_Bip_R_UpperLeg", "rightLowerLeg": "J_Bip_R_LowerLeg", "rightFoot": "J_Bip_R_Foot",
    },
    # Mixamo characters imported from FBX
    "mixamo": {
        "hips": "mixamorig:Hips", "spine": "mixamorig:Spine", "chest": "mixamorig:Spine1",
        "upperChest": "mixamorig:Spine2", "neck": "mixamorig:Neck", "head": "mixamorig:Head",
        "leftUpperArm": "mixamorig:LeftArm", "leftLowerArm": "mixamorig:LeftForeArm", "leftHand": "mixamorig:LeftHand",
        "rightUpperArm": "mixamorig:RightArm", "rightLowerArm": "mixamorig:RightForeArm",
        "rightHand": "mixamorig:RightHand",
        "leftUpperLeg": "mixamorig:LeftUpLeg", "leftLowerLeg": "mixamorig:LeftLeg", "leftFoot": "mixamorig:LeftFoot",
        "rightUpperLeg": "mixamorig:RightUpLeg", "rightLowerLeg": "mixamorig:RightLeg",
        "rightFoot": "mixamorig:RightFoot",
    },
    # "vrm": resolved inside Blender from the VRM add-on's humanoid mapping (works for any VRM)
}

# Face: target shape key -> {ARKit source: weight}. Values are summed and clamped to 0..1.
FACES = {
    # VRoid Studio's standard expression keys. Negative weights subtract, so a key can
    # mean "this but not that" (sad brows = inner brows up WITHOUT the outer brows).
    # Keys a model doesn't have are skipped.
    "vroid": {
        # mouth vowels (also what lip sync drives)
        "Fcl_MTH_A": {"jawOpen": 1.0, "mouthFunnel": -0.3, "mouthPucker": -0.3},
        "Fcl_MTH_I": {"mouthStretchLeft": 0.5, "mouthStretchRight": 0.5},
        "Fcl_MTH_U": {"mouthPucker": 1.0},
        "Fcl_MTH_E": {"mouthLowerDownLeft": 0.5, "mouthLowerDownRight": 0.5, "jawOpen": 0.2},
        "Fcl_MTH_O": {"mouthFunnel": 1.0},
        # mouth moods
        "Fcl_MTH_Fun": {"mouthSmileLeft": 0.5, "mouthSmileRight": 0.5},
        "Fcl_MTH_Sorrow": {"mouthFrownLeft": 0.5, "mouthFrownRight": 0.5},
        "Fcl_MTH_Angry": {"mouthPressLeft": 0.5, "mouthPressRight": 0.5, "mouthShrugLower": 0.3},
        "Fcl_MTH_Close": {"mouthClose": 1.0, "mouthRollLower": 0.3, "mouthRollUpper": 0.3},
        # eyes
        "Fcl_EYE_Close_L": {"eyeBlinkLeft": 1.0},
        "Fcl_EYE_Close_R": {"eyeBlinkRight": 1.0},
        "Fcl_EYE_Joy_L": {"cheekSquintLeft": 1.0, "eyeBlinkLeft": -0.5},
        "Fcl_EYE_Joy_R": {"cheekSquintRight": 1.0, "eyeBlinkRight": -0.5},
        "Fcl_EYE_Surprised": {"eyeWideLeft": 0.5, "eyeWideRight": 0.5},
        "Fcl_EYE_Angry": {"eyeSquintLeft": 0.5, "eyeSquintRight": 0.5},
        # brows
        "Fcl_BRW_Surprised": {"browOuterUpLeft": 0.5, "browOuterUpRight": 0.5, "browInnerUp": 0.3},
        "Fcl_BRW_Sorrow": {"browInnerUp": 1.0, "browOuterUpLeft": -0.5, "browOuterUpRight": -0.5},
        "Fcl_BRW_Angry": {"browDownLeft": 0.5, "browDownRight": 0.5},
        "Fcl_BRW_Fun": {"mouthSmileLeft": 0.3, "mouthSmileRight": 0.3, "cheekSquintLeft": 0.2,
                        "cheekSquintRight": 0.2},
    },
    # "arkit": identity map built in Blender for every ARKit-named shape key the mesh has
}

ARKIT_PRESET = "arkit"
