"""
Best-effort mapping from MediaPipe BlazePose's 33 landmarks to the 25-joint
skeleton layout the EC3D dataset uses (EC3D's own data-processing code was
adapted from NTU RGB+D / Microsoft's SGN, which uses the standard Kinect v2
25-joint layout -- see classifier/train.py's docstring for the citation).

READ THIS BEFORE TRUSTING THE CLASSIFIER'S OUTPUT:
This mapping is NOT validated against ground truth. It's a reasonable,
documented best-effort bridge (most joints map directly -- shoulders,
elbows, wrists, hips, knees, ankles -- a few are approximated where
BlazePose has no equivalent point, see comments below), but two things are
still unverified:
  1. Coordinate frame compatibility: EC3D's 3D coordinates come from
     triangulating 4 calibrated GoPro cameras (real-world meters, origin
     wherever their rig was set up). BlazePose's world landmarks are also
     meters, but hip-centered with the model's own axis convention. These
     are NOT guaranteed to line up in scale/orientation without extra
     normalization -- the model may perform worse on live data than its
     training/test accuracy suggested, purely from this mismatch.
  2. A few NTU joints (hand tips, thumbs, spine-mid, neck) don't have a
     direct BlazePose equivalent and are approximated by interpolation --
     see NTU_FROM_BLAZEPOSE below.

Treat any prediction from this bridge as an experimental second opinion
alongside the rule-based classifier (frontend/js/deviationRules.js), not
as ground truth. If you want a live classifier you can actually trust,
the more reliable fix is training a small model directly on the live
pipeline's own 4 joint-angle channels (frontend/js/jointAngles.js) instead
of bridging into EC3D's coordinate space -- see the README section on this.
"""
from __future__ import annotations

# BlazePose (MediaPipe Pose / PoseLandmarker) landmark indices used below.
BP = {
    "NOSE": 0,
    "L_EAR": 7, "R_EAR": 8,
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW": 13, "R_ELBOW": 14,
    "L_WRIST": 15, "R_WRIST": 16,
    "L_PINKY": 17, "R_PINKY": 18,
    "L_INDEX": 19, "R_INDEX": 20,
    "L_THUMB": 21, "R_THUMB": 22,
    "L_HIP": 23, "R_HIP": 24,
    "L_KNEE": 25, "R_KNEE": 26,
    "L_ANKLE": 27, "R_ANKLE": 28,
    "L_FOOT_INDEX": 31, "R_FOOT_INDEX": 32,
}

# NTU RGB+D 25-joint order (1-indexed in the original Kinect v2 spec;
# 0-indexed here to match array position). This ordering is what EC3D's
# poses array's last axis (size 25) is assumed to follow, per their
# acknowledged adaptation from the SGN/NTU RGB+D pipeline.
NTU_JOINT_NAMES = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight",
]


def _mid(a, b):
    return {"x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2, "z": (a["z"] + b["z"]) / 2}


def _lerp(a, b, t):
    return {
        "x": a["x"] + (b["x"] - a["x"]) * t,
        "y": a["y"] + (b["y"] - a["y"]) * t,
        "z": a["z"] + (b["z"] - a["z"]) * t,
    }


def blazepose_frame_to_ntu25(landmarks: list[dict]) -> list[list[float]]:
    """
    @param landmarks: one frame's worth of 33 BlazePose landmarks, each
        {"x":..., "y":..., "z":...} (visibility ignored here).
    @returns: 25 [x,y,z] points in NTU_JOINT_NAMES order.
    """
    lm = landmarks
    hip_mid = _mid(lm[BP["L_HIP"]], lm[BP["R_HIP"]])
    shoulder_mid = _mid(lm[BP["L_SHOULDER"]], lm[BP["R_SHOULDER"]])
    head = lm[BP["NOSE"]]

    points = {
        "SpineBase": hip_mid,
        "SpineMid": _lerp(hip_mid, shoulder_mid, 0.5),  # no direct BlazePose point -- interpolated
        "Neck": _lerp(shoulder_mid, head, 0.3),  # approximated -- BlazePose has no neck landmark
        "Head": head,
        "ShoulderLeft": lm[BP["L_SHOULDER"]],
        "ElbowLeft": lm[BP["L_ELBOW"]],
        "WristLeft": lm[BP["L_WRIST"]],
        "HandLeft": lm[BP["L_WRIST"]],  # approximated -- BlazePose has no separate hand-center point
        "ShoulderRight": lm[BP["R_SHOULDER"]],
        "ElbowRight": lm[BP["R_ELBOW"]],
        "WristRight": lm[BP["R_WRIST"]],
        "HandRight": lm[BP["R_WRIST"]],
        "HipLeft": lm[BP["L_HIP"]],
        "KneeLeft": lm[BP["L_KNEE"]],
        "AnkleLeft": lm[BP["L_ANKLE"]],
        "FootLeft": lm[BP["L_FOOT_INDEX"]],
        "HipRight": lm[BP["R_HIP"]],
        "KneeRight": lm[BP["R_KNEE"]],
        "AnkleRight": lm[BP["R_ANKLE"]],
        "FootRight": lm[BP["R_FOOT_INDEX"]],
        "SpineShoulder": shoulder_mid,
        "HandTipLeft": lm[BP["L_INDEX"]],
        "ThumbLeft": lm[BP["L_THUMB"]],
        "HandTipRight": lm[BP["R_INDEX"]],
        "ThumbRight": lm[BP["R_THUMB"]],
    }

    return [[points[name]["x"], points[name]["y"], points[name]["z"]] for name in NTU_JOINT_NAMES]


def blazepose_sequence_to_ntu25(frames: list[list[dict]]) -> list[list[list[float]]]:
    """Applies blazepose_frame_to_ntu25 to a whole rep's worth of frames."""
    return [blazepose_frame_to_ntu25(frame) for frame in frames]
