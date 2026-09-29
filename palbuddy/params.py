"""Output targets.

Two naming schemes are accepted as a class's target:
  * SRanipal lip shapes (LIP_SHAPES) - what the original script and the old
    VRCFaceTracking module used,
  * Unified Expressions (UNIFIED_EXPRESSIONS) - VRCFaceTracking v5/v6's shapes,
    including eye-area shapes like BrowLowererLeft that SRanipal can't express.

The v6 module (protocol 2) receives Unified Expression names; SRanipal names are
translated with LEGACY_TO_UNIFIED. The old module (protocol 1) only understands
SRanipal names.
"""

LIP_SHAPES = {
    "JawRight": 0,  # +JawX
    "JawLeft": 1,  # -JawX
    "JawForward": 2,
    "JawOpen": 3,
    "MouthApeShape": 4,
    "MouthUpperRight": 5,  # +MouthUpper
    "MouthUpperLeft": 6,  # -MouthUpper
    "MouthLowerRight": 7,  # +MouthLower
    "MouthLowerLeft": 8,  # -MouthLower
    "MouthUpperOverturn": 9,
    "MouthLowerOverturn": 10,
    "MouthPout": 11,
    "MouthSmileRight": 12,  # +SmileSadRight
    "MouthSmileLeft": 13,  # +SmileSadLeft
    "MouthSadRight": 14,  # -SmileSadRight
    "MouthSadLeft": 15,  # -SmileSadLeft
    "CheekPuffRight": 16,
    "CheekPuffLeft": 17,
    "CheekSuck": 18,
    "MouthUpperUpRight": 19,
    "MouthUpperUpLeft": 20,
    "MouthLowerDownRight": 21,
    "MouthLowerDownLeft": 22,
    "MouthUpperInside": 23,
    "MouthLowerInside": 24,
    "MouthLowerOverlay": 25,
    "TongueLongStep1": 26,
    "TongueLeft": 27,  # -TongueX
    "TongueRight": 28,  # +TongueX
    "TongueUp": 29,  # +TongueY
    "TongueDown": 30,  # -TongueY
    "TongueRoll": 31,
    "TongueLongStep2": 32,
    "TongueUpRightMorph": 33,
    "TongueUpLeftMorph": 34,
    "TongueDownRightMorph": 35,
    "TongueDownLeftMorph": 36,
}

SHAPE_NAMES = {v: k for k, v in LIP_SHAPES.items()}

# VRCFaceTracking.Core.Params.Expressions.UnifiedExpressions, in enum order (VRCFT v6)
UNIFIED_EXPRESSIONS = (
    "EyeSquintRight", "EyeSquintLeft", "EyeWideRight", "EyeWideLeft", "BrowPinchRight", "BrowPinchLeft",
    "BrowLowererRight", "BrowLowererLeft", "BrowInnerUpRight", "BrowInnerUpLeft", "BrowOuterUpRight",
    "BrowOuterUpLeft", "NasalDilationRight", "NasalDilationLeft", "NasalConstrictRight", "NasalConstrictLeft",
    "CheekSquintRight", "CheekSquintLeft", "CheekPuffRight", "CheekPuffLeft", "CheekSuckRight",
    "CheekSuckLeft", "JawOpen", "JawRight", "JawLeft", "JawForward", "JawBackward", "JawClench",
    "JawMandibleRaise", "MouthClosed", "LipSuckUpperRight", "LipSuckUpperLeft", "LipSuckLowerRight",
    "LipSuckLowerLeft", "LipSuckCornerRight", "LipSuckCornerLeft", "LipFunnelUpperRight",
    "LipFunnelUpperLeft", "LipFunnelLowerRight", "LipFunnelLowerLeft", "LipPuckerUpperRight",
    "LipPuckerUpperLeft", "LipPuckerLowerRight", "LipPuckerLowerLeft", "MouthUpperUpRight",
    "MouthUpperUpLeft", "MouthUpperDeepenRight", "MouthUpperDeepenLeft", "NoseSneerRight", "NoseSneerLeft",
    "MouthLowerDownRight", "MouthLowerDownLeft", "MouthUpperRight", "MouthUpperLeft", "MouthLowerRight",
    "MouthLowerLeft", "MouthCornerPullRight", "MouthCornerPullLeft", "MouthCornerSlantRight",
    "MouthCornerSlantLeft", "MouthFrownRight", "MouthFrownLeft", "MouthStretchRight", "MouthStretchLeft",
    "MouthDimpleRight", "MouthDimpleLeft", "MouthRaiserUpper", "MouthRaiserLower", "MouthPressRight",
    "MouthPressLeft", "MouthTightenerRight", "MouthTightenerLeft", "TongueOut", "TongueUp", "TongueDown",
    "TongueRight", "TongueLeft", "TongueRoll", "TongueBendDown", "TongueCurlUp", "TongueSquish", "TongueFlat",
    "TongueTwistRight", "TongueTwistLeft", "SoftPalateClose", "ThroatSwallow", "NeckFlexRight",
    "NeckFlexLeft",
)

# SRanipal lip shape -> Unified Expression shape(s) it corresponds to
LEGACY_TO_UNIFIED = {
    "JawRight": ("JawRight",), "JawLeft": ("JawLeft",), "JawForward": ("JawForward",), "JawOpen": ("JawOpen",),
    "MouthApeShape": ("MouthClosed",),
    "MouthUpperRight": ("MouthUpperRight",), "MouthUpperLeft": ("MouthUpperLeft",),
    "MouthLowerRight": ("MouthLowerRight",), "MouthLowerLeft": ("MouthLowerLeft",),
    "MouthUpperOverturn": ("LipFunnelUpperRight", "LipFunnelUpperLeft"),
    "MouthLowerOverturn": ("LipFunnelLowerRight", "LipFunnelLowerLeft"),
    "MouthPout": ("LipPuckerUpperRight", "LipPuckerUpperLeft", "LipPuckerLowerRight", "LipPuckerLowerLeft"),
    "MouthSmileRight": ("MouthCornerPullRight", "MouthCornerSlantRight"),
    "MouthSmileLeft": ("MouthCornerPullLeft", "MouthCornerSlantLeft"),
    "MouthSadRight": ("MouthFrownRight",), "MouthSadLeft": ("MouthFrownLeft",),
    "CheekPuffRight": ("CheekPuffRight",), "CheekPuffLeft": ("CheekPuffLeft",),
    "CheekSuck": ("CheekSuckRight", "CheekSuckLeft"),
    "MouthUpperUpRight": ("MouthUpperUpRight",), "MouthUpperUpLeft": ("MouthUpperUpLeft",),
    "MouthLowerDownRight": ("MouthLowerDownRight",), "MouthLowerDownLeft": ("MouthLowerDownLeft",),
    "MouthUpperInside": ("LipSuckUpperRight", "LipSuckUpperLeft"),
    "MouthLowerInside": ("LipSuckLowerRight", "LipSuckLowerLeft"),
    "MouthLowerOverlay": ("MouthRaiserLower",),
    "TongueLongStep1": ("TongueOut",), "TongueLongStep2": ("TongueOut",),
    "TongueLeft": ("TongueLeft",), "TongueRight": ("TongueRight",),
    "TongueUp": ("TongueUp",), "TongueDown": ("TongueDown",), "TongueRoll": ("TongueRoll",),
    "TongueUpRightMorph": ("TongueUp", "TongueRight"), "TongueUpLeftMorph": ("TongueUp", "TongueLeft"),
    "TongueDownRightMorph": ("TongueDown", "TongueRight"), "TongueDownLeftMorph": ("TongueDown", "TongueLeft"),
}

_UNIFIED_SET = frozenset(UNIFIED_EXPRESSIONS)


def is_valid_target(name):
    return name in LIP_SHAPES or name in _UNIFIED_SET


def unified_targets(name):
    """Unified Expression names driven by a target (SRanipal names are translated)."""
    if name in LEGACY_TO_UNIFIED:
        return LEGACY_TO_UNIFIED[name]
    if name in _UNIFIED_SET:
        return (name,)
    raise KeyError(name)


def legacy_index(name):
    """SRanipal index for the old VRCFT module, or None if the target has no SRanipal equivalent."""
    return LIP_SHAPES.get(name)


def all_target_names():
    """SRanipal names (in SRanipal order) followed by Unified-only names, for pickers."""
    legacy = sorted(LIP_SHAPES, key=LIP_SHAPES.get)
    return legacy + [n for n in UNIFIED_EXPRESSIONS if n not in LIP_SHAPES]


def shape_id(name_or_id):
    """SRanipal index of a lip shape name ("JawOpen") or a raw index (legacy helper)."""
    if isinstance(name_or_id, int):
        return name_or_id
    if isinstance(name_or_id, str) and name_or_id.isdigit():
        return int(name_or_id)
    return LIP_SHAPES[name_or_id]
