"""SRanipal lip shape indices understood by the VRCFaceTracking PalBuddyGuy module."""

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


def shape_id(name_or_id):
    """Accept either a shape name ("JawOpen") or a raw index and return the index."""
    if isinstance(name_or_id, int):
        return name_or_id
    if isinstance(name_or_id, str) and name_or_id.isdigit():
        return int(name_or_id)
    return LIP_SHAPES[name_or_id]
