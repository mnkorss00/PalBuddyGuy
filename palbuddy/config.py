"""Persistent settings (config.json) shared by the GUI and the CLI.

Replaces the constants that used to be hand-edited at the top of script.py
(DATASET_FOLDER, order, to_replace, max_power_array).
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .params import is_valid_target

# config.json lives next to script.py / PalBuddyGuy.bat, independent of the working directory
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(APP_DIR, "config.json")

# Which trackers feed the network. Recordings always store both halves
# (eye channels 0-63, face channels 64-127; a missing tracker is zero-filled),
# so a recording made with both trackers can be reused for any mode.
INPUT_MODES = ("both", "face", "eye")
INPUT_CHANNELS = {"both": slice(0, 128), "eye": slice(0, 64), "face": slice(64, 128)}


def channels_for(mode):
    return INPUT_CHANNELS[mode]


@dataclass
class ExpressionClass:
    """One output of the network.

    files: dataset recordings (.mmap, or legacy .pkl) that show this expression.
    target: SRanipal lip shape to drive with this output (name or index), or None
            to only train it (e.g. the mandatory "neutral" class).
    max_power: raw network output that maps to a fully expressed shape (+1).
    """

    name: str
    files: List[str] = field(default_factory=list)
    target: Optional[str] = None
    max_power: float = 0.9


PARAM_NAME_RE = re.compile(r"^[A-Za-z0-9_./-]+$")

# (min, max) presets for merged parameters: negative class -> min, neutral -> centre, positive -> max
RANGE_PRESETS = {"-1..1": (-1.0, 1.0), "0..1": (0.0, 1.0), "0..2": (0.0, 2.0)}


@dataclass
class MergedParam:
    """A custom avatar parameter combining two trained classes, sent to VRChat over OSC.

    value = positive class weight - negative class weight   (-1..1, 0 = neither)
    mapped linearly so that -1 -> out_min, 0 -> the middle, +1 -> out_max.
    E.g. smile (0..1) and sad (0..1) -> "SmileSad" in -1..1, or 0..2 with neutral at 1.
    Either class may be empty (e.g. only a positive side).
    """

    name: str
    positive: Optional[str] = None
    negative: Optional[str] = None
    out_min: float = -1.0
    out_max: float = 1.0
    enabled: bool = True

    def combine(self, weights):
        """weights: {class name: weight 0..1} -> output value."""
        pos = weights.get(self.positive, 0.0) if self.positive else 0.0
        neg = weights.get(self.negative, 0.0) if self.negative else 0.0
        c = min(1.0, max(-1.0, pos - neg))
        return self.out_min + (c + 1.0) / 2.0 * (self.out_max - self.out_min)

    @property
    def neutral(self):
        return (self.out_min + self.out_max) / 2.0

    @property
    def beyond_sync_range(self):
        """VRChat syncs float parameters as -1..1; larger values only work locally."""
        return min(self.out_min, self.out_max) < -1.0 or max(self.out_min, self.out_max) > 1.0


@dataclass
class Config:
    dataset_folder: str = "datasets"
    model_path: str = "buddyguy.pt"

    # Networking. Listening on 127.0.0.1 keeps the ports off the LAN.
    bind_host: str = "127.0.0.1"
    face_port: int = 18452  # SRanipal stream that is the face tracker when not swapped
    eye_port: int = 18453
    proxy_port: int = 18454  # tvm_proxy.py output (legacy two-process mode)
    vrcft_port: int = 26421
    # "direct": receive from SRanipal in this process (fastest, default)
    # "proxy":  connect to a separately running tvm_proxy.py
    source: str = "direct"
    swapped: bool = False
    stall_timeout: float = 2.0  # seconds without frames before a stream counts as stalled
    # "both" (eye + face tracker), "face" (facial tracker only) or "eye" (Pro Eye only)
    input_mode: str = "both"

    # Recording
    record_frames: int = 2048
    record_countdown: float = 5.0

    # Training
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 5e-5
    mixed_precision: bool = True
    cache_datasets_in_ram: bool = False
    model_arch: str = "standard"  # "standard" (original network) or "lite" (~8x smaller, faster)
    validation_split: float = 0.1  # end of each recording held out to measure accuracy; 0 = off

    # Inference
    smoothing: float = 0.0  # 0 = off, otherwise EMA factor in (0, 1)
    max_send_rate: float = 120.0  # Hz cap for VRCFT updates
    # VRCFT v6 module: "replace" SRanipal's value of a driven shape, or keep the "max" of both
    vrcft_override_mode: str = "replace"
    vrcft_wrap_sranipal: bool = True  # module installer: run SRanipal inside the PalBuddyGuy module
    vrcft_custom_libs: str = ""  # VRCFT modules folder; empty = %APPDATA%\VRCFaceTracking\CustomLibs
    # "auto": ONNX Runtime if installed, else PyTorch. ONNX Runtime is faster on the
    # CPU (int8) and tracking with it never loads PyTorch (~500 MB less RAM).
    infer_engine: str = "auto"  # "auto", "onnx" or "pytorch"
    # CPU by default: int8 inference costs ~1/3 of one core and leaves the GPU to VR
    infer_device: str = "cpu"  # "cpu", "auto" (GPU if available) or "gpu"
    infer_threads: int = 1  # CPU threads for inference; more = lower latency but more total CPU
    infer_int8: bool = True  # int8-quantize the linear layers when inferring on the CPU
    max_infer_rate: float = 0.0  # Hz cap for running the network, 0 = every new frame

    # Process placement (see palbuddy/system.py). Tracking needs little CPU, so by
    # default it yields to VR / the game when the CPU is busy.
    process_priority: str = "below_normal"  # "above_normal", "normal", "below_normal", "idle"
    cpu_affinity: str = "all"  # "all" or "ecores" (hybrid Intel CPUs: efficiency cores only)
    efficiency_mode: bool = False  # Windows 11 EcoQoS

    # GUI
    language: str = "auto"  # "auto", "en" or "ko"
    show_preview: bool = True

    classes: List[ExpressionClass] = field(default_factory=list)

    # Merged parameters (sent straight to VRChat over OSC, see palbuddy/osc.py)
    merged_params: List[MergedParam] = field(default_factory=list)
    osc_enabled: bool = True
    osc_host: str = "127.0.0.1"
    osc_port: int = 9000

    # ---------------------------------------------------------------- helpers
    @property
    def num_classes(self):
        return len(self.classes)

    def targets(self):
        """[(class_index, target_name, max_power)] for every class that drives a shape.
        Target names are SRanipal lip shapes or Unified Expressions (see params.py)."""
        out = []
        for i, c in enumerate(self.classes):
            if c.target not in (None, "") and is_valid_target(c.target):
                out.append((i, c.target, c.max_power))
        return out

    def dataset_path(self, filename):
        return filename if os.path.isabs(filename) else os.path.join(self.dataset_folder, filename)

    def validate(self):
        problems = []
        if self.model_arch not in ("standard", "lite"):
            problems.append("model_arch must be 'standard' or 'lite'.")
        if not 0 <= self.validation_split < 0.5:
            problems.append("validation_split must be between 0 and 0.5.")
        if self.input_mode not in INPUT_MODES:
            problems.append("input_mode must be one of %s." % ", ".join(INPUT_MODES))
        if not self.classes:
            problems.append("No expression classes are configured.")
        for c in self.classes:
            if c.target not in (None, ""):
                if not is_valid_target(c.target):
                    problems.append("Class '%s' has an unknown target shape '%s'." % (c.name, c.target))
            if c.max_power <= 0:
                problems.append("Class '%s' needs max_power > 0." % c.name)
        names = {c.name for c in self.classes}
        for m in self.merged_params:
            problems.extend(merged_param_problems(m, names))
        return problems

    # ---------------------------------------------------------------- io
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Unknown keys (from newer/older versions or hand edits) are ignored."""
        data = dict(data)
        class_fields = set(ExpressionClass.__dataclass_fields__)
        classes = [ExpressionClass(**{k: v for k, v in c.items() if k in class_fields})
                   for c in data.pop("classes", []) if isinstance(c, dict) and "name" in c]
        merged_fields = set(MergedParam.__dataclass_fields__)
        merged = [MergedParam(**{k: v for k, v in m.items() if k in merged_fields})
                  for m in data.pop("merged_params", []) if isinstance(m, dict) and "name" in m]
        known = set(cls.__dataclass_fields__)
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.classes = classes
        cfg.merged_params = merged
        return cfg

    def save(self, path=DEFAULT_CONFIG_PATH):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path=DEFAULT_CONFIG_PATH):
        if not os.path.exists(path):
            cfg = default_config()
            cfg.save(path)
            return cfg
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def merged_param_problems(m, class_names):
    """Reasons a merged parameter can't be used (empty list = fine)."""
    problems = []
    if not PARAM_NAME_RE.match(m.name or ""):
        problems.append("Merged parameter '%s': use letters, digits, _ . / - only." % m.name)
    if not m.positive and not m.negative:
        problems.append("Merged parameter '%s' needs a positive or a negative class." % m.name)
    for side in (m.positive, m.negative):
        if side and side not in class_names:
            problems.append("Merged parameter '%s' uses unknown class '%s'." % (m.name, side))
    if m.out_min == m.out_max:
        problems.append("Merged parameter '%s' needs different min and max values." % m.name)
    return problems


def default_config():
    """Same expressions and shape mapping as the original example script."""
    cfg = Config()
    cfg.classes = [
        ExpressionClass("neutral", ["neutral2-em.mmap", "neutral1-em.mmap", "nut3-em.mmap"]),
        ExpressionClass("happy", ["happy-em.mmap", "happy3-em.mmap"], "JawLeft"),
        ExpressionClass("mad", ["mad1.mmap", "mad2.mmap", "mad4-em.mmap"], "JawRight"),
        ExpressionClass("sad", ["sad-em.mmap"], "MouthPout"),
        ExpressionClass("purseleft", ["purseleft-em.mmap"], "MouthUpperUpLeft"),
        ExpressionClass("purseright", ["purseright-em.mmap"], "MouthUpperUpRight"),
        ExpressionClass("open", ["open-em.mmap"]),
        ExpressionClass("pog", ["pog-em.mmap"], "JawForward"),
        ExpressionClass("nwigleft", ["nwigleft-em.mmap"]),
        ExpressionClass("nwigright", ["nwigright-em.mmap"]),
        ExpressionClass("showteth", ["showteth-em.mmap"]),
    ]
    return cfg
