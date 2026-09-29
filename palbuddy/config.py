"""Persistent settings (config.json) shared by the GUI and the CLI.

Replaces the constants that used to be hand-edited at the top of script.py
(DATASET_FOLDER, order, to_replace, max_power_array).
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .params import shape_id

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

    # Inference
    smoothing: float = 0.0  # 0 = off, otherwise EMA factor in (0, 1)
    max_send_rate: float = 120.0  # Hz cap for VRCFT updates
    infer_device: str = "auto"  # "auto" (GPU if available), "cpu" or "gpu"
    infer_threads: int = 1  # CPU threads for inference; more = lower latency but more total CPU
    infer_int8: bool = True  # int8-quantize the linear layers when inferring on the CPU
    max_infer_rate: float = 0.0  # Hz cap for running the network, 0 = every new frame

    # GUI
    language: str = "auto"  # "auto", "en" or "ko"
    show_preview: bool = True

    classes: List[ExpressionClass] = field(default_factory=list)

    # ---------------------------------------------------------------- helpers
    @property
    def num_classes(self):
        return len(self.classes)

    def targets(self):
        """[(class_index, shape_index, max_power)] for every class that drives a shape."""
        out = []
        for i, c in enumerate(self.classes):
            if c.target not in (None, ""):
                out.append((i, shape_id(c.target), c.max_power))
        return out

    def dataset_path(self, filename):
        return filename if os.path.isabs(filename) else os.path.join(self.dataset_folder, filename)

    def validate(self):
        problems = []
        if self.input_mode not in INPUT_MODES:
            problems.append("input_mode must be one of %s." % ", ".join(INPUT_MODES))
        if not self.classes:
            problems.append("No expression classes are configured.")
        for c in self.classes:
            if c.target not in (None, ""):
                try:
                    shape_id(c.target)
                except KeyError:
                    problems.append("Class '%s' has an unknown target shape '%s'." % (c.name, c.target))
            if c.max_power <= 0:
                problems.append("Class '%s' needs max_power > 0." % c.name)
        return problems

    # ---------------------------------------------------------------- io
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        classes = [ExpressionClass(**c) for c in data.pop("classes", [])]
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.classes = classes
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
