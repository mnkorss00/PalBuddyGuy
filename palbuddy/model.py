"""The expression network.

Architecture and checkpoint format are identical to the original script, so
existing buddyguy.pt files load unchanged. Differences:
  * a real nn.Module: dropout is controlled by train()/eval() instead of
    monkey-patching the dropout functions to identity for inference,
  * flatten(1) instead of reshaping with the global batch_size (which broke
    whenever the last batch or an inference batch had a different size),
  * runs on CPU when CUDA isn't available,
  * can take only the eye (64 ch) or only the face (64 ch) features, for
    setups with a single tracker. The checkpoint records which,
  * an optional "lite" architecture (same layout, fewer channels, ~8x fewer
    weights, ~5x less compute). Whether it tracks as well as "standard"
    depends on the data; compare the validation accuracy after training.
"""

import logging
import os

import torch
from torch import nn

log = logging.getLogger(__name__)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# arch -> (conv channels, hidden units). "standard" is the original network.
ARCHS = {"standard": (256, 1024), "lite": (128, 256)}


class BuddyNet(nn.Module):
    def __init__(self, num_outputs, input_mode="both", arch="standard"):
        super().__init__()
        self.num_outputs = num_outputs
        self.input_mode = input_mode
        self.arch = arch
        channels, hidden = ARCHS[arch]
        in_channels = 128 if input_mode == "both" else 64
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.linear1 = nn.Linear(channels * 100, hidden)
        self.linear2 = nn.Linear(hidden, num_outputs)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.dropout_in = nn.Dropout(p=0.7)

    def forward(self, x):
        x = self.act(self.conv1(self.dropout_in(x)))
        x = x + self.act(self.conv2(self.dropout(x)))
        x = self.act(self.linear1(self.dropout(x.flatten(1))))
        return self.act(self.linear2(self.dropout(x)))

    # ------------------------------------------------------------ checkpoints
    def save(self, path, class_names=None):
        tmp = path + ".tmp"
        data = {name: getattr(self, name).state_dict() for name in ("conv1", "conv2", "linear1", "linear2")}
        if class_names is not None:
            data["class_names"] = list(class_names)
        data["input_mode"] = self.input_mode
        data["arch"] = self.arch
        torch.save(data, tmp)
        os.replace(tmp, path)  # never leave a half-written checkpoint behind

    @classmethod
    def load(cls, path, device=None, expected_outputs=None, expected_mode=None):
        data = torch.load(path, map_location="cpu")
        n = data["linear2"]["weight"].shape[0]
        if expected_outputs is not None and n != expected_outputs:
            raise ValueError(
                "Model in %s has %d outputs but %d classes are configured. Retrain or fix the class list."
                % (path, n, expected_outputs))
        in_channels = data["conv1"]["weight"].shape[1]
        # checkpoints from the original script have no "input_mode" and are always "both"
        mode = data.get("input_mode") or ("both" if in_channels == 128 else expected_mode or "face")
        if expected_mode is not None and mode != expected_mode:
            raise ValueError(
                "Model in %s was trained for input '%s' but input mode '%s' is selected. "
                "Switch the input mode back or retrain." % (path, mode, expected_mode))
        channels = data["conv1"]["weight"].shape[0]
        arch = data.get("arch") or next((a for a, (c, _) in ARCHS.items() if c == channels), "standard")
        model = cls(n, mode, arch)
        for name in ("conv1", "conv2", "linear1", "linear2"):
            getattr(model, name).load_state_dict(data[name])
        model.class_names = data.get("class_names")
        if device is not None:
            model.to(device)
        return model
