"""Read the shared SFM dataset through any of the nine confirmed input variants."""
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

from .sfm_preprocessing import select_variant


class DefaultSFMDataset(Dataset):
    """Current RGB uint8 views, normalized inputs, H=5 targets and validity mask.

    Images are [head,wrist] x C x H x W. The vision encoder's tensor scaling and
    backbone normalization remain the model's responsibility. F/T is already
    standardized exactly once; the old GRU input normalizer must be disabled.
    """
    def __init__(self, root, variant, *, allow_smoke=False, max_open_files=8):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if not self.manifest.get("complete") or (self.manifest.get("smoke_test") and not allow_smoke):
            raise ValueError("Require a completed full-cohort dataset; smoke files need explicit allow_smoke=True")
        contract_bytes = (self.root / "data_contract.json").read_bytes()
        if hashlib.sha256(contract_bytes).hexdigest() != self.manifest["data_contract_sha256"]:
            raise ValueError("Data contract hash differs from completed manifest")
        self.contract = json.loads(contract_bytes)
        if hashlib.sha256((self.root / "normalization.json").read_bytes()).hexdigest() != self.manifest["normalization_sha256"]:
            raise ValueError("Normalization hash differs from completed manifest")
        self.variant = self.contract["variants"][variant]
        self.shards = self.manifest["shards"]
        self.starts = np.r_[0, np.cumsum([row["frames"] for row in self.shards])].tolist()
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = max_open_files
        self.handles = OrderedDict()
        self.pid = os.getpid()

    def __len__(self):
        return self.starts[-1]

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __getstate__(self):
        return {**self.__dict__, "handles": OrderedDict()}

    def _file(self, episode):
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        if episode not in self.handles:
            f = h5py.File(self.root / self.shards[episode]["file"], "r")
            if not f.attrs.get("normalized") or f.attrs["normalization_sha256"] != self.manifest["normalization_sha256"]:
                f.close()
                raise ValueError("Episode is not normalized with the shared coefficients")
            self.handles[episode] = f
            while len(self.handles) > self.max_open_files:
                _, oldest = self.handles.popitem(last=False)
                oldest.close()
        self.handles.move_to_end(episode)
        return self.handles[episode]

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.starts, index) - 1
        step = index - self.starts[episode]
        f = self._file(episode)
        ft = f["observations/ft_history"][step]
        result = {
            "instruction": self.shards[episode]["instruction"],
            "images": np.ascontiguousarray(np.stack([f["observations/" + key][step]
                for key in ("image_head", "image_wrist_right")]).transpose(0, 3, 1, 2)),
            "state": select_variant(f["observations/state"][step], ft, self.variant),
            "actions": f["actions"][step], "action_mask": f["action_mask"][step],
            "episode_index": episode, "source_index": int(f["source_indices"][step]),
        }
        if self.variant["ft_method"] != "state_embedding":
            result["ft_history"] = ft
        return result
