"""Read-only adapter for preprocessed real-robot HDF5 bundles.

Preserves per-anchor arm chunks, symlog, standardized states, RGB order and tail
masks. Optionally relabels sparse gripper commands as latched binary targets.
Adapted from the local RepresentationFM real_robot_data reader.
No dependency on that project or the remote preprocessing server at runtime.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .ft_history import resample_ft_history, resampling_contract


INSTRUCTION_KEYS = ("task_instruction", "instruction", "language_instruction")


def read_ft_history(handle, frames, history_length, input_dim=6, resampling=None):
    key = "observations/ft_history"
    if key not in handle:
        raise ValueError("GRU-conditioned policy requires observations/ft_history")
    values = handle[key][:]
    source_length = history_length if resampling is None else resampling["source_length"]
    if (values.shape != (frames, source_length, input_dim) or values.dtype.kind != "f"
            or not np.isfinite(values).all()):
        raise ValueError(f"Expected finite floating FT history [T,{history_length},{input_dim}]")
    values = resample_ft_history(values, resampling)
    if values.shape != (frames, history_length, input_dim) or not np.isfinite(values).all():
        raise ValueError("Resampled FT history differs from the configured model input")
    return values


def require_raw_ft_history(statistics):
    if statistics.get("observations", {}).get("ft_history", {}).get("method") != "none":
        raise ValueError("FT history must be raw physical measurements (method=none); the model normalizes it")


def fit_ft_history_statistics(data, history_length, input_dim=6):
    """Fit only the supplied training windows, including valid zero readings."""
    root = Path(data["dataset_dir"]).expanduser().resolve()
    require_raw_ft_history(json.loads((root / "normalization.json").read_text()))
    count, mean, m2 = 0, np.zeros(input_dim), np.zeros(input_dim)
    for path in sorted(root.glob("*.hdf5")):
        with h5py.File(path, "r") as handle:
            values = read_ft_history(handle, len(handle["actions"]), history_length, input_dim,
                                     resampling=resampling_contract(data.get("ft_history_resampling")))
        values = values.reshape(-1, input_dim).astype(np.float64)
        n, batch_mean = len(values), values.mean(axis=0)
        delta = batch_mean - mean
        m2 += np.square(values - batch_mean).sum(axis=0) + delta ** 2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    if not count:
        raise ValueError("No training FT-history windows found")
    return {"mean": mean.tolist(), "std": np.sqrt(m2 / count).tolist()}


def read_task_instructions(handle, length, default=None):
    """Read explicit language labels, filling only absent/blank instructions."""
    def decode(value):
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8")
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("Task instructions must be UTF-8 strings")
        return value.strip()

    value, source = None, "default_instruction"
    for key in INSTRUCTION_KEYS:
        for name in (key, f"metadata/{key}", f"observations/{key}"):
            if name in handle:
                value, source = handle[name][()], name
                break
        if value is not None:
            break
        if key in handle.attrs:
            value, source = handle.attrs[key], f"attribute:{key}"
            break
    if value is None:
        for name in ("metadata/recording_metadata", "metadata/collection_config"):
            if name not in handle:
                continue
            metadata = json.loads(handle[name][()])
            for key in INSTRUCTION_KEYS:
                if key in metadata:
                    value, source = metadata[key], f"{name}:{key}"
                    break
            if value is not None:
                break
    if value is None or np.ndim(value) == 0:
        values = [decode(value)] * length
    else:
        if np.shape(value) != (length,):
            raise ValueError("Expected scalar or per-frame task instructions")
        values = [decode(item) for item in value]
    fallback = decode(default)
    sources = [source if item else "default_instruction" for item in values]
    values = [item or fallback for item in values]
    if any(not item for item in values):
        raise ValueError("Missing task instruction; configure default_instruction")
    return values, sources


@dataclass(frozen=True)
class GripperRelabelConfig:
    method: str = "latched_command"
    close_threshold: float = -0.05
    open_threshold: float = 0.05
    initial_pose_threshold: float = 0.5

    def __post_init__(self):
        if self.method != "latched_command":
            raise ValueError("gripper_relabel.method must be latched_command")
        values = (self.close_threshold, self.open_threshold, self.initial_pose_threshold)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError("Gripper thresholds must be finite numbers")
        if not np.isfinite(values).all() or not -1 <= self.close_threshold < 0 < self.open_threshold <= 1:
            raise ValueError("Require finite thresholds: -1 <= close < 0 < open <= 1")
        if not 0 < self.initial_pose_threshold < 1:
            raise ValueError("initial_pose_threshold must be inside (0, 1)")

    def contract(self):
        return {"version": 1, **asdict(self), "source": "source_actions[:, 6] before symlog",
                "closed_target": -1.0, "open_target": 1.0,
                "initial_state": "first retained measured gripper_pose >= initial_pose_threshold means open",
                "timing": "apply command at the same timestep, then hold within the episode; no lookahead",
                "chunk_target": "episode target at anchor + offset; zero only for padding",
                "proprioception": "unchanged measured values"}


def latch_gripper_commands(commands, initial_pose, config):
    """Make one causal target sequence per episode, before sampling chunks."""
    commands = np.asarray(commands, dtype=np.float32)
    if commands.ndim != 1 or not len(commands) or not np.isfinite(commands).all():
        raise ValueError("Expected a nonempty finite gripper command vector")
    if np.any(np.abs(commands) > 1.000001):
        raise ValueError("Expected signed gripper commands in [-1, 1]")
    if not np.isfinite(initial_pose) or not -1e-6 <= initial_pose <= 1 + 1e-6:
        raise ValueError("Expected initial measured gripper opening in [0, 1]")
    target = 1.0 if initial_pose >= config.initial_pose_threshold else -1.0
    labels = np.empty_like(commands)
    for step, command in enumerate(commands):
        # Compare in the stored command dtype, including exact boundaries.
        if command <= np.float32(config.close_threshold):
            target = -1.0
        elif command >= np.float32(config.open_threshold):
            target = 1.0
        labels[step] = target
    return labels


class RealRobotDataset(Dataset):
    def __init__(self, config):
        self.config = deepcopy(config)
        settings = config.get("gripper_relabel")
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("gripper_relabel must be an object or null")
        try:
            self.gripper_relabel = None if settings is None else GripperRelabelConfig(**settings)
        except TypeError as error:
            raise ValueError("Unknown gripper_relabel setting") from error
        self.root = Path(config["dataset_dir"]).expanduser().resolve()
        if (self.root / "INCOMPLETE").exists() or (self.root.parent / "INCOMPLETE").exists():
            raise ValueError("Dataset preprocessing is incomplete")
        self.paths = sorted(self.root.glob("*.hdf5"))
        self.use_language = bool(config.get("use_language", False))
        self.use_ft_history = config.get("use_ft_history", False)
        if not isinstance(self.use_ft_history, bool):
            raise ValueError("use_ft_history must be a boolean")
        self.ft_history_length = config.get("ft_history_length", 50)
        self.ft_history_dim = config.get("ft_history_dim", 6)
        self.ft_history_resampling = resampling_contract(config.get("ft_history_resampling")) if self.use_ft_history else None
        if self.use_ft_history and (isinstance(self.ft_history_length, bool)
                                   or not isinstance(self.ft_history_length, int) or self.ft_history_length < 1
                                   or self.ft_history_dim != 6):
            raise ValueError("Expected positive FT-history length and six FT channels")
        self.instructions, self.instruction_sources = [], {}
        self.instruction_file = self.root / "instructions.json" if self.use_language else None
        instruction_entries = None
        if self.instruction_file is not None and self.instruction_file.exists():
            instruction_entries = json.loads(self.instruction_file.read_text())["episodes"]
            if sorted(instruction_entries) != [path.name for path in self.paths]:
                raise ValueError("Instruction manifest does not match episode files")
        self.camera_keys = tuple(config["camera_keys"])
        self.mode, self.horizon = config["action_mode"], int(config["horizon"])
        if self.mode not in ("relative", "delta") or not self.camera_keys:
            raise ValueError("Expected relative/delta and current RGB camera keys")
        if any(not key.startswith("observations/image_") for key in self.camera_keys):
            raise ValueError("Only current camera observations may enter inference")
        self.normalization = json.loads((self.root / "normalization.json").read_text())
        if self.use_ft_history:
            require_raw_ft_history(self.normalization)
        manifest = json.loads((self.root / "manifest.json").read_text())
        retained = [entry for entry in manifest if not entry.get("skipped", False)]
        entries = {Path(entry["output"]).name: entry for entry in retained}
        if not self.paths or len(entries) != len(retained) or sorted(entries) != [p.name for p in self.paths]:
            raise ValueError("Episode files do not match the manifest")
        if len(self.paths) != config.get("expected_episodes", len(self.paths)):
            raise ValueError("Unexpected episode count")
        schema = self.normalization["schema"]
        if schema["action"]["mode"] != self.mode or schema["action_shape"] != [self.horizon, 7]:
            raise ValueError("Use the original stored action mode, horizon and dimensions")
        if self.normalization["observations"]["state"]["method"] != "standard":
            raise ValueError("Expected already standardized states")
        temperature = self.normalization["actions"]["temperature"]
        if (self.normalization["actions"]["method"] not in ("symlog", "symlog_minmax")
                or not np.isfinite(temperature) or temperature <= 0):
            raise ValueError("Expected already symlog-transformed actions")
        if self.normalization["actions"]["method"] == "symlog_minmax":
            for key in ("offset", "scale"):
                coefficients = np.asarray(self.normalization["actions"].get(key, []))
                if coefficients.shape != (7,) or not np.isfinite(coefficients).all():
                    raise ValueError("Expected finite 7D action normalization coefficients")
                if key == "scale" and np.any(coefficients <= 0):
                    raise ValueError("Action normalization scales must be positive")
        self.windows, self.numeric, self.images = [], [], []
        self.gripper_targets = []
        self.gripper_label_counts = {key: 0 for key in (
            "source_close", "source_hold", "source_open", "timesteps_closed", "timesteps_open",
            "valid_chunk_closed", "valid_chunk_open")}
        self.handles, self.action_contract = {}, None
        self.valid_action_count = 0
        self.preload = bool(config.get("preload_images", True))
        size = int(config["image_size"])
        for episode, path in enumerate(self.paths):
            with h5py.File(path, "r") as f:
                if f.attrs.get("schema_version") != 2 or f.attrs.get("image_color") != "RGB":
                    raise ValueError(f"Unsupported schema or image color: {path.name}")
                if f.attrs.get("action_transform") != self.mode:
                    raise ValueError("Episode action mode changed")
                if json.loads(f["metadata/normalization"][()]) != self.normalization:
                    raise ValueError("Episode normalization differs")
                contract = json.loads(f["metadata/action_contract"][()])
                if self.action_contract is None:
                    self.action_contract = contract
                if contract != self.action_contract:
                    raise ValueError("Episode action contract differs")
                actions, state, mask = f["actions"][:], f["observations/state"][:], f["action_mask"][:]
                length = len(actions)
                if length != entries[path.name]["frames"] or length < 1:
                    raise ValueError("Manifest length differs")
                if self.use_language:
                    if instruction_entries is None:
                        instructions, sources = read_task_instructions(f, length, config.get("default_instruction"))
                    else:
                        entry = instruction_entries[path.name]
                        instructions, sources = entry["instructions"], entry["sources"]
                        if (len(instructions) != length or len(sources) != length
                                or any(not isinstance(item, str) or not item.strip() for item in instructions)):
                            raise ValueError("Invalid prepared task instructions")
                    self.instructions.append(instructions)
                    for source in sources:
                        self.instruction_sources[source] = self.instruction_sources.get(source, 0) + 1
                if actions.shape != (length, self.horizon, 7) or state.shape != (length, 19):
                    raise ValueError("Expected action [T,H,7] and state [T,19]")
                expected = np.arange(self.horizon)[None, :] < length - np.arange(length)[:, None]
                if mask.dtype != np.bool_ or not np.array_equal(mask, expected):
                    raise ValueError("Invalid episode tail mask")
                if not np.isfinite(actions).all() or not np.isfinite(state).all():
                    raise ValueError("Nonfinite state or action")
                if np.any(np.abs(actions[mask]) > 1.00001) or np.any(actions[~mask] != 0):
                    raise ValueError("Actions must fit tanh output and use normalized zero padding")
                if self.gripper_relabel is not None:
                    self._relabel_episode(f, actions, state, mask)
                for key in self.camera_keys:
                    if f[key].shape != (length, size, size, 3) or f[key].dtype != np.uint8:
                        raise ValueError("Expected fixed-size uint8 RGB cameras")
                numeric = {"state": state.astype(np.float32), "actions": actions.astype(np.float32), "action_mask": mask}
                if self.use_ft_history:
                    numeric["ft_history"] = read_ft_history(f, length, self.ft_history_length, self.ft_history_dim,
                                                            self.ft_history_resampling)
                self.numeric.append(numeric)
                if self.preload:
                    self.images.append([f[key][:] for key in self.camera_keys])
                self.windows.extend((episode, t) for t in range(length))
                self.valid_action_count += int(mask.sum())
        if len(self) != config.get("expected_frames", len(self)):
            raise ValueError("Unexpected retained frame count")
        if self.gripper_relabel is not None:
            self.stored_action_contract, self.stored_normalization = self.action_contract, self.normalization
            self.action_contract = {**deepcopy(self.action_contract),
                                    "gripper": "latched binary target: -1 closed, +1 open",
                                    "gripper_relabel": self.gripper_relabel.contract()}
            self.normalization = deepcopy(self.normalization)
            self.normalization["schema"]["action"].update(
                gripper=self.action_contract["gripper"], gripper_relabel=self.gripper_relabel.contract())
            valid = np.concatenate([row["actions"][..., 6][row["action_mask"]] for row in self.numeric]).astype(np.float64)
            # Update descriptive statistics only. Symlog coefficients are unchanged.
            for key, value in {"mean": valid.mean(), "std": valid.std(), "min": valid.min(), "max": valid.max()}.items():
                if key in self.normalization["actions"]:
                    self.normalization["actions"][key][6] = float(value)

    def _relabel_episode(self, handle, actions, state, mask):
        contract, schema = self.action_contract, self.normalization["schema"]
        if contract.get("gripper_relabel") is not None or "gripper_targets" in handle:
            raise ValueError("Gripper targets are already relabeled; use the original bundle")
        if contract.get("sides") != ["right"] or contract.get("relative_gripper_action_mode") != "absolute":
            raise ValueError("Gripper relabeling requires single-right-arm absolute signed commands")
        if "source_actions" not in handle or handle["source_actions"].shape != (len(actions), 7):
            raise ValueError("Expected source_actions [T,7] for gripper relabeling")
        try:
            column = schema["state_columns"]["state"].index("right/gripper_pose/pos")
            stats = self.normalization["observations"]["state"]
            scale, offset = stats["scale"][column], stats["offset"][column]
            action_stats = self.normalization["actions"]
            if (not np.isfinite([scale, offset]).all() or scale <= 0
                    or action_stats["offset"][6] != 0 or action_stats["scale"][6] != 1):
                raise ValueError("Invalid gripper normalization coefficients")
            initial = float(state[0, column]) * scale + offset
        except (KeyError, IndexError) as error:
            raise ValueError("Gripper relabeling requires measured gripper column and normalization coefficients") from error
        commands = handle["source_actions"][:, 6].astype(np.float32)
        labels = latch_gripper_commands(commands, initial, self.gripper_relabel)
        future = np.arange(len(actions))[:, None] + np.arange(self.horizon)[None, :]
        temperature = action_stats["temperature"]
        encoded = np.sign(commands) * np.log1p(np.abs(commands) * temperature) / np.log1p(temperature)
        if not np.allclose(actions[..., 6][mask], encoded[future[mask]], atol=1e-6, rtol=1e-6):
            raise ValueError("Source gripper commands differ from stored chunk labels")
        # symlog(+/-1) == +/-1; keep padded entries at normalized zero.
        actions[..., 6] = 0
        actions[..., 6][mask] = labels[future[mask]]
        self.gripper_targets.append(labels)
        close = commands <= np.float32(self.gripper_relabel.close_threshold)
        opened = commands >= np.float32(self.gripper_relabel.open_threshold)
        counts = {"source_close": close.sum(), "source_open": opened.sum(),
                  "source_hold": (~(close | opened)).sum(), "timesteps_closed": (labels == -1).sum(),
                  "timesteps_open": (labels == 1).sum(),
                  "valid_chunk_closed": (actions[..., 6][mask] == -1).sum(),
                  "valid_chunk_open": (actions[..., 6][mask] == 1).sum()}
        for key, value in counts.items():
            self.gripper_label_counts[key] += int(value)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, step = self.windows[index]
        sample = {key: torch.from_numpy(array[step].copy()) for key, array in self.numeric[episode].items()}
        if self.preload:
            images = np.stack([camera[step] for camera in self.images[episode]])
        else:
            path = self.paths[episode]
            if path not in self.handles:
                self.handles[path] = h5py.File(path, "r")
            images = np.stack([self.handles[path][key][step] for key in self.camera_keys])
        sample["images"] = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))
        if self.use_language:
            sample["instruction"] = self.instructions[episode][step]
        return sample

    def signature(self):
        result = {"files": [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in self.paths],
                "metadata_sha256": {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                                    for name in ("normalization.json", "manifest.json")},
                "camera_keys": list(self.camera_keys), "state_key": "observations/state", "mode": self.mode}
        if self.gripper_relabel is not None:
            result["gripper_relabel"] = self.gripper_relabel.contract()
        if self.use_ft_history:
            result["ft_history"] = {"key": "observations/ft_history", "history_length": self.ft_history_length,
                                    "input_dim": self.ft_history_dim, "normalization": "raw physical measurements"}
            if self.ft_history_resampling is not None:
                result["ft_history"]["resampling"] = self.ft_history_resampling
        if self.use_language:
            result["instructions_sha256"] = hashlib.sha256(
                json.dumps(self.instructions, ensure_ascii=False).encode("utf-8")).hexdigest()
            if self.instruction_file.exists():
                result["metadata_sha256"]["instructions.json"] = hashlib.sha256(self.instruction_file.read_bytes()).hexdigest()
        return result

    def metadata(self):
        result = {"episodes": len(self.paths), "frames": len(self), "valid_action_entries": self.valid_action_count,
                "action_contract": self.action_contract, "normalization": self.normalization,
                "split": "all training demonstrations; no held-out data", "use_language": False,
                "input": "two current RGB views and standardized current 19D state",
                "unused": ["next_observations", "ft_history", "source_actions", "future states"],
                "sample_output_units": "stored symlog actions; invert symlog then apply the saved action contract"}
        if self.gripper_relabel is not None:
            result.update(stored_action_contract=self.stored_action_contract, stored_normalization=self.stored_normalization,
                          gripper_label_counts=self.gripper_label_counts,
                          label_transform_inputs=["source_actions[:, 6]", "first retained measured gripper opening"],
                          unused=["next_observations", "ft_history", "future states"],
                          sample_output_units="symlog arm actions, latched gripper -1 closed / +1 open; zero padding")
        if self.normalization["actions"]["method"] == "symlog_minmax":
            result["sample_output_units"] = "stored symlog_minmax actions; inverse symlog, then multiply scale and add offset"
        if self.use_language:
            result.update(use_language=True, instruction_sources=self.instruction_sources,
                          default_instruction=self.config.get("default_instruction"),
                          instructions=sorted({item for episode in self.instructions for item in episode}),
                          input="two current RGB views, standardized current 19D state, task instruction")
        if self.use_ft_history:
            result.update(use_ft_history=True, ft_history_shape=[self.ft_history_length, self.ft_history_dim],
                          ft_history_order="stored oldest to newest; newest last",
                          ft_history_zero_handling="all samples retained, including entirely zero windows",
                          ft_history_normalization="training mean/std stored and applied inside model",
                          input=result["input"] + ", raw FT history encoded by GRU")
            result["unused"].remove("ft_history")
            if self.ft_history_resampling is not None:
                result["ft_history_resampling"] = self.ft_history_resampling
        return result

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handles"] = {}
        return state


def normalize_images(images, mean, std, color_jitter=None):
    """Vectorized, resize-free RGB normalization; state/actions are untouched."""
    if images.dtype != torch.uint8 or images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("Expected uint8 RGB [B,V,3,H,W]")
    values = images.float() / 255
    if color_jitter is not None:
        from torchvision.transforms import ColorJitter

        settings = dict(color_jitter)
        top, bottom, left, right = settings.pop("padding_tblr", [0, 0, 0, 0])
        height, width = values.shape[-2:]
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (top, bottom, left, right))
                or top + bottom >= height or left + right >= width):
            raise ValueError("Invalid color jitter padding")
        jitter = ColorJitter(**settings)
        # Each observation draws its own photometric transform. Both current
        # cameras share it; black letterbox padding is excluded and preserved.
        content = values[..., top:height-bottom, left:width-right]
        values[..., top:height-bottom, left:width-right] = torch.stack([jitter(sample) for sample in content])
    mean = values.new_tensor(mean).view(1, 1, 3, 1, 1)
    std = values.new_tensor(std).view(1, 1, 3, 1, 1)
    return (values - mean) / std


def inverse_symlog(actions, temperature=20.0):
    return actions.sign() * torch.expm1(actions.abs() * np.log1p(temperature)) / temperature


def inverse_normalized_actions(actions, statistics):
    """Restore physical units with the exact coefficients saved in the bundle."""
    if statistics["method"] not in ("symlog", "symlog_minmax"):
        raise ValueError("Unsupported action normalization")
    physical = inverse_symlog(actions, statistics["temperature"])
    if statistics["method"] == "symlog_minmax":
        physical = physical * physical.new_tensor(statistics["scale"]) + physical.new_tensor(statistics["offset"])
    return physical
