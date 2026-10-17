from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from .config import PipelineConfig
from .data_discovery import META_FEATURE_COLS, WEIGHT_COLS


def compute_slice_indices(
    depth: int, seq_len: int, mode: str, rng: np.random.Generator
) -> np.ndarray:
    if depth <= 0:
        raise ValueError(f"Invalid MRI volume depth: {depth}")
    start_d = int(depth * 0.08) if depth >= 14 else 0
    end_d = max(start_d + 1, int(depth * 0.92)) if depth >= 14 else depth

    if mode == "stratified_jitter":
        edges = np.linspace(float(start_d), float(end_d), seq_len + 1)
        lows = np.floor(edges[:-1]).astype(np.int64)
        highs = np.maximum(lows + 1, np.floor(edges[1:]).astype(np.int64))
        highs = np.minimum(highs, depth)
        lows = np.minimum(lows, highs - 1)
        return rng.integers(low=lows, high=highs, size=seq_len)

    return np.linspace(start_d, max(start_d, end_d - 1), seq_len).round().astype(np.int64)


def pad_and_resize_25d_slices(
    slices_25d: np.ndarray, target_size: Tuple[int, int], pad_to_square: bool = True
) -> torch.Tensor:
    """
    slices_25d: shape (S, 3, H, W) containing [z-1, z, z+1] adjacent slices.
    Pads symmetrically to square, resizes to target_size, and standardizes with
    zero-centered mean=0.45, std=0.225 for pretrained timm backbones.
    """
    t = torch.from_numpy(slices_25d)  # (S, 3, H, W)
    _, _, h, w = t.shape
    if pad_to_square and h != w:
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        t = F.pad(
            t,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            mode="constant",
            value=0.0,
        )
    if t.shape[-2:] != target_size:
        t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)

    # Standardize [0, 1] -> zero-centered distribution expected by pretrained backbones
    t = (t - 0.45) / 0.225
    return t.to(torch.float16)


class StudyBagDataset(Dataset):
    """
    2.5D Multi-Series Study-Level Dataset:
    - For each selected series and each sampled slice index `z`, extracts the 2.5D
      triplet `[z-1, z, z+1]` as a 3-channel image `(S, 3, H, W)`.
    - Computes the normalized anatomical slice position `z_pos in [-1.0, +1.0]` for every slice
      so class-specific attention knows whether a slice is medial, central (notch), or lateral.
    """

    def __init__(
        self,
        studies_df: pd.DataFrame,
        series_df: pd.DataFrame,
        path_lookup: Dict[str, str],
        cfg: PipelineConfig,
        is_train: bool = True,
    ):
        self.is_train = is_train
        self.max_series = cfg.max_series_per_study if is_train else max(cfg.max_series_per_study, 5)
        self.seq_len = cfg.seq_len
        self.image_size = cfg.image_size
        self.pad_to_square = cfg.pad_to_square
        self.sampling_mode = cfg.train_sampling_mode if is_train else "uniform"
        self.seed = cfg.seed
        self.epoch = 0

        self.study_uids = np.array(studies_df["StudyInstanceUID"].astype(str).tolist(), dtype="S128")

        if all(col in studies_df.columns for col in cfg.labels):
            self.targets = studies_df[cfg.labels].values.astype(np.float32)
        else:
            self.targets = np.zeros((len(studies_df), len(cfg.labels)), dtype=np.float32)

        if all(col in studies_df.columns for col in WEIGHT_COLS):
            self.conf_weights = studies_df[WEIGHT_COLS].values.astype(np.float32)
        else:
            self.conf_weights = (~np.isnan(self.targets)).astype(np.float32)

        s_df = series_df.copy()
        s_df["resolved_path"] = [path_lookup.get(p, p) for p in s_df["npz_path"].tolist()]
        self.all_series_paths = np.array(s_df["resolved_path"].tolist(), dtype="S512")
        self.all_series_metas = s_df[META_FEATURE_COLS].values.astype(np.float32)

        uid_to_indices: Dict[str, List[int]] = {}
        for idx, st_uid in enumerate(s_df["StudyInstanceUID"].astype(str).tolist()):
            uid_to_indices.setdefault(st_uid, []).append(idx)

        self.study_series_indices: List[np.ndarray] = [
            np.array(uid_to_indices.get(u.decode("utf-8"), []), dtype=np.int32)
            for u in self.study_uids
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.study_uids)

    def _select_complementary_series(
        self, candidate_indices: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if len(candidate_indices) <= self.max_series:
            return candidate_indices

        metas = self.all_series_metas[candidate_indices]
        selected: List[int] = []
        remaining = list(range(len(candidate_indices)))
        if self.is_train:
            rng.shuffle(remaining)
        else:
            remaining.sort(key=lambda i: float(metas[i, 0] + metas[i, 1]), reverse=True)

        for plane_col in (2, 3, 4):
            if len(selected) >= self.max_series:
                break
            for r_idx in remaining:
                if r_idx not in selected and metas[r_idx, plane_col] > 0.5:
                    selected.append(r_idx)
                    break

        for r_idx in remaining:
            if len(selected) >= self.max_series:
                break
            if r_idx not in selected:
                selected.append(r_idx)

        return candidate_indices[np.array(selected[: self.max_series], dtype=np.int32)]

    def _load_one_series_25d(
        self, ser_idx: int, rng: np.random.Generator
    ) -> Tuple[torch.Tensor, np.ndarray]:
        path = self.all_series_paths[ser_idx].decode("utf-8")
        try:
            if path.endswith(".npy"):
                mm = np.load(path, mmap_mode="r")
                depth = int(mm.shape[0])
                center_idx = compute_slice_indices(depth, self.seq_len, self.sampling_mode, rng)
                # Build 2.5D triplets [z-1, z, z+1]
                prev_idx = np.clip(center_idx - 1, 0, depth - 1)
                next_idx = np.clip(center_idx + 1, 0, depth - 1)
                triplet_idx = np.stack([prev_idx, center_idx, next_idx], axis=1)  # (S, 3)
                slices_25d = np.array(mm[triplet_idx.reshape(-1)], dtype=np.float32, copy=True)
                slices_25d = slices_25d.reshape(self.seq_len, 3, mm.shape[1], mm.shape[2])
                del mm
            else:
                with np.load(path) as data:
                    key = "volume" if "volume" in data else ("arr_0" if "arr_0" in data else data.files[0])
                    raw_vol = data[key]
                    if raw_vol.ndim == 4:
                        raw_vol = raw_vol[..., 0] if raw_vol.shape[-1] <= 3 else raw_vol[0]
                    depth = int(raw_vol.shape[0])
                    center_idx = compute_slice_indices(depth, self.seq_len, self.sampling_mode, rng)
                    prev_idx = np.clip(center_idx - 1, 0, depth - 1)
                    next_idx = np.clip(center_idx + 1, 0, depth - 1)
                    triplet_idx = np.stack([prev_idx, center_idx, next_idx], axis=1)
                    slices_25d = np.array(raw_vol[triplet_idx.reshape(-1)], dtype=np.float32, copy=True)
                    slices_25d = slices_25d.reshape(self.seq_len, 3, raw_vol.shape[1], raw_vol.shape[2])
                    del raw_vol
        except Exception:
            depth = self.seq_len
            center_idx = np.arange(self.seq_len)
            slices_25d = np.zeros((self.seq_len, 3, self.image_size[0], self.image_size[1]), dtype=np.float32)

        p_low, p_high = np.percentile(slices_25d, [0.5, 99.5])
        if p_high > p_low:
            slices_25d = np.clip((slices_25d - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            max_v = float(slices_25d.max())
            if max_v > 0:
                slices_25d = slices_25d / max_v

        z_pos = (2.0 * (center_idx.astype(np.float32) / max(float(depth - 1), 1.0))) - 1.0
        tensor_25d = pad_and_resize_25d_slices(slices_25d, self.image_size, pad_to_square=self.pad_to_square)
        return tensor_25d, z_pos

    def __getitem__(self, idx: int):
        study_uid = self.study_uids[idx].decode("utf-8")
        rng = np.random.default_rng(self.seed + self.epoch * 100_003 + idx)

        cand_indices = self.study_series_indices[idx]
        chosen = self._select_complementary_series(cand_indices, rng)

        k_max = self.max_series
        h, w = self.image_size
        bag_slices = torch.zeros((k_max, self.seq_len, 3, h, w), dtype=torch.float16)
        # 6 metadata features per slice token: [fluid, fat, sag, cor, ax, z_pos]
        bag_metas = torch.zeros((k_max, self.seq_len, len(META_FEATURE_COLS) + 1), dtype=torch.float32)
        series_mask = torch.zeros((k_max,), dtype=torch.bool)

        for slot, ser_idx in enumerate(chosen[:k_max]):
            t_25d, z_pos = self._load_one_series_25d(int(ser_idx), rng)
            bag_slices[slot] = t_25d
            ser_meta = self.all_series_metas[int(ser_idx)]  # (5,)
            # Broadcast (5,) across S slices and append z_pos (S, 1) -> (S, 6)
            repeated_meta = np.repeat(ser_meta[None, :], self.seq_len, axis=0)
            slice_meta = np.concatenate([repeated_meta, z_pos[:, None]], axis=1).astype(np.float32)
            bag_metas[slot] = torch.from_numpy(slice_meta)
            series_mask[slot] = True

        if not series_mask.any():
            series_mask[0] = True

        target_t = torch.from_numpy(self.targets[idx].copy())
        weight_t = torch.from_numpy(self.conf_weights[idx].copy())
        return bag_slices, bag_metas, series_mask, target_t, weight_t, study_uid


def study_bag_collate_fn(batch):
    slices = torch.stack([b[0] for b in batch], dim=0)       # (B, K, S, 3, H, W)
    metas = torch.stack([b[1] for b in batch], dim=0)        # (B, K, S, 6)
    masks = torch.stack([b[2] for b in batch], dim=0)        # (B, K)
    targets = torch.stack([b[3] for b in batch], dim=0)      # (B, 12)
    weights = torch.stack([b[4] for b in batch], dim=0)      # (B, 12)
    study_uids = [b[5] for b in batch]
    return slices, metas, masks, targets, weights, study_uids
