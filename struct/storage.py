import os
import shutil
from typing import Dict, List
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .diagnostics import release_host_memory


def _extract_primary_volume(npz_handle) -> np.ndarray:
    key = "volume" if "volume" in npz_handle else ("arr_0" if "arr_0" in npz_handle else npz_handle.files[0])
    arr = npz_handle[key]
    if arr.ndim == 4:
        arr = arr[..., 0] if arr.shape[-1] <= 3 else arr[0]
    return arr


def prepare_mmap_cache(
    dfs: List[pd.DataFrame], cfg: PipelineConfig, rank: int = 0
) -> Dict[str, str]:
    """
    Non-destructive, resumable, disk-budget-aware conversion of compressed .npz
    volumes to uncompressed .npy files suitable for zero-copy mmap_mode='r'.
    Preserves original dimensions, orientation, and intensity dtype.
    """
    os.makedirs(cfg.mmap_cache_dir, exist_ok=True)
    path_lookup: Dict[str, str] = {}

    unique_npz_paths: List[str] = []
    seen = set()
    for df in dfs:
        if df is None:
            continue
        for p in df["npz_path"].tolist():
            if p and p not in seen:
                seen.add(p)
                unique_npz_paths.append(p)

    if not cfg.enable_mmap_conversion:
        return {p: p for p in unique_npz_paths}

    converted_count = 0
    reused_count = 0
    fallback_npz_count = 0
    min_free_bytes = int(cfg.min_free_disk_gb * (1024 ** 3))

    for idx, npz_path in enumerate(unique_npz_paths):
        stem = os.path.basename(npz_path)[:-4]
        npy_path = os.path.join(cfg.mmap_cache_dir, f"{stem}.npy")

        # 1. Resumable check: if valid .npy already exists on disk, reuse it immediately
        if os.path.exists(npy_path) and os.path.getsize(npy_path) > 128:
            try:
                mm = np.load(npy_path, mmap_mode="r")
                if mm.ndim == 3 and mm.shape[0] >= 1:
                    path_lookup[npz_path] = npy_path
                    reused_count += 1
                    del mm
                    continue
            except Exception:
                try:
                    os.remove(npy_path)
                except OSError:
                    pass

        # 2. Check available disk space before converting
        free_bytes = shutil.disk_usage(cfg.mmap_cache_dir).free
        if free_bytes <= min_free_bytes:
            path_lookup[npz_path] = npz_path
            fallback_npz_count += 1
            continue

        # 3. Decompress once, validate, and write atomically via temp file
        tmp_path = f"{npy_path}.tmp.{os.getpid()}"
        try:
            with np.load(npz_path) as data:
                vol = _extract_primary_volume(data)
                orig_shape = vol.shape
                orig_dtype = vol.dtype
                orig_mid_val = float(vol[orig_shape[0] // 2].mean())

                if vol.nbytes + min_free_bytes > free_bytes:
                    path_lookup[npz_path] = npz_path
                    fallback_npz_count += 1
                    del vol
                    continue

                with open(tmp_path, "wb") as f_out:
                    np.save(f_out, vol, allow_pickle=False)
                del vol

            os.replace(tmp_path, npy_path)

            if cfg.validate_converted_files:
                mm = np.load(npy_path, mmap_mode="r")
                assert mm.shape == orig_shape, f"Shape mismatch: {mm.shape} != {orig_shape}"
                assert mm.dtype == orig_dtype, f"Dtype mismatch: {mm.dtype} != {orig_dtype}"
                mm_mid_val = float(mm[orig_shape[0] // 2].mean())
                assert abs(mm_mid_val - orig_mid_val) < 1e-4, "Intensity verification failed!"
                del mm

            path_lookup[npz_path] = npy_path
            converted_count += 1
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            path_lookup[npz_path] = npz_path
            fallback_npz_count += 1

        if (idx + 1) % 100 == 0:
            release_host_memory()

    release_host_memory()
    if rank == 0:
        free_gb = shutil.disk_usage(cfg.mmap_cache_dir).free / (1024 ** 3)
        print("=== STORAGE CACHE SUMMARY ===")
        print(
            f"  Converted new .npy: {converted_count} | Reused existing .npy: {reused_count} | "
            f"Fallback .npz: {fallback_npz_count} | Free Working Disk: {free_gb:.2f} GB"
        )
    return path_lookup
