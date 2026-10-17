import ctypes
import gc
import os
from dataclasses import dataclass, field
from typing import Dict, List
import psutil
import torch
import torch.distributed as dist

try:
    _libc = ctypes.CDLL("libc.so.6")
except Exception:
    _libc = None


def release_host_memory() -> None:
    """Runs Python GC and trims unreferenced glibc heap arenas back to Linux OS."""
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


@dataclass
class MemorySnapshot:
    stage: str
    process_rss_gb: float
    process_uss_gb: float
    sys_available_gb: float
    sys_total_gb: float
    gpu_allocated_gb: List[float]
    gpu_peak_gb: List[float]
    gpu_hw_used_gb: List[float]

    def format_short(self) -> str:
        gpu_parts = []
        for i in range(len(self.gpu_hw_used_gb)):
            alloc = self.gpu_allocated_gb[i] if i < len(self.gpu_allocated_gb) else 0.0
            peak = self.gpu_peak_gb[i] if i < len(self.gpu_peak_gb) else 0.0
            hw = self.gpu_hw_used_gb[i]
            gpu_parts.append(f"GPU{i}:Alloc={alloc:.2f}GB(Peak={peak:.2f}GB,HW={hw:.2f}GB)")
        gpu_str = " | ".join(gpu_parts)
        return (
            f"[{self.stage}] ProcRSS={self.process_rss_gb:.2f}GB "
            f"(USS={self.process_uss_gb:.2f}GB) | "
            f"SysAvail={self.sys_available_gb:.2f}/{self.sys_total_gb:.1f}GB"
            + (f" | {gpu_str}" if gpu_str else "")
        )


def capture_memory_snapshot(stage: str, sync_ddp: bool = False) -> MemorySnapshot:
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    rss_gb = mem_info.rss / (1024 ** 3)
    try:
        uss_gb = proc.memory_full_info().uss / (1024 ** 3)
    except Exception:
        uss_gb = rss_gb

    vm = psutil.virtual_memory()
    avail_gb = vm.available / (1024 ** 3)
    total_gb = vm.total / (1024 ** 3)

    gpu_alloc, gpu_peak, gpu_hw = [], [], []
    if torch.cuda.is_available():
        n_dev = torch.cuda.device_count()
        for i in range(n_dev):
            gpu_alloc.append(torch.cuda.memory_allocated(i) / (1024 ** 3))
            gpu_peak.append(torch.cuda.max_memory_allocated(i) / (1024 ** 3))
            try:
                free_b, total_b = torch.cuda.mem_get_info(i)
                gpu_hw.append((total_b - free_b) / (1024 ** 3))
            except Exception:
                gpu_hw.append(0.0)

        if sync_ddp and dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            local_stats = (
                rank,
                torch.cuda.memory_allocated(rank) / (1024 ** 3),
                torch.cuda.max_memory_allocated(rank) / (1024 ** 3),
            )
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local_stats)
            for r_idx, r_alloc, r_peak in gathered:
                if 0 <= r_idx < n_dev:
                    gpu_alloc[r_idx] = r_alloc
                    gpu_peak[r_idx] = r_peak

    return MemorySnapshot(
        stage=stage,
        process_rss_gb=rss_gb,
        process_uss_gb=uss_gb,
        sys_available_gb=avail_gb,
        sys_total_gb=total_gb,
        gpu_allocated_gb=gpu_alloc,
        gpu_peak_gb=gpu_peak,
        gpu_hw_used_gb=gpu_hw,
    )


@dataclass
class StageProfiler:
    timings: Dict[str, List[float]] = field(default_factory=lambda: {
        "data_load": [],
        "gpu_prep": [],
        "forward": [],
        "backward": [],
        "optim_step": [],
    })

    def reset(self) -> None:
        for k in self.timings:
            self.timings[k].clear()

    def record(self, key: str, elapsed_sec: float) -> None:
        if key in self.timings:
            self.timings[key].append(elapsed_sec)

    def summary_dict(self) -> Dict[str, float]:
        out = {}
        total_mean = 0.0
        for k, vals in self.timings.items():
            m = float(sum(vals) / max(len(vals), 1))
            out[f"{k}_ms"] = m * 1000.0
            total_mean += m
        out["step_total_ms"] = total_mean * 1000.0
        if total_mean > 0:
            out["data_Wait_pct"] = 100.0 * (out["data_load_ms"] / out["step_total_ms"])
            out["gpu_Active_pct"] = 100.0 - out["data_Wait_pct"]
        else:
            out["data_Wait_pct"] = 0.0
            out["gpu_Active_pct"] = 0.0
        return out

    def format_summary(self, batch_size: int) -> str:
        s = self.summary_dict()
        step_sec = max(s["step_total_ms"] / 1000.0, 1e-6)
        throughput = batch_size / step_sec
        return (
            f"Throughput={throughput:.1f} series/s | "
            f"Load={s['data_load_ms']:.1f}ms ({s['data_Wait_pct']:.0f}%) | "
            f"GPUPrep={s['gpu_prep_ms']:.1f}ms | "
            f"Fwd={s['forward_ms']:.1f}ms | "
            f"Bwd={s['backward_ms']:.1f}ms | "
            f"Opt={s['optim_step_ms']:.1f}ms"
        )
