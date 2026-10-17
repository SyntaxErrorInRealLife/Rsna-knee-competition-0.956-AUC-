import os
import random
from typing import Any, Dict, Optional
import numpy as np
import torch


def save_checkpoint_atomic(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_auc: float,
    val_metrics: Dict[str, float],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "epoch": int(epoch),
        "best_val_auc": float(best_val_auc),
        "val_metrics": val_metrics,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra_meta": extra_meta or {},
    }
    tmp_path = f"{filepath}.tmp.{os.getpid()}"
    torch.save(state, tmp_path)
    os.replace(tmp_path, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    map_location: str = "cpu",
    restore_rng: bool = False,
) -> Dict[str, Any]:
    ckpt = torch.load(filepath, map_location=map_location, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        raw_model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if restore_rng and "rng_state" in ckpt:
            try:
                random.setstate(ckpt["rng_state"]["python"])
                np.random.set_state(ckpt["rng_state"]["numpy"])
                torch.set_rng_state(ckpt["rng_state"]["torch"].cpu())
            except Exception:
                pass
        return {
            "epoch": int(ckpt.get("epoch", 0)),
            "best_val_auc": float(ckpt.get("best_val_auc", 0.0)),
            "val_metrics": ckpt.get("val_metrics", {}),
        }

    # Fallback for legacy weights-only checkpoints
    raw_model.load_state_dict(ckpt)
    return {"epoch": 0, "best_val_auc": 0.0, "val_metrics": {}}
