import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch

TARGET_LABELS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

@dataclass
class PipelineConfig:
    seed: int = 42

    # Paths
    input_root: str = "/kaggle/input"
    work_dir: str = "/kaggle/working"
    mmap_cache_dir: str = "/kaggle/working/npy_mmap_cache"
    checkpoint_dir: str = "/kaggle/working/checkpoints_study_mil"

    # Storage & Conversion
    enable_mmap_conversion: bool = True
    min_free_disk_gb: float = 5.0
    preserve_source_dtype: bool = True
    validate_converted_files: bool = True

    # Validation & Weak Supervision
    fold: int = 0
    num_folds: int = 5
    use_report_supervision: bool = True
    weak_label_weight_scale: float = 0.75   # Downweights report-derived labels vs 1.0 gold labels
    gold_oversample_factor: int = 4         # Repeat the 46 gold train studies 4x per epoch (184 bags)
    weak_studies_per_epoch: int = 400       # Sample 400 weak studies per epoch (rotates through all 4,349 across epochs)

    # Study-Level Multi-Series Sampling (No Horizontal Flip, Square Pad before Resize)
    max_series_per_study: int = 3           # 3 complementary series per study (Sagittal, Coronal, Axial)
    seq_len: int = 8                        # 8 slices per series -> 3 * 8 = 24 slices per study bag
    val_chunks_per_series: int = 1
    image_size: Tuple[int, int] = (224, 224)
    pad_to_square: bool = True              # Preserves aspect ratio before resize (per .903 recipe)
    train_sampling_mode: str = "stratified_jitter"
    study_aggregation: str = "attention"    # Learned class-specific cross-series attention

    # Model & Metadata
    backbone_name: str = "efficientnet_b0"
    pretrained: bool = True
    attention_dim: int = 256
    dropout: float = 0.3
    use_metadata: Optional[bool] = True
    metadata_dim: int = 6                   # [Fluid_Sensitive, Fat_Suppression, Sagittal, Coronal, Axial]
    channels_last: bool = False

    # Optimization & Schedule
    epochs: int = 12
    max_steps_per_epoch: Optional[int] = None
    batch_size_per_gpu: int = 4             # 4 studies (12 series = 96 slices) per GPU step
    grad_accumulation_steps: int = 4        # Effective batch = 4 * 2 GPUs * 4 accum = 32 studies
    lr: float = 2e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    pos_weight_cap: float = 8.0
    use_amp: bool = True

    # DataLoader & Hardware
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True

    # Runtime Budget
    max_runtime_hours: float = 8.5

    labels: List[str] = field(default_factory=lambda: list(TARGET_LABELS))

    def ensure_dirs(self) -> None:
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.mmap_cache_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


def seed_everything(seed: int = 42, rank: int = 0) -> None:
    actual_seed = seed + rank
    random.seed(actual_seed)
    os.environ["PYTHONHASHSEED"] = str(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
