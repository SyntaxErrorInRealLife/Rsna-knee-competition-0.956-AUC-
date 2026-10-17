import os
import tempfile
import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint_atomic
from .config import PipelineConfig, seed_everything
from .data_discovery import discover_and_prepare_data
from .dataset import StudyBagDataset
from .diagnostics import release_host_memory
from .model import ConfidenceMaskedBCELoss, MultimodalKneeEngine


def run_phase3_verification_suite(cfg: PipelineConfig) -> None:
    print("====================================================")
    print("PHASE 3: STUDY-LEVEL + WEAK-LABEL VERIFICATION SUITE")
    print("====================================================")
    seed_everything(cfg.seed)

    disc = discover_and_prepare_data(cfg, rank=0)
    val_uids = set(disc.val_gold_studies_df["StudyInstanceUID"])
    tr_gold_uids = set(disc.train_gold_studies_df["StudyInstanceUID"])
    tr_weak_uids = set(disc.train_weak_studies_df["StudyInstanceUID"])

    assert val_uids.isdisjoint(tr_gold_uids), "Leakage between Val Gold and Train Gold!"
    assert val_uids.isdisjoint(tr_weak_uids), "Leakage between Val Gold and Train Weak!"
    print(
        f"[PASS] Zero Study Leakage Verified: "
        f"{len(tr_gold_uids)} Train Gold Studies + {len(tr_weak_uids)} Train Weak Report Studies | "
        f"{len(val_uids)} Held-Out Val Gold Studies."
    )

    path_lookup = {p: p for p in disc.series_df["npz_path"].tolist()}
    ds = StudyBagDataset(disc.train_gold_studies_df.head(4), disc.series_df, path_lookup, cfg, is_train=True)
    bag_slices, bag_metas, series_mask, target_t, weight_t, study_uid = ds[0]

    assert bag_slices.shape == (cfg.max_series_per_study, cfg.seq_len, 3, cfg.image_size[0], cfg.image_size[1])
    assert bag_metas.shape == (cfg.max_series_per_study, cfg.seq_len, cfg.metadata_dim)
    assert series_mask.any(), "At least one series slot must be valid!"
    print(
        f"[PASS] StudyBagDataset verified for Study {study_uid[:24]}...: "
        f"bag_slices={tuple(bag_slices.shape)}, valid_series={int(series_mask.sum())}/{cfg.max_series_per_study}"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultimodalKneeEngine(
        num_classes=12, pretrained=False, use_metadata=True, text_dim=cfg.metadata_dim
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    criterion = ConfidenceMaskedBCELoss()

    bx = bag_slices.unsqueeze(0).to(device)
    bm = bag_metas.unsqueeze(0).to(device)
    bmask = series_mask.unsqueeze(0).to(device)
    by = target_t.unsqueeze(0).to(device)
    bw = weight_t.unsqueeze(0).to(device)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits = model(bx, bm, bmask)
        assert logits.shape == (1, 12), f"Expected (1, 12) study logits, got {logits.shape}"
        loss = criterion(logits, by, bw)

    assert torch.isfinite(loss).item(), "Non-finite study loss!"
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    sched.step()
    print(f"[PASS] HierarchicalStudyMIL AMP forward/backward pass verified (Study Loss={loss.item():.4f}).")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_file = os.path.join(tmpdir, "test_study_ckpt.pth")
        save_checkpoint_atomic(ckpt_file, model, opt, sched, scaler, epoch=2, best_val_auc=0.75, val_metrics={"ACL": 0.8})
        model2 = MultimodalKneeEngine(
            num_classes=12, pretrained=False, use_metadata=True, text_dim=cfg.metadata_dim
        ).to(device)
        restored = load_checkpoint(ckpt_file, model2, map_location=str(device))
        assert restored["epoch"] == 2 and abs(restored["best_val_auc"] - 0.75) < 1e-6
    print("[PASS] Atomic study-level checkpoint save & reload verified.")

    release_host_memory()
    print("====================================================")
    print("ALL STUDY-LEVEL & WEAK-LABEL CHECKS PASSED!")
    print("====================================================")
