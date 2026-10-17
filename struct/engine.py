import copy
import math
import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint_atomic
from .config import PipelineConfig, seed_everything
from .data_discovery import DiscoveredDataset, discover_and_prepare_data
from .dataset import StudyBagDataset, study_bag_collate_fn
from .diagnostics import StageProfiler, capture_memory_snapshot, release_host_memory
from .metrics import compute_study_level_metrics
from .model import ConfidenceMaskedBCELoss, MultimodalKneeEngine
from .storage import prepare_mmap_cache


def build_epoch_study_dataframe(
    discovered: DiscoveredDataset, cfg: PipelineConfig, epoch: int
) -> pd.DataFrame:
    gold_part = pd.concat([discovered.train_gold_studies_df] * max(1, cfg.gold_oversample_factor), ignore_index=True)

    weak_df = discovered.train_weak_studies_df
    if cfg.use_report_supervision and len(weak_df) > 0 and cfg.weak_studies_per_epoch > 0:
        rng = np.random.default_rng(cfg.seed + epoch * 997)
        n_take = min(len(weak_df), cfg.weak_studies_per_epoch)
        chosen_idx = rng.choice(len(weak_df), size=n_take, replace=False)
        weak_part = weak_df.iloc[chosen_idx].reset_index(drop=True)
        combined = pd.concat([gold_part, weak_part], ignore_index=True)
    else:
        combined = gold_part

    return combined.sample(frac=1.0, random_state=cfg.seed + epoch).reset_index(drop=True)


def build_differential_optimizer(
    base_model: MultimodalKneeEngine, cfg: PipelineConfig
) -> torch.optim.Optimizer:
    """
    Uses a conservative 5e-5 learning rate for unfrozen pretrained backbone layers
    and a faster 4e-4 learning rate for the randomly initialized attention & classifier heads.
    """
    backbone_params = [p for p in base_model.backbone.parameters() if p.requires_grad]
    head_params = [
        p for n, p in base_model.named_parameters()
        if (not n.startswith("backbone.")) and p.requires_grad
    ]
    param_groups = [
        {"params": backbone_params, "lr": cfg.lr * 0.25, "weight_decay": cfg.weight_decay},
        {"params": head_params, "lr": cfg.lr * 1.5, "weight_decay": cfg.weight_decay},
    ]
    return torch.optim.AdamW(param_groups)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: ConfidenceMaskedBCELoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    cfg: PipelineConfig,
    profiler: StageProfiler,
    rank: int = 0,
) -> float:
    model.train()
    profiler.reset()

    total_steps = len(loader)
    if cfg.max_steps_per_epoch is not None:
        total_steps = min(total_steps, cfg.max_steps_per_epoch)

    accum_steps = max(1, cfg.grad_accumulation_steps)
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    completed_steps = 0

    iterator = tqdm(
        enumerate(loader),
        total=total_steps,
        desc=f"Train Ep {epoch + 1}/{cfg.epochs}",
        disable=(rank != 0),
        leave=True,
    )

    t_load_start = time.perf_counter()
    for step, (slices, metas, masks, targets, weights, _) in iterator:
        if step >= total_steps:
            break
        t_load_end = time.perf_counter()
        profiler.record("data_load", t_load_end - t_load_start)

        t_prep_start = time.perf_counter()
        slices = slices.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        t_prep_end = time.perf_counter()
        profiler.record("gpu_prep", t_prep_end - t_prep_start)

        is_last_step = (step + 1) == total_steps
        steps_in_window = accum_steps
        if is_last_step and ((step + 1) % accum_steps != 0):
            steps_in_window = (step + 1) % accum_steps

        t_fwd_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            logits = model(slices, metas, masks)
            raw_loss = criterion(logits, targets, weights)
            scaled_loss = raw_loss / steps_in_window
        t_fwd_end = time.perf_counter()
        profiler.record("forward", t_fwd_end - t_fwd_start)

        t_bwd_start = time.perf_counter()
        scaler.scale(scaled_loss).backward()
        t_bwd_end = time.perf_counter()
        profiler.record("backward", t_bwd_end - t_bwd_start)

        t_opt_start = time.perf_counter()
        if ((step + 1) % accum_steps == 0) or is_last_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        t_opt_end = time.perf_counter()
        profiler.record("optim_step", t_opt_end - t_opt_start)

        loss_val = float(raw_loss.detach().item())
        running_loss += loss_val
        completed_steps += 1

        del slices, metas, masks, targets, weights, logits, raw_loss, scaled_loss

        if rank == 0 and ((step + 1) % 10 == 0 or is_last_step):
            snap = capture_memory_snapshot("step", sync_ddp=False)
            iterator.set_postfix({
                "Loss": f"{running_loss / completed_steps:.4f}",
                "RSS": f"{snap.process_rss_gb:.2f}GB",
                "VRAM0": f"{snap.gpu_allocated_gb[0]:.2f}GB" if snap.gpu_allocated_gb else "N/A",
            })

        t_load_start = time.perf_counter()

    return running_loss / max(completed_steps, 1)


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: ConfidenceMaskedBCELoss,
    device: torch.device,
    cfg: PipelineConfig,
    world_size: int = 1,
    rank: int = 0,
) -> Tuple[float, float, Dict[str, float], int]:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_study_uids: List[str] = []

    for slices, metas, masks, targets, weights, study_uids in val_loader:
        slices = slices.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets_gpu = targets.to(device, non_blocking=True)
        weights_gpu = weights.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            logits = model(slices, metas, masks)
            loss = criterion(logits, targets_gpu, weights_gpu)

        total_loss += float(loss.detach().item())
        total_batches += 1

        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        all_probs.append(probs)
        all_targets.append(targets.numpy().astype(np.float32))
        all_study_uids.extend(study_uids)

        del slices, metas, masks, targets_gpu, weights_gpu, logits, loss

    local_probs = np.vstack(all_probs) if all_probs else np.zeros((0, len(cfg.labels)), dtype=np.float32)
    local_targets = np.vstack(all_targets) if all_targets else np.zeros((0, len(cfg.labels)), dtype=np.float32)

    if world_size > 1 and dist.is_initialized():
        gathered_objs = [None for _ in range(world_size)]
        dist.all_gather_object(
            gathered_objs,
            (all_study_uids, local_probs, local_targets, total_loss, total_batches),
        )
        merged_uids: List[str] = []
        merged_probs: List[np.ndarray] = []
        merged_targets: List[np.ndarray] = []
        total_loss = 0.0
        total_batches = 0
        for uids_part, probs_part, targs_part, l_part, b_part in gathered_objs:
            merged_uids.extend(uids_part)
            if len(probs_part) > 0:
                merged_probs.append(probs_part)
                merged_targets.append(targs_part)
            total_loss += l_part
            total_batches += b_part
        all_study_uids = merged_uids
        local_probs = np.vstack(merged_probs) if merged_probs else local_probs
        local_targets = np.vstack(merged_targets) if merged_targets else local_targets

    val_loss = total_loss / max(total_batches, 1)
    macro_auc, per_class_auc, num_studies = compute_study_level_metrics(
        all_study_uids, local_probs, local_targets, cfg.labels, aggregation="mean"
    )

    del all_probs, all_targets, local_probs, local_targets
    release_host_memory()
    return val_loss, macro_auc, per_class_auc, num_studies


def run_training_worker(
    rank: int,
    world_size: int,
    cfg: PipelineConfig,
    resume: bool = True,
    use_dataparallel_if_single_proc: bool = False,
) -> Dict[str, float]:
    is_ddp = world_size > 1
    if is_ddp:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29505")
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, device_id=device
        )
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg.ensure_dirs()
    seed_everything(cfg.seed, rank=rank)
    start_wall_time = time.time()

    discovered = discover_and_prepare_data(cfg, rank=rank)

    gold_uids = set(discovered.train_gold_studies_df["StudyInstanceUID"]).union(
        set(discovered.val_gold_studies_df["StudyInstanceUID"])
    )
    gold_series_df = discovered.series_df[discovered.series_df["StudyInstanceUID"].isin(gold_uids)]

    if rank == 0:
        path_lookup = prepare_mmap_cache([gold_series_df], cfg, rank=0)
    if is_ddp:
        dist.barrier(device_ids=[rank])
        if rank != 0:
            path_lookup = prepare_mmap_cache([gold_series_df], cfg, rank=rank)

    use_meta = cfg.use_metadata if cfg.use_metadata is not None else discovered.has_valid_metadata

    val_dataset = StudyBagDataset(
        discovered.val_gold_studies_df, discovered.series_df, path_lookup, cfg, is_train=False
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    eff_batch_size = cfg.batch_size_per_gpu

    val_loader = DataLoader(
        val_dataset,
        batch_size=eff_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=study_bag_collate_fn,
    )

    base_model = MultimodalKneeEngine(
        num_classes=len(cfg.labels),
        priors=discovered.priors,
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained,
        att_dim=cfg.attention_dim,
        dropout=cfg.dropout,
        use_metadata=use_meta,
        text_dim=cfg.metadata_dim,
        channels_last=cfg.channels_last,
    ).to(device)

    pos_weight_t = torch.tensor(discovered.pos_weights, dtype=torch.float32, device=device)
    criterion = ConfidenceMaskedBCELoss(pos_weight=pos_weight_t)
    optimizer = build_differential_optimizer(base_model, cfg)

    epoch0_df = build_epoch_study_dataframe(discovered, cfg, epoch=0)
    est_steps = math.ceil(len(epoch0_df) / max(1, eff_batch_size * world_size))
    if cfg.max_steps_per_epoch is not None:
        est_steps = min(est_steps, cfg.max_steps_per_epoch)
    opt_steps_per_epoch = max(1, math.ceil(est_steps / max(1, cfg.grad_accumulation_steps)))
    total_opt_steps = max(1, opt_steps_per_epoch * cfg.epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_opt_steps, eta_min=cfg.min_lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    latest_ckpt_path = os.path.join(cfg.checkpoint_dir, "checkpoint_latest.pth")
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, "checkpoint_best_val_auc.pth")
    top3_paths = [os.path.join(cfg.checkpoint_dir, f"checkpoint_top{i}.pth") for i in (1, 2, 3)]

    start_epoch = 0
    best_val_auc = 0.0
    top_scores: List[Tuple[float, str]] = []
    best_per_class: Dict[str, float] = {c: 0.0 for c in cfg.labels}

    if resume and os.path.exists(latest_ckpt_path):
        restored = load_checkpoint(
            latest_ckpt_path, base_model, optimizer, scheduler, scaler, map_location=str(device)
        )
        start_epoch = restored["epoch"]
        best_val_auc = restored["best_val_auc"]
        if rank == 0:
            print(f"Resumed from {latest_ckpt_path} at Epoch {start_epoch} (Best Val Macro AUC={best_val_auc:.4f})")

    if is_ddp:
        model: nn.Module = nn.parallel.DistributedDataParallel(
            base_model, device_ids=[rank], output_device=rank,
            find_unused_parameters=False, gradient_as_bucket_view=False
        )
    else:
        model = base_model

    profiler = StageProfiler()
    init_snap = capture_memory_snapshot("pre_train_init", sync_ddp=is_ddp)
    if rank == 0:
        print(init_snap.format_short())

    for epoch in range(start_epoch, cfg.epochs):
        elapsed_hours = (time.time() - start_wall_time) / 3600.0
        if elapsed_hours >= cfg.max_runtime_hours:
            if rank == 0:
                print(f"Reached runtime budget ({elapsed_hours:.2f}h). Stopping cleanly.")
            break

        epoch_studies_df = build_epoch_study_dataframe(discovered, cfg, epoch=epoch)
        train_dataset = StudyBagDataset(
            epoch_studies_df, discovered.series_df, path_lookup, cfg, is_train=True
        )
        train_dataset.set_epoch(epoch)
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed + epoch
        ) if is_ddp else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=eff_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=False,
            collate_fn=study_bag_collate_fn,
        )

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, cfg, profiler, rank=rank
        )

        val_loss, val_macro_auc, per_class_auc, num_val_studies = evaluate_validation(
            model, val_loader, criterion, device, cfg, world_size=world_size, rank=rank
        )

        del train_loader, train_dataset, epoch_studies_df
        release_host_memory()
        torch.cuda.empty_cache()

        mem_snap = capture_memory_snapshot(f"epoch_{epoch + 1}_end", sync_ddp=is_ddp)

        if rank == 0:
            for k, v in per_class_auc.items():
                if not math.isnan(v) and v > best_per_class.get(k, 0.0):
                    best_per_class[k] = v
            oracle_macro = float(np.mean(list(best_per_class.values())))

            prof_str = profiler.format_summary(eff_batch_size * world_size)
            print("")
            print(
                f"Epoch {epoch + 1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Macro AUC: {val_macro_auc:.4f} "
                f"(Peak Per-Class Mean: {oracle_macro:.4f})"
            )
            print(f"  Perf: {prof_str}")
            print(f"  Mem:  {mem_snap.format_short()}")
            per_cls_str = " | ".join(
                f"{k}:{v:.3f}" if not math.isnan(v) else f"{k}:N/A" for k, v in per_class_auc.items()
            )
            print(f"  Per-Class Val AUC: {per_cls_str}")

            is_new_best = (not math.isnan(val_macro_auc)) and (val_macro_auc > best_val_auc)
            if is_new_best:
                best_val_auc = val_macro_auc
                save_checkpoint_atomic(
                    best_ckpt_path, model, optimizer, scheduler, scaler, epoch + 1, best_val_auc, per_class_auc
                )
                print(f"  New Best Validation Macro AUC ({best_val_auc:.4f}) saved -> {best_ckpt_path}")

            # Maintain Top-3 epoch checkpoints for multi-epoch SWA/Ensemble inference
            if not math.isnan(val_macro_auc):
                if len(top_scores) < 3:
                    slot_path = top3_paths[len(top_scores)]
                    save_checkpoint_atomic(
                        slot_path, model, optimizer, scheduler, scaler, epoch + 1, val_macro_auc, per_class_auc
                    )
                    top_scores.append((val_macro_auc, slot_path))
                    top_scores.sort(key=lambda x: x[0], reverse=True)
                elif val_macro_auc > top_scores[-1][0]:
                    worst_path = top_scores[-1][1]
                    save_checkpoint_atomic(
                        worst_path, model, optimizer, scheduler, scaler, epoch + 1, val_macro_auc, per_class_auc
                    )
                    top_scores[-1] = (val_macro_auc, worst_path)
                    top_scores.sort(key=lambda x: x[0], reverse=True)

            save_checkpoint_atomic(
                latest_ckpt_path, model, optimizer, scheduler, scaler, epoch + 1, best_val_auc, per_class_auc
            )

        if is_ddp:
            dist.barrier(device_ids=[rank])

    summary = profiler.summary_dict()
    summary["best_val_auc"] = best_val_auc
    if is_ddp:
        dist.destroy_process_group()
    return summary


@torch.no_grad()
def generate_study_submission(cfg: PipelineConfig, output_csv: str = "submission.csv") -> pd.DataFrame:
    """
    Loads the Top-3 validation checkpoints (or best checkpoint) and averages their
    study-level probability predictions with per-class validation AUC weighting.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    discovered = discover_and_prepare_data(cfg, rank=0)

    if discovered.test_studies_df is not None and discovered.test_series_df is not None and len(discovered.test_series_df) > 0:
        studies_df = discovered.test_studies_df
        series_df = discovered.test_series_df
    else:
        studies_df = discovered.val_gold_studies_df
        series_df = discovered.series_df

    path_lookup = {p: p for p in series_df["npz_path"].tolist()}
    use_meta = cfg.use_metadata if cfg.use_metadata is not None else discovered.has_valid_metadata

    dataset = StudyBagDataset(studies_df, series_df, path_lookup, cfg, is_train=False)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_per_gpu * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=study_bag_collate_fn,
    )

    cand_ckpts = [
        os.path.join(cfg.checkpoint_dir, f"checkpoint_top{i}.pth") for i in (1, 2, 3)
    ]
    cand_ckpts = [p for p in cand_ckpts if os.path.exists(p)]
    if not cand_ckpts:
        best_p = os.path.join(cfg.checkpoint_dir, "checkpoint_best_val_auc.pth")
        latest_p = os.path.join(cfg.checkpoint_dir, "checkpoint_latest.pth")
        cand_ckpts = [best_p if os.path.exists(best_p) else latest_p]

    model = MultimodalKneeEngine(
        num_classes=len(cfg.labels),
        priors=discovered.priors,
        backbone_name=cfg.backbone_name,
        pretrained=False,
        att_dim=cfg.attention_dim,
        dropout=cfg.dropout,
        use_metadata=use_meta,
        text_dim=cfg.metadata_dim,
        channels_last=cfg.channels_last,
    ).to(device)

    ensemble_probs = None
    weight_sum = None
    all_study_uids: List[str] = []

    for ckpt_idx, ckpt_path in enumerate(cand_ckpts):
        restored = load_checkpoint(ckpt_path, model, map_location=str(device))
        val_m = restored.get("val_metrics", {})
        cls_w = np.array(
            [max(float(val_m.get(c, 0.5)), 0.5) if not math.isnan(float(val_m.get(c, 0.5))) else 0.5 for c in cfg.labels],
            dtype=np.float32,
        )
        print(f"Loaded ensemble checkpoint {ckpt_idx + 1}/{len(cand_ckpts)}: {ckpt_path} (Val AUC={restored.get('best_val_auc', 0.0):.4f})")

        model.eval()
        ckpt_probs: List[np.ndarray] = []
        uids_this_pass: List[str] = []

        for slices, metas, masks, _, _, study_uids in tqdm(loader, desc=f"Inference ({os.path.basename(ckpt_path)})"):
            slices = slices.to(device, non_blocking=True)
            metas = metas.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                logits = model(slices, metas, masks)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            ckpt_probs.append(probs)
            uids_this_pass.extend(study_uids)

        pass_mat = np.vstack(ckpt_probs)
        if ensemble_probs is None:
            ensemble_probs = pass_mat * cls_w[None, :]
            weight_sum = cls_w[None, :].copy()
            all_study_uids = uids_this_pass
        else:
            ensemble_probs += pass_mat * cls_w[None, :]
            weight_sum += cls_w[None, :]

    final_probs = ensemble_probs / np.maximum(weight_sum, 1e-6)
    sub_df = pd.DataFrame(final_probs, columns=cfg.labels)
    sub_df.insert(0, "StudyInstanceUID", all_study_uids)
    sub_df.to_csv(output_csv, index=False)
    print(f"Saved Top-{len(cand_ckpts)} ensemble study-level submission ({len(sub_df)} studies) -> {output_csv}")
    return sub_df
