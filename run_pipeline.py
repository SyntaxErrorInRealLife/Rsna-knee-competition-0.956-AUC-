import argparse
import os
import torch
import torch.multiprocessing as mp

from rsna_knee.config import PipelineConfig
from rsna_knee.engine import generate_study_submission, run_training_worker
from rsna_knee.verify import run_phase3_verification_suite


def parse_args():
    parser = argparse.ArgumentParser(description="RSNA Knee Study-Level MIL + Weak Supervision Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="train_ddp",
        choices=["verify", "benchmark_1gpu", "benchmark_2gpu_ddp", "train_1gpu", "train_ddp", "infer"],
        help="Execution mode",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_series_per_study", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--weak_studies_per_epoch", type=int, default=400)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--disable_mmap_cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    is_benchmark = args.mode in ("benchmark_1gpu", "benchmark_2gpu_ddp")

    cfg = PipelineConfig(
        epochs=2 if is_benchmark else args.epochs,
        max_steps_per_epoch=25 if is_benchmark else args.max_steps_per_epoch,
        batch_size_per_gpu=args.batch_size_per_gpu,
        grad_accumulation_steps=args.grad_accum,
        max_series_per_study=args.max_series_per_study,
        seq_len=args.seq_len,
        weak_studies_per_epoch=args.weak_studies_per_epoch,
        num_workers=args.num_workers,
        enable_mmap_conversion=(not args.disable_mmap_cache),
        checkpoint_dir="/kaggle/working/checkpoints_study_bench" if is_benchmark else "/kaggle/working/checkpoints_study_mil",
    )

    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        run_training_worker(
            rank=local_rank,
            world_size=world_size,
            cfg=cfg,
            resume=(False if is_benchmark else (not args.no_resume)),
        )
        return

    if args.mode == "verify":
        run_phase3_verification_suite(cfg)
    elif args.mode == "benchmark_1gpu":
        run_training_worker(rank=0, world_size=1, cfg=cfg, resume=False)
    elif args.mode == "benchmark_2gpu_ddp":
        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            mp.spawn(run_training_worker, args=(n_gpus, cfg, False, False), nprocs=n_gpus, join=True)
        else:
            run_training_worker(rank=0, world_size=1, cfg=cfg, resume=False)
    elif args.mode == "train_1gpu":
        run_training_worker(rank=0, world_size=1, cfg=cfg, resume=(not args.no_resume))
    elif args.mode == "train_ddp":
        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            mp.spawn(run_training_worker, args=(n_gpus, cfg, not args.no_resume, False), nprocs=n_gpus, join=True)
        else:
            run_training_worker(rank=0, world_size=1, cfg=cfg, resume=(not args.no_resume))
    elif args.mode == "infer":
        generate_study_submission(cfg, output_csv="submission.csv")


if __name__ == "__main__":
    main()
