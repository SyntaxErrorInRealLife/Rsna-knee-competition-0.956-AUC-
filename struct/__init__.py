"""RSNA Knee Abnormality Detection — Modular Training & Inference Package."""
import os

# Configure glibc allocator to use mmap for allocations >= 64KB so freed buffers
# return immediately to the Linux OS instead of fragmenting heap arenas.
os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "65536")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
