"""Console diagnostics for the active compute device."""

from __future__ import annotations

import platform

import psutil
import torch


def print_device_info(device: torch.device) -> None:
    """Print a short summary of the selected device and host hardware."""
    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        print(f" -> {num_gpus} CUDA device(s) available")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"   [GPU {i}] {torch.cuda.get_device_name(i)}")
            print(f"       Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"       Memory Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
            print(f"       Total Memory:     {props.total_memory / 1024 ** 3:.2f} GB")
            print(f"       Compute Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print(f" -> CPU: {platform.processor() or 'Unknown'}")
        print(
            f" -> CPU cores: {psutil.cpu_count(logical=False)} physical, "
            f"{psutil.cpu_count(logical=True)} logical"
        )
        print(f" -> RAM available: {psutil.virtual_memory().total / 1024 ** 3:.2f} GB")
