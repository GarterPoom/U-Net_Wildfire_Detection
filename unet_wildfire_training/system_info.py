"""Console diagnostics for the active compute device.

Utility that prints a concise summary of the currently selected torch device
(and, when running on CPU, some basic host information such as processor count
and RAM usage).  The output is meant for quick interactive inspection rather than
programmatic consumption.
"""

from __future__ import annotations                     # Allows forward references in type hints

import platform                                       # For system / processor name detection
import psutil                                         # System‑level utilities (CPU count, memory)
import torch                                          # PyTorch – provides CUDA device info

def print_device_info(device: torch.device) -> None:
    """
    Print a short human‑readable summary of the selected ``torch.device``.

    The function distinguishes between **CUDA** devices and **CPU** execution:

    * **CUDA**  
        - Number of visible GPUs (`torch.cuda.device_count()`).
        - For each GPU: name, total memory, currently allocated,
          currently reserved (cached), compute capability.
    * **CPU**  
        - Processor brand string (`platform.processor()`).
        - Logical and physical core counts via ``psutil``.
        - Total RAM available on the machine.

    Parameters
    ----------
    device : torch.device
        The device whose information we want to display (e.g. ``torch.device('cuda:0')``).

    Returns
    -------
    None
        The function prints directly to stdout; it does not return a value.
    """
    if device.type == "cuda":
        # ---------- CUDA specific ----------
        num_gpus = torch.cuda.device_count()
        print(f" -> {num_gpus} CUDA device(s) available")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"   [GPU {i}] {torch.cuda.get_device_name(i)}")
            # Memory stats are shown in megabytes for readability.
            print(f"       Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"       Memory Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
            # Total memory is expressed in gigabytes.
            print(f"       Total Memory:     {props.total_memory / 1024 ** 3:.2f} GB")
            # Compute capability (major.minor) – useful when checking kernel support.
            print(f"       Compute Capability: {torch.cuda.get_device_capability(i)}")
    else:
        # ---------- CPU specific ----------
        print(f" -> CPU: {platform.processor() or 'Unknown'}")
        # psutil gives both physical (non‑logical) and logical core counts.
        print(
            f" -> CPU cores: {psutil.cpu_count(logical=False)} physical, "
            f"{psutil.cpu_count(logical=True)} logical"
        )
        mem_total = psutil.virtual_memory().total
        print(f" -> RAM available: {mem_total / 1024 ** 3:.2f} GB")
