"""Keep optional evaluation passes from changing subsequent training randomness."""
from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def evaluation_rng(policy="legacy", device="cpu"):
    """Isolate global RNGs, or preserve the original evaluation trajectory."""
    if policy == "legacy":
        yield
        return
    if policy != "isolated":
        raise ValueError("evaluation RNG policy must be legacy or isolated")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("evaluation RNG isolation supports CPU and CUDA devices")
    # Explicit logical indices avoid fork_rng's default initialization of all GPUs.
    devices = []
    if device.type == "cuda":
        devices = [torch.cuda.current_device() if device.index is None else device.index]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
