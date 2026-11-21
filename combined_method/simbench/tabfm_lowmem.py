"""Load the TabFM v1.0.0 classifier (PyTorch, bfloat16, CPU) once per process.

Default: the official loader, which downloads google/tabfm-1.0.0-pytorch from Hugging Face on first use
(needs about 7 GB of RAM while loading; fine on a 16-32 GB machine).
Low-memory option: set TABFM_BF16 to a folder with a pre-converted bfloat16 copy (config.json +
model.safetensors); the weights are then loaded straight into bfloat16 tensors (about 3.3 GB of RAM).
"""
import json
import os

import torch

_M = None


def _load_bf16(d):
    from safetensors import safe_open
    from tabfm.src.pytorch.tabfm_v1_0_0 import TabFM_HF
    cfg = json.load(open(os.path.join(d, "config.json")))
    with torch.device("meta"):
        m = TabFM_HF(**cfg)
    sd = {}
    with safe_open(os.path.join(d, "model.safetensors"), "pt") as s:
        for k in s.keys():
            sd[k] = s.get_tensor(k).to(torch.bfloat16)
    m.load_state_dict(sd, strict=True, assign=True)
    return m.eval()


def load():
    d = os.environ.get("TABFM_BF16")
    if d and os.path.exists(os.path.join(d, "model.safetensors")):
        return _load_bf16(d)
    from tabfm.src.pytorch.tabfm_v1_0_0 import load as official
    return official("classification", device="cpu", dtype=torch.bfloat16)


def load_cached():
    global _M
    if _M is None:
        _M = load()
    return _M
