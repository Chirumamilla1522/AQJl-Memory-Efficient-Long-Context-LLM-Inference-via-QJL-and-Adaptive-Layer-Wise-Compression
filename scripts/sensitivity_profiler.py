#!/usr/bin/env python3
"""
A-QJL Sensitivity Profiler: estimates per-layer sensitivity to compression.

Uses key magnitude variance as a proxy for sensitivity. Layers with higher
key variance are assigned higher projection dimension (k); layers with lower
variance get lower k, under a fixed memory budget.

Output: layer_group_boundaries, key_quantization_bits_per_group suitable
for run_longbench.py --layer_group_boundaries ... --key_quantization_bits_per_group ...

Backends:
  * torch — QJL-patched Llama (CUDA if available, else CPU).
  * mlx — Apple Silicon: native Llama via mlx-lm (install requirements-mlx.txt); same k_proj variance proxy.
  * auto — CUDA torch if available; else on macOS use MLX if installed; else torch CPU.
"""

import argparse
import json
import os
import random
import sys
from typing import List, Tuple

# Repo root must be on sys.path when invoked as `python3 scripts/sensitivity_profiler.py`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from datasets import load_dataset

from models.aqjl_budget import (
    group_layer_counts,
    percentile_boundaries,
    repair_surrogate_budget,
    surrogate_budget,
)


def _detect_backend(explicit: str) -> str:
    if explicit in ("torch", "mlx"):
        return explicit
    if explicit != "auto":
        raise ValueError(f"unknown --backend {explicit}")
    try:
        import torch

        if torch.cuda.is_available():
            return "torch"
    except ImportError:
        pass
    if sys.platform == "darwin":
        try:
            import mlx.core as mx  # noqa: F401

            return "mlx"
        except ImportError:
            pass
    return "torch"


def setup_model_torch(model_name: str, device: str, seed: int = 42):
    """Load QJL-patched Llama in PyTorch (CUDA/CPU) for k_proj variance profiling."""
    import torch
    from transformers import LlamaConfig, AutoTokenizer

    from models.llama2_utils_qjl import QJLSketch
    from models.llama2_qjl import LlamaForCausalLM_QJL

    dtype = torch.float16 if device != "cpu" else torch.float32
    config = LlamaConfig.from_pretrained(model_name)
    config._flash_attn_2_enabled = True
    config.attention_dropout = 0.0
    config.key_quantization_bits = 256
    config.key_quantization_bits_initial_layers = 256
    config.initial_layers_count = 32  # use same for all during profile
    config.outlier_count_general = 0
    config.outlier_count_initial_layers = 0
    config.value_quantization_bits = 2
    config.group_size = 32
    config.buffer_size = 128
    config.layer_group_boundaries = None
    config.qjl_groups = None

    torch.manual_seed(int(seed))
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    gen2 = torch.Generator(device=torch.device(device))
    gen2.manual_seed(int(seed) + 1)
    config.qjl = QJLSketch(dim=(128, 256), dim_outlier=0, rot=True, rng=generator)
    config.qjl_initial_layers = QJLSketch(dim=(128, 256), dim_outlier=0, rot=True, rng=gen2)
    config._flash_attn_2_enabled = device == "cuda"
    config.use_flash = device == "cuda"

    device_map = "auto" if device == "cuda" else None
    model = LlamaForCausalLM_QJL.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    if device == "cpu":
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, trust_remote_code=True, tokenizer_type="llama"
    )
    return model, tokenizer


def setup_model_mlx(model_name: str, seed: int = 42):
    """Load a Llama-compatible checkpoint in MLX (Apple Silicon) via mlx-lm."""
    import mlx.core as mx
    from mlx_lm import load

    mx.random.seed(int(seed))
    model, tokenizer = load(model_name)
    return model, tokenizer


def collect_key_variance_torch(
    model,
    tokenizer,
    calibration_data: List[str],
    max_seq_len: int = 1024,
    device: str = "cuda",
) -> List[float]:
    """
    Run forward passes and collect per-layer key variance (PyTorch QJL model).
    Returns sensitivity scores (higher = more sensitive to compression).
    """
    import torch

    model.eval()
    num_layers = model.config.num_hidden_layers
    layer_variances = [0.0] * num_layers
    counts = [0] * num_layers

    # Hook to capture key states
    key_states_per_layer = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # out is (hidden_states, attn_weights, past_kv)
            # We need key from attention - hook k_proj output
            pass
        return hook

    # We need to capture keys from inside the attention. Simpler: run forward with
    # output_hidden_states and a custom forward that records keys. That requires
    # model changes.
    # Alternative: run model with a hook on k_proj. The decoder layer calls
    # self_attn which does k_proj(hidden_states). So we hook each layer's k_proj.

    hooks = []
    for idx, layer in enumerate(model.model.layers):
        k_proj = layer.self_attn.k_proj

        def make_capture(li):
            def hook(module, inp, out):
                # out: (batch, seq, num_kv_heads * head_dim)
                with torch.no_grad():
                    v = out.float().var().item()
                    layer_variances[li] += v
                    counts[li] += 1
            return hook

        hooks.append(k_proj.register_forward_hook(make_capture(idx)))

    try:
        for i, text in enumerate(calibration_data):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
                return_attention_mask=True,
            )
            inputs = inputs.to(device)
            with torch.no_grad():
                _ = model(**inputs, output_attentions=False)
    finally:
        for h in hooks:
            h.remove()

    for i in range(num_layers):
        if counts[i] > 0:
            layer_variances[i] /= counts[i]
        else:
            layer_variances[i] = 1.0  # default

    return layer_variances


def collect_key_variance_mlx(
    model,
    tokenizer,
    calibration_data: List[str],
    max_seq_len: int = 1024,
) -> List[float]:
    """
    Per-layer variance of k_proj inputs (same proxy as PyTorch path) using mlx-lm Llama.
    Temporarily wraps each Attention.__call__ to record Var(k_proj(x)).
    """
    import types

    import mlx.core as mx

    body = model.model if hasattr(model, "model") else model
    layers = body.layers
    num_layers = len(layers)
    variances = [0.0] * num_layers
    counts = [0] * num_layers

    saved = []
    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        orig = attn.__call__

        def make_wrapped(attention_module, layer_i, original_call):
            def wrapped(x, mask=None, cache=None):
                k_lin = attention_module.k_proj(x)
                variances[layer_i] += float(np.var(np.asarray(k_lin)))
                counts[layer_i] += 1
                return original_call(x, mask=mask, cache=cache)

            return types.MethodType(wrapped, attention_module)

        attn.__call__ = make_wrapped(attn, idx, orig)
        saved.append((attn, orig))

    try:
        for text in calibration_data:
            if hasattr(tokenizer, "encode"):
                ids = tokenizer.encode(text)
            else:
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            ids = list(ids)[:max_seq_len]
            if not ids:
                continue
            x = mx.array(ids, dtype=mx.int32)[None]
            logits = model(x)
            mx.eval(logits)
    finally:
        for attn, orig in saved:
            attn.__call__ = orig

    for i in range(num_layers):
        if counts[i] > 0:
            variances[i] /= counts[i]
        else:
            variances[i] = 1.0
    return variances


def allocate_k_from_sensitivity(
    sensitivity: List[float],
    num_groups: int,
    memory_budget_total: int,
    k_min: int = 128,
    k_max: int = 576,
    num_layers: int = None,
    boundary_mode: str = "uniform",
) -> Tuple[List[int], List[int]]:
    """
    Given per-layer sensitivity, compute group boundaries and sketch width per group.

    - memory_budget_total: target surrogate B = sum_j Delta_j * m_j (paper).
    - boundary_mode: "uniform" (equal layers per group) or "percentile" (equal sensitivity mass).
    - Returns: (layer_group_boundaries, key_quantization_bits_per_group)
    """
    num_layers = num_layers or len(sensitivity)
    if boundary_mode == "percentile":
        boundaries = percentile_boundaries(sensitivity, num_groups, num_layers)
    elif boundary_mode == "uniform":
        layers_per_group = num_layers // num_groups
        boundaries = [layers_per_group * (i + 1) for i in range(num_groups - 1)]
    else:
        raise ValueError(f"unknown boundary_mode: {boundary_mode}")

    n_layers = group_layer_counts(boundaries, num_layers)

    group_sensitivity: List[float] = []
    starts = [0] + boundaries
    ends = boundaries + [num_layers]
    for g in range(num_groups):
        lo, hi = starts[g], ends[g]
        mass = sum(sensitivity[i] for i in range(lo, hi))
        group_sensitivity.append(mass / max(1, hi - lo))

    min_s = min(group_sensitivity)
    max_s = max(group_sensitivity)
    if max_s <= min_s:
        base = memory_budget_total // max(1, sum(n_layers))
        snapped = max(64, (min(base, k_max) // 64) * 64)
        k_draft = [max(k_min, min(k_max, snapped))] * num_groups
    else:
        raw_k = [
            k_min + (k_max - k_min) * (s - min_s) / (max_s - min_s)
            for s in group_sensitivity
        ]
        total_raw = sum(n_layers[i] * raw_k[i] for i in range(num_groups))
        if total_raw <= 0:
            k_draft = [k_min] * num_groups
        else:
            scale = memory_budget_total / total_raw
            k_draft = [
                max(k_min, min(k_max, int(round(raw_k[i] * scale))))
                for i in range(num_groups)
            ]

    k_per_group = repair_surrogate_budget(
        n_layers,
        k_draft,
        group_sensitivity,
        memory_budget_total,
        multiple=64,
        m_min=max(64, k_min // 64 * 64),
        m_max=k_max,
    )

    return boundaries, k_per_group


def main():
    parser = argparse.ArgumentParser(description="A-QJL sensitivity profiler")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "torch", "mlx"],
        help="mlx: Apple Silicon via mlx-lm; torch: QJL-patched HF model; auto: CUDA->torch else mac MLX->mlx else torch CPU",
    )
    parser.add_argument(
        "--torch_device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for --backend torch (ignored for mlx)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HF repo id (torch) or mlx-lm model id (mlx). Default depends on --backend.",
    )
    parser.add_argument("--dataset_name", type=str, default="qasper")
    parser.add_argument("--n_calib", type=int, default=10)
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--memory_budget", type=int, default=8000,
                        help="Target surrogate B = sum_j Delta_j * m_j (paper)")
    parser.add_argument(
        "--boundary_mode",
        type=str,
        default="uniform",
        choices=["uniform", "percentile"],
        help="uniform: equal layers per group; percentile: equal per-layer sensitivity mass per group",
    )
    parser.add_argument("--k_min", type=int, default=128, help="min sketch width (snapped to 64)")
    parser.add_argument("--k_max", type=int, default=576, help="max sketch width")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="config/aqjl_profiled.json")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    args = parser.parse_args()

    backend = _detect_backend(args.backend)
    if args.model_name is None:
        args.model_name = (
            "mlx-community/Llama-3.2-3B-Instruct-4bit"
            if backend == "mlx"
            else "lmsys/longchat-7b-v1.5-32k"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Backend: {backend}  model: {args.model_name}")
    print("Loading model and tokenizer...")
    torch_dev = None
    if backend == "mlx":
        try:
            import mlx.core as mx
            import mlx_lm  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "MLX backend requested but mlx / mlx-lm is missing. "
                "Install: pip install -r requirements-mlx.txt"
            ) from e
        model, tokenizer = setup_model_mlx(args.model_name, seed=args.seed)

        body = model.model if hasattr(model, "model") else model
        num_layers = int(
            getattr(getattr(model, "args", None), "num_hidden_layers", None)
            or getattr(getattr(model, "config", None), "num_hidden_layers", None)
            or len(body.layers)
        )
        runtime_info = {"backend": "mlx", "mlx_version": getattr(mx, "__version__", "unknown")}
    else:
        import torch

        torch.manual_seed(args.seed)
        dev = args.torch_device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = setup_model_torch(args.model_name, device=dev, seed=args.seed)
        num_layers = int(model.config.num_hidden_layers)
        torch_dev = dev
        runtime_info = {"backend": "torch", "torch_version": torch.__version__, "torch_device": dev}

    print(f"Loading calibration data from LongBench {args.dataset_name}...")
    data = load_dataset("THUDM/LongBench", f"{args.dataset_name}_e", split="test")
    texts = [
        (data[i].get("context", "")[:6000] + "\n" + data[i].get("input", ""))[:8000]
        for i in range(min(args.n_calib, len(data)))
    ]

    print("Profiling per-layer key variance...")
    if backend == "mlx":
        sensitivity = collect_key_variance_mlx(
            model, tokenizer, texts, max_seq_len=args.max_seq_len
        )
    else:
        sensitivity = collect_key_variance_torch(
            model,
            tokenizer,
            texts,
            max_seq_len=args.max_seq_len,
            device=torch_dev or "cpu",
        )

    boundaries, k_per_group = allocate_k_from_sensitivity(
        sensitivity,
        num_groups=args.num_groups,
        memory_budget_total=args.memory_budget,
        num_layers=num_layers,
        k_min=args.k_min,
        k_max=args.k_max,
        boundary_mode=args.boundary_mode,
    )

    b_achieved = surrogate_budget(boundaries, k_per_group, num_layers)

    out = {
        "layer_group_boundaries": boundaries,
        "key_quantization_bits_per_group": k_per_group,
        "outlier_count_per_group": [8] * len(k_per_group),
        "sensitivity_per_layer": [round(s, 6) for s in sensitivity],
        "memory_budget_target": args.memory_budget,
        "surrogate_B_achieved": b_achieved,
        "boundary_mode": args.boundary_mode,
        "num_hidden_layers": num_layers,
        "seed": args.seed,
        "profile_model_name": args.model_name,
        **runtime_info,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Profiled config written to {args.output}")
    print(f"  boundaries: {boundaries}")
    print(f"  k_per_group: {k_per_group}")
    print("\nRun with:")
    print(f"  --layer_group_boundaries '{','.join(map(str,boundaries))}' \\")
    print(f"  --key_quantization_bits_per_group '{','.join(map(str,k_per_group))}'")


if __name__ == "__main__":
    main()
