#!/usr/bin/env python3
"""
A-QJL Sensitivity Profiler: estimates per-layer sensitivity to compression.

Uses key magnitude variance as a proxy for sensitivity. Layers with higher
key variance are assigned higher projection dimension (k); layers with lower
variance get lower k, under a fixed memory budget.

Output: layer_group_boundaries, key_quantization_bits_per_group suitable
for run_longbench.py --layer_group_boundaries ... --key_quantization_bits_per_group ...
"""

import argparse
import json
import os
from typing import List, Tuple

import torch
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset

from models.llama2_utils_qjl import QJLSketch
from models.llama2_qjl import LlamaForCausalLM_QJL


def setup_model(model_name: str, dtype=torch.float16, device="cuda"):
    """Load model in 2-group mode for profiling (we only need key statistics)."""
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

    generator = torch.Generator(device=torch.device(device))
    config.qjl = QJLSketch(dim=(128, 256), dim_outlier=0, rot=True, rng=generator)
    config.qjl_initial_layers = QJLSketch(dim=(128, 256), dim_outlier=0, rot=True, rng=generator)
    config.use_flash = True

    model = LlamaForCausalLM_QJL.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, trust_remote_code=True, tokenizer_type="llama"
    )
    return model, tokenizer


def collect_key_variance_per_layer(
    model,
    tokenizer,
    calibration_data: List[str],
    max_seq_len: int = 1024,
    device: str = "cuda",
) -> List[float]:
    """
    Run forward passes and collect per-layer key variance.
    Returns sensitivity scores (higher = more sensitive to compression).
    """
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
            ).to(device)
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


def allocate_k_from_sensitivity(
    sensitivity: List[float],
    num_groups: int,
    memory_budget_total: int,
    k_min: int = 128,
    k_max: int = 576,
    num_layers: int = None,
) -> Tuple[List[int], List[int]]:
    """
    Given per-layer sensitivity, compute group boundaries and k per group.

    - sensitivity: list of scores (higher = more sensitive)
    - num_groups: e.g. 4
    - memory_budget_total: target sum of (layers_in_group * k_group) - simplified proxy
    - Returns: (layer_group_boundaries, key_quantization_bits_per_group)
    """
    num_layers = num_layers or len(sensitivity)
    layers_per_group = num_layers // num_groups
    boundaries = [layers_per_group * (i + 1) for i in range(num_groups - 1)]

    # Mean sensitivity per group
    group_sensitivity = []
    for g in range(num_groups):
        start = 0 if g == 0 else boundaries[g - 1]
        end = boundaries[g] if g < len(boundaries) else num_layers
        s = sum(sensitivity[i] for i in range(start, end)) / max(1, end - start)
        group_sensitivity.append(s)

    # Allocate k proportionally to sensitivity, normalized to meet budget
    # Budget ~ sum( layers_in_group * k_group ). Simplified: B = L * k_avg, so k_avg = B/L.
    # We want k_i proportional to sens_i, and sum(n_i * k_i) = budget.
    # n_i = layers in group i. sum(n_i * k_i) = budget.
    # k_i = base + scale * sens_i. Solve for scale given budget.
    total_weight = sum(
        (boundaries[0] if g == 0 else boundaries[g] - boundaries[g - 1])
        for g in range(num_groups)
    )
    # Actually each group has n_i layers. n_0 = boundaries[0], n_1 = boundaries[1]-boundaries[0], ...
    n_layers = []
    for g in range(num_groups):
        start = 0 if g == 0 else boundaries[g - 1]
        end = boundaries[g] if g < len(boundaries) else num_layers
        n_layers.append(end - start)

    # k_i = k_min + (k_max - k_min) * (sens_i - min_s) / (max_s - min_s)  then scale to budget
    min_s = min(group_sensitivity)
    max_s = max(group_sensitivity)
    if max_s <= min_s:
        k_per_group = [memory_budget_total // num_layers] * num_groups
    else:
        raw_k = [
            k_min + (k_max - k_min) * (s - min_s) / (max_s - min_s)
            for s in group_sensitivity
        ]
        # Scale to meet budget: sum(n_i * k_i) = budget => scale = budget / sum(n_i * raw_k_i)
        total_raw = sum(n_layers[i] * raw_k[i] for i in range(num_groups))
        if total_raw <= 0:
            k_per_group = [k_min] * num_groups
        else:
            scale = memory_budget_total / total_raw
            k_per_group = [
                max(k_min, min(k_max, int(round(raw_k[i] * scale))))
                for i in range(num_groups)
            ]
            # Snap to multiples of 64 for compatibility
            k_per_group = [max(64, (k // 64) * 64) for k in k_per_group]

    return boundaries, k_per_group


def main():
    parser = argparse.ArgumentParser(description="A-QJL sensitivity profiler")
    parser.add_argument("--model_name", type=str, default="lmsys/longchat-7b-v1.5-32k")
    parser.add_argument("--dataset_name", type=str, default="qasper")
    parser.add_argument("--n_calib", type=int, default=10)
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--memory_budget", type=int, default=8000,
                        help="Approx total layers*k for allocation (proxy)")
    parser.add_argument("--output", type=str, default="config/aqjl_profiled.json")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model, tokenizer = setup_model(args.model_name)
    num_layers = model.config.num_hidden_layers

    print(f"Loading calibration data from LongBench {args.dataset_name}...")
    data = load_dataset("THUDM/LongBench", f"{args.dataset_name}_e", split="test")
    texts = [
        (data[i].get("context", "")[:6000] + "\n" + data[i].get("input", ""))[:8000]
        for i in range(min(args.n_calib, len(data)))
    ]

    print("Profiling per-layer key variance...")
    sensitivity = collect_key_variance_per_layer(
        model, tokenizer, texts, max_seq_len=args.max_seq_len
    )

    boundaries, k_per_group = allocate_k_from_sensitivity(
        sensitivity,
        num_groups=args.num_groups,
        memory_budget_total=args.memory_budget,
        num_layers=num_layers,
    )

    out = {
        "layer_group_boundaries": boundaries,
        "key_quantization_bits_per_group": k_per_group,
        "outlier_count_per_group": [8] * len(k_per_group),
        "sensitivity_per_layer": [round(s, 6) for s in sensitivity],
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
