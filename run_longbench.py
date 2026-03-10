import os
import argparse
import random
import time
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset
from eval_long_bench import dataset2metric
from fastchat.model import get_conversation_template
from models.llama2_utils_qjl import QJLSketch
from models.llama2_qjl import LlamaForCausalLM_QJL


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def build_chat(prompt, model_name):
    if "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "longchat" in model_name or "vicuna" in model_name:
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    else:
        raise NotImplementedError
    return prompt


def setup_model_and_tokenizer(
        model_name,
        dtype=torch.float16,
        key_quantization_bits=256,
        key_quantization_bits_initial_layers=512,
        initial_layers_count=15,
        outlier_count_general=8,
        outlier_count_initial_layers=8,
        value_quantization_bits=2,
        group_size=32,
        buffer_size=128,
        layer_group_boundaries=None,
        key_quantization_bits_per_group=None,
        outlier_count_per_group=None,
):
    device = 'cuda'
    config = LlamaConfig.from_pretrained(model_name)
    config._flash_attn_2_enabled = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        tokenizer_type='llama'
    )

    config = LlamaConfig.from_pretrained(model_name)
    config.attention_dropout = 0.0
    config.key_quantization_bits = key_quantization_bits
    config.key_quantization_bits_initial_layers = key_quantization_bits_initial_layers
    config.initial_layers_count = initial_layers_count

    config.outlier_count_general = outlier_count_general
    config.outlier_count_initial_layers = outlier_count_initial_layers

    config.value_quantization_bits = value_quantization_bits
    config.group_size = group_size
    config.buffer_size = buffer_size

    generator = torch.Generator(device=torch.device(device))

    # A-QJL: 3+ layer groups
    if layer_group_boundaries is not None and key_quantization_bits_per_group is not None:
        num_layers = config.num_hidden_layers
        assert len(layer_group_boundaries) + 1 == len(key_quantization_bits_per_group), \
            f"layer_group_boundaries ({len(layer_group_boundaries)}) + 1 must equal len(key_quantization_bits_per_group) ({len(key_quantization_bits_per_group)})"
        oc_per_group = outlier_count_per_group if outlier_count_per_group else [8] * len(key_quantization_bits_per_group)
        config.layer_group_boundaries = layer_group_boundaries
        config.qjl_groups = []
        for g, k in enumerate(key_quantization_bits_per_group):
            oc = oc_per_group[g] if g < len(oc_per_group) else 8
            dim_outlier = 256 if k >= 256 else 128
            qjl = QJLSketch(dim=(128, k), dim_outlier=dim_outlier, rot=True, rng=generator)
            config.qjl_groups.append((qjl, oc, k))
        config.qjl = None  # unused in multi-group
        config.qjl_initial_layers = None
    else:
        config.layer_group_boundaries = None
        config.qjl_groups = None
        config.qjl = QJLSketch(dim=(128, config.key_quantization_bits), dim_outlier=256, rot=True, rng=generator)
        config.qjl_initial_layers = QJLSketch(dim=(128, config.key_quantization_bits_initial_layers), dim_outlier=128,
                                                  rot=True,
                                                  rng=generator)

    config.use_flash = True

    model_qjl = LlamaForCausalLM_QJL.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=config,
        cache_dir=None,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    return model_qjl, tokenizer


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="lmsys/longchat-7b-v1.5-32k")
    parser.add_argument('--dtype', type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument('--key_quantization_bits', type=int, default=256)
    parser.add_argument('--key_quantization_bits_initial_layers', type=int, default=512)
    parser.add_argument('--initial_layers_count', type=int, default=15)
    parser.add_argument('--outlier_count_general', type=int, default=8)
    parser.add_argument('--outlier_count_initial_layers', type=int, default=8)
    parser.add_argument('--value_quantization_bits', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--buffer_size', type=int, default=128)
    parser.add_argument('--layer_group_boundaries', type=str, default=None,
                        help="A-QJL 3+ groups: comma-separated boundaries, e.g. '8,16,24' for 4 groups.")
    parser.add_argument('--key_quantization_bits_per_group', type=str, default=None,
                        help="A-QJL: comma-separated k per group, e.g. '512,384,256,192'.")
    parser.add_argument('--outlier_count_per_group', type=str, default=None,
                        help="A-QJL: comma-separated outlier counts per group (optional).")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--n_data', type=int, default=150)
    parser.add_argument('--config_dir', type=str, default="config")
    parser.add_argument('--output_json', type=str, default=None,
                        help="Optional path to save run metrics as JSON.")
    return parser.parse_args(args)


def load_configurations(config_dir):
    with open(os.path.join(config_dir, 'dataset2maxlen.json')) as f:
        dataset2maxlen = json.load(f)
    with open(os.path.join(config_dir, 'dataset2prompt.json')) as f:
        dataset2prompt = json.load(f)
    with open(os.path.join(config_dir, 'model2maxlen.json')) as f:
        model2maxlen = json.load(f)

    return dataset2maxlen, dataset2prompt, model2maxlen


def evaluate_model(
        model_qjl,
        tokenizer,
        dataset_name,
        dataset2maxlen,
        dataset2prompt,
        model2maxlen,
        n_data=150,
):
    device = 'cuda'
    prompt_format = dataset2prompt.get(dataset_name,
                                       "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:")
    max_length = dataset2maxlen.get(dataset_name, 31500)
    max_gen = model2maxlen.get(dataset_name, 64)

    data = load_dataset('THUDM/LongBench', f"{dataset_name}_e", split='test')
    total_score = 0.
    aa = []
    start = time.time()
    mem_peak_max = 0.0

    for i in tqdm(range(n_data), desc="Evaluating"):
        json_obj = data[i]
        prompt = prompt_format.format(**json_obj)

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
                                "repobench-p"]:
            prompt = build_chat(prompt, model_qjl.config.name_or_path)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        output = model_qjl.generate(
            **input,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        ground_truths = json_obj['answers']
        all_classes = json_obj['all_classes']
        prediction = pred

        score = 0.
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset_name](prediction, ground_truth, all_classes=all_classes))

        total_score += score

        mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        mem_reserve = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
        mem_peak = torch.cuda.memory_stats()['active_bytes.all.peak'] / 1024 / 1024 / 1024
        mem_peak_max = max(mem_peak_max, mem_peak)

        mem_info = f"mem_alloc: {mem_alloc:.5f}, mem_reserved: {mem_reserve:.5f}, mem_peak: {mem_peak:.5f}"
        aa.append(score)
        print(f"[{i:>4}] score: {score:.4f}, avg_score: {total_score / (i + 1):.4f}, | {mem_info}")

    avg_score = float(np.mean(aa))
    total_time = float(time.time() - start)
    print(f"Average score for dataset {dataset_name}: {avg_score}")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    return {
        "dataset_name": dataset_name,
        "avg_score": avg_score,
        "n_data": n_data,
        "peak_memory_gb": float(mem_peak_max),
        "total_eval_time_sec": total_time,
        "tokens_per_sec_estimate": float(n_data / total_time) if total_time > 0 else 0.0,
    }


def _parse_int_list(s, default=None):
    if s is None:
        return default
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(args):
    seed_everything(args.seed)
    dataset2maxlen, dataset2prompt, model2maxlen,  = load_configurations(args.config_dir)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    layer_bounds = _parse_int_list(args.layer_group_boundaries)
    k_per_group = _parse_int_list(args.key_quantization_bits_per_group)
    oc_per_group = _parse_int_list(args.outlier_count_per_group)

    model_qjl, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        dtype,
        args.key_quantization_bits,
        args.key_quantization_bits_initial_layers,
        args.initial_layers_count,
        args.outlier_count_general,
        args.outlier_count_initial_layers,
        args.value_quantization_bits,
        args.group_size,
        args.buffer_size,
        layer_group_boundaries=layer_bounds,
        key_quantization_bits_per_group=k_per_group,
        outlier_count_per_group=oc_per_group,
    )
    print(f"Model and tokenizer for {args.model_name} are set up successfully.")
    metrics = evaluate_model(
        model_qjl,
        tokenizer,
        args.dataset_name,
        dataset2maxlen,
        dataset2prompt,
        model2maxlen,
        args.n_data,
    )
    print("RUN_METRICS_JSON::" + json.dumps(metrics))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved run metrics JSON to: {args.output_json}")


if __name__ == "__main__":
    args = parse_args()
    main(args)