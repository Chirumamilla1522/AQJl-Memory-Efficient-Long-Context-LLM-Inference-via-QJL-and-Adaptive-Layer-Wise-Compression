#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RunConfig:
    name: str
    key_quantization_bits: int
    key_quantization_bits_initial_layers: int
    initial_layers_count: int
    outlier_count_general: int
    outlier_count_initial_layers: int
    value_quantization_bits: int
    group_size: int
    buffer_size: int
    # A-QJL 3+ groups (optional)
    layer_group_boundaries: Optional[List[int]] = None
    key_quantization_bits_per_group: Optional[List[int]] = None
    outlier_count_per_group: Optional[List[int]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed QJL vs adaptive layer-group QJL experiments.")
    parser.add_argument("--config", type=str, default="config/aqjl_experiments.json")
    parser.add_argument("--output_dir", type=str, default="results/runs")
    parser.add_argument("--aggregate_csv", type=str, default="results/aqjl_results.csv")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true", help="Print commands but do not execute.")
    return parser.parse_args()


def load_exp_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_run_cmd(
    python_bin: str,
    run_cfg: RunConfig,
    global_cfg: Dict,
    dataset_name: str,
    n_data: int,
    output_json: str,
) -> List[str]:
    base = [
        python_bin,
        "run_longbench.py",
        "--model_name", global_cfg["model_name"],
        "--dtype", global_cfg.get("dtype", "float16"),
        "--dataset_name", dataset_name,
        "--n_data", str(n_data),
        "--seed", str(global_cfg.get("seed", 42)),
        "--config_dir", global_cfg.get("config_dir", "config"),
        "--value_quantization_bits", str(run_cfg.value_quantization_bits),
        "--group_size", str(run_cfg.group_size),
        "--buffer_size", str(run_cfg.buffer_size),
        "--output_json", output_json,
    ]
    if run_cfg.layer_group_boundaries is not None and run_cfg.key_quantization_bits_per_group is not None:
        base += [
            "--layer_group_boundaries", ",".join(map(str, run_cfg.layer_group_boundaries)),
            "--key_quantization_bits_per_group", ",".join(map(str, run_cfg.key_quantization_bits_per_group)),
        ]
        if run_cfg.outlier_count_per_group:
            base += ["--outlier_count_per_group", ",".join(map(str, run_cfg.outlier_count_per_group))]
    else:
        base += [
            "--key_quantization_bits", str(run_cfg.key_quantization_bits),
            "--key_quantization_bits_initial_layers", str(run_cfg.key_quantization_bits_initial_layers),
            "--initial_layers_count", str(run_cfg.initial_layers_count),
            "--outlier_count_general", str(run_cfg.outlier_count_general),
            "--outlier_count_initial_layers", str(run_cfg.outlier_count_initial_layers),
        ]
    return base


def run_single(cmd: List[str], dry_run: bool = False) -> Optional[Dict]:
    print(" ".join(cmd))
    if dry_run:
        return None
    subprocess.run(cmd, check=True)
    output_json = cmd[-1]
    with open(output_json, "r", encoding="utf-8") as f:
        return json.load(f)


def dict_to_cfg(name: str, d: Dict) -> RunConfig:
    if "layer_group_boundaries" in d and "key_quantization_bits_per_group" in d:
        return RunConfig(
            name=name,
            key_quantization_bits=0,  # unused
            key_quantization_bits_initial_layers=0,
            initial_layers_count=0,
            outlier_count_general=8,
            outlier_count_initial_layers=8,
            value_quantization_bits=d.get("value_quantization_bits", 2),
            group_size=d.get("group_size", 32),
            buffer_size=d.get("buffer_size", 128),
            layer_group_boundaries=d["layer_group_boundaries"],
            key_quantization_bits_per_group=d["key_quantization_bits_per_group"],
            outlier_count_per_group=d.get("outlier_count_per_group"),
        )
    return RunConfig(
        name=name,
        key_quantization_bits=d["key_quantization_bits"],
        key_quantization_bits_initial_layers=d["key_quantization_bits_initial_layers"],
        initial_layers_count=d["initial_layers_count"],
        outlier_count_general=d.get("outlier_count_general", 8),
        outlier_count_initial_layers=d.get("outlier_count_initial_layers", 8),
        value_quantization_bits=d.get("value_quantization_bits", 2),
        group_size=d.get("group_size", 32),
        buffer_size=d.get("buffer_size", 128),
    )


def maybe_calibrate_adaptive(
    exp_cfg: Dict,
    global_cfg: Dict,
    args: argparse.Namespace,
) -> Dict:
    adaptive_cfg = exp_cfg["adaptive_qjl"]
    calib = adaptive_cfg.get("calibration", {})

    # Optionally run sensitivity profiler first
    if calib.get("run_profiler_first") and not args.dry_run:
        profiler_out = calib.get("profiler_output", "config/aqjl_profiled.json")
        print("Running sensitivity profiler...")
        subprocess.run(
            [
                args.python_bin,
                "scripts/sensitivity_profiler.py",
                "--model_name", global_cfg["model_name"],
                "--dataset_name", calib.get("dataset_name", global_cfg["datasets"][0]),
                "--n_calib", str(calib.get("n_calib", 10)),
                "--output", profiler_out,
            ],
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        with open(profiler_out, "r") as f:
            profiled = json.load(f)
        return {
            "layer_group_boundaries": profiled["layer_group_boundaries"],
            "key_quantization_bits_per_group": profiled["key_quantization_bits_per_group"],
            "outlier_count_per_group": profiled.get("outlier_count_per_group"),
            "value_quantization_bits": 2,
            "group_size": 32,
            "buffer_size": 128,
        }

    candidates = adaptive_cfg.get("candidates", [])
    if not adaptive_cfg.get("enabled", True) or not candidates:
        return adaptive_cfg["selected"]

    calib_dataset = calib.get("dataset_name", global_cfg["datasets"][0])
    calib_n_data = int(calib.get("n_data", 20))
    memory_budget_gb = float(calib.get("memory_budget_gb", 0.0))
    if memory_budget_gb <= 0:
        return adaptive_cfg["selected"]

    os.makedirs(args.output_dir, exist_ok=True)
    best = None
    for idx, cand in enumerate(candidates):
        cfg = dict_to_cfg(f"aqjl_candidate_{idx}", cand)
        out_json = os.path.join(args.output_dir, f"calib_{cfg.name}.json")
        cmd = build_run_cmd(
            args.python_bin,
            cfg,
            global_cfg,
            calib_dataset,
            calib_n_data,
            out_json,
        )
        metrics = run_single(cmd, dry_run=args.dry_run)
        if args.dry_run:
            continue
        if metrics["peak_memory_gb"] <= memory_budget_gb:
            if best is None or metrics["avg_score"] > best["avg_score"]:
                best = {"candidate": cand, "avg_score": metrics["avg_score"]}

    if args.dry_run:
        return adaptive_cfg["selected"]
    if best is None:
        print("No candidate fit memory budget during calibration. Falling back to selected config.")
        return adaptive_cfg["selected"]
    print(f"Calibration selected candidate with avg_score={best['avg_score']:.4f}")
    return best["candidate"]


def write_aggregate(csv_path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fields = [
        "method",
        "dataset_name",
        "avg_score",
        "peak_memory_gb",
        "total_eval_time_sec",
        "tokens_per_sec_estimate",
        "n_data",
        "key_quantization_bits",
        "key_quantization_bits_initial_layers",
        "initial_layers_count",
        "layer_group_boundaries",
        "key_quantization_bits_per_group",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            if isinstance(out.get("layer_group_boundaries"), list):
                out["layer_group_boundaries"] = ",".join(map(str, out["layer_group_boundaries"]))
            if isinstance(out.get("key_quantization_bits_per_group"), list):
                out["key_quantization_bits_per_group"] = ",".join(map(str, out["key_quantization_bits_per_group"]))
            writer.writerow(out)
    print(f"Wrote aggregate CSV: {csv_path}")


def main() -> None:
    args = parse_args()
    exp_cfg = load_exp_config(args.config)
    global_cfg = exp_cfg["global"]
    os.makedirs(args.output_dir, exist_ok=True)

    fixed_cfg = dict_to_cfg("qjl_fixed", exp_cfg["fixed_qjl"])
    adaptive_selected = maybe_calibrate_adaptive(exp_cfg, global_cfg, args)
    adaptive_cfg = dict_to_cfg("aqjl", adaptive_selected)

    rows = []
    for dataset_name in global_cfg["datasets"]:
        for run_cfg in [fixed_cfg, adaptive_cfg]:
            out_json = os.path.join(args.output_dir, f"{run_cfg.name}_{dataset_name}.json")
            cmd = build_run_cmd(
                args.python_bin,
                run_cfg,
                global_cfg,
                dataset_name,
                int(global_cfg.get("n_data", 100)),
                out_json,
            )
            metrics = run_single(cmd, dry_run=args.dry_run)
            if args.dry_run:
                continue
            row = {
                "method": run_cfg.name,
                "dataset_name": dataset_name,
                **metrics,
                "key_quantization_bits": run_cfg.key_quantization_bits,
                "key_quantization_bits_initial_layers": run_cfg.key_quantization_bits_initial_layers,
                "initial_layers_count": run_cfg.initial_layers_count,
                "layer_group_boundaries": run_cfg.layer_group_boundaries,
                "key_quantization_bits_per_group": run_cfg.key_quantization_bits_per_group,
            }
            rows.append(row)

    if not args.dry_run:
        write_aggregate(args.aggregate_csv, rows)


if __name__ == "__main__":
    main()
