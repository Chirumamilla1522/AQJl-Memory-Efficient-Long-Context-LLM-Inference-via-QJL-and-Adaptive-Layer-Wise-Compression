#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot A-QJL experiment results from aggregate CSV.")
    parser.add_argument("--input_csv", type=str, default="results/aqjl_results.csv")
    parser.add_argument("--out_dir", type=str, default="results/plots")
    return parser.parse_args()


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["avg_score"] = float(row["avg_score"])
            row["peak_memory_gb"] = float(row["peak_memory_gb"])
            row["tokens_per_sec_estimate"] = float(row["tokens_per_sec_estimate"])
            rows.append(row)
    return rows


def plot_metric(rows, metric, ylabel, out_path):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["dataset_name"]][row["method"]] = row[metric]

    datasets = sorted(grouped.keys())
    methods = sorted({r["method"] for r in rows})
    x = list(range(len(datasets)))
    width = 0.35 if len(methods) == 2 else 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = [grouped[d].get(method, 0.0) for d in datasets]
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar([xx + offset for xx in x], vals, width=width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


def write_summary(rows, out_dir):
    datasets = sorted({r["dataset_name"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    by_ds = defaultdict(dict)
    for r in rows:
        by_ds[r["dataset_name"]][r["method"]] = r

    lines = []
    lines.append("# A-QJL Results Summary")
    lines.append("")
    lines.append("| Dataset | Method | Avg Score | Peak Memory (GB) | Tokens/sec (est.) |")
    lines.append("|---|---:|---:|---:|---:|")
    for d in datasets:
        for m in methods:
            rr = by_ds[d].get(m)
            if rr is None:
                continue
            lines.append(
                f"| {d} | {m} | {rr['avg_score']:.4f} | {rr['peak_memory_gb']:.4f} | {rr['tokens_per_sec_estimate']:.4f} |"
            )
    lines.append("")
    lines.append("## Delta (A-QJL - Fixed)")
    lines.append("")
    lines.append("| Dataset | Delta Score | Delta Memory (GB) | Delta Tokens/sec |")
    lines.append("|---|---:|---:|---:|")
    for d in datasets:
        fixed = by_ds[d].get("qjl_fixed")
        aqjl = by_ds[d].get("aqjl")
        if fixed and aqjl:
            lines.append(
                f"| {d} | {aqjl['avg_score'] - fixed['avg_score']:+.4f} | "
                f"{aqjl['peak_memory_gb'] - fixed['peak_memory_gb']:+.4f} | "
                f"{aqjl['tokens_per_sec_estimate'] - fixed['tokens_per_sec_estimate']:+.4f} |"
            )
    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {summary_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_rows(args.input_csv)
    plot_metric(rows, "avg_score", "Average Score", os.path.join(args.out_dir, "avg_score.png"))
    plot_metric(rows, "peak_memory_gb", "Peak Memory (GB)", os.path.join(args.out_dir, "peak_memory_gb.png"))
    plot_metric(rows, "tokens_per_sec_estimate", "Tokens/sec (Estimate)", os.path.join(args.out_dir, "tokens_per_sec.png"))
    write_summary(rows, args.out_dir)


if __name__ == "__main__":
    main()
