# A-QJL Results Guide

This directory stores all experiment outputs for the fixed-QJL vs adaptive-QJL comparison.

## What is in this folder

- `sample_aqjl_results.csv`  
  Example results table showing the expected CSV format.
- `plots_sample/`  
  Example plots and a markdown summary generated from the sample CSV.
- `runs/`  
  Per-run JSON metrics produced by `scripts/aqjl_experiments.py`.

## Prerequisites

Run commands from the project root (`QJL-AQJL/`), not from this folder.

- CUDA-capable GPU machine
- QJL kernel compiled in `qjl_kernel/`
- Python dependencies installed

Kernel build command:

```bash
cd qjl_kernel
python setup.py build_ext --inplace
cd ..
```

## Recommended workflow

### 1) Check experiment commands (safe dry-run)

```bash
python scripts/aqjl_experiments.py --dry_run --config config/aqjl_experiments.json
```

This prints all runs without executing model inference.

### 2) Run experiments

```bash
python scripts/aqjl_experiments.py --config config/aqjl_experiments.json
```

Outputs:

- `results/runs/*.json` (per dataset + method)
- `results/aqjl_results.csv` (aggregated metrics table)

### 3) Generate plots and summary

```bash
python scripts/plot_aqjl_results.py --input_csv results/aqjl_results.csv --out_dir results/plots
```

Outputs:

- `results/plots/avg_score.png`
- `results/plots/peak_memory_gb.png`
- `results/plots/tokens_per_sec.png`
- `results/plots/summary.md`

## Metrics reported

- `avg_score`: task quality score from LongBench evaluation
- `peak_memory_gb`: peak active CUDA memory during run
- `tokens_per_sec_estimate`: throughput estimate (`n_data / total_eval_time_sec`)

## Quick interpretation

- A-QJL is better if it achieves **higher `avg_score` at similar or lower `peak_memory_gb`**.
- If memory is matched, compare `avg_score` first, then throughput.

## Troubleshooting

- `ImportError ... cuda_qjl_quant`  
  Build the CUDA extension in `qjl_kernel/` (`python setup.py build_ext --inplace`).
- `CUDA_HOME environment variable is not set`  
  CUDA toolkit is not available/configured on the machine.
- macOS Apple Silicon without CUDA  
  Full QJL kernel runs are not supported; use dry-run and sample plots only.
