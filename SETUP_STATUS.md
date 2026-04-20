# Setup / Current Status

## Current setup progress

We were able to get most of the base environment working in WSL.

Completed so far:
- Installed WSL Ubuntu
- Confirmed GPU is visible in WSL
- Installed PyTorch with CUDA support
- Compiled the custom `qjl_kernel` successfully
- Verified that the experiment driver `aqjl_experiments.py --dry_run` works

## Current issue

The main blocker right now is not the base environment anymore.

The current problem is a compatibility issue between the repo’s custom `llama2_qjl.py` implementation and the installed `transformers` version when trying to run the actual smoke test. So the setup is mostly working, but the full model run is still not clean yet.

## What was tested

These parts were confirmed:
- CUDA works in WSL
- `torch.cuda.is_available()` returns `True`
- the custom CUDA kernel builds successfully
- experiment dry-run prints the expected commands

## Current experiment direction

Our current plan is still:
- replicate the existing results
- compare fixed QJL vs A-QJL
- try adaptive dimensions for different layer groups

For now, we can show the current progress/results and mention that we are still working on improving the final runs.

## Notes

Because this repo seems sensitive to package versions, the next step is probably to either:
1. align the repo with the correct dependency versions, or
2. continue testing on a cleaner environment / different machine if available
