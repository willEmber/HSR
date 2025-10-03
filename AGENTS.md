# Repository Guidelines

## Project Structure & Module Organization
- `main/`: entry points for training (`train.py`, `train_1.5.py`) and evaluation (`benchmark.py`).
- `models/`, `models/lib/`: model definitions (AIDN adaptations, RevealNet, blocks).
- `dataset/`: DIV2K loaders, transforms, and bicubic utilities.
- `config/`: YAML configs (e.g., `config/DIV2K/AIDN.yaml`).
- `base/`: config loader, logging, trainer helpers.
- `utils/`, `metrics/`: I/O, image helpers, PSNR/SSIM, losses.
- `scripts/`: shell wrappers for train/benchmark.
- `Data/`, `LOG/`, `assets/`: data roots, experiment outputs, figures.
- `DeepMIH/`: DeepMIH repository code for the Importance Map referenced in this project; refer to some of the code in this directory when implementing related features.

## Build, Test, and Development Commands
- Create env and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Train (recommended):
  - `bash scripts/train.sh EXP-1.5x3 config/DIV2K/AIDN.yaml`
- Train (direct):
  - `python main/train_1.5.py --config config/DIV2K/AIDN.yaml save_path LOG/DIV2K/exp1`
- Benchmark saved checkpoint:
  - `bash scripts/benchmark.sh EXP-1.5x3 config/DIV2K/AIDN_benchmark.yaml`
- Pretrained AIDN benchmark:
  - `bash scripts/AIDN_benchmark.sh config/DIV2K/AIDN_benchmark.yaml`

## Coding Style & Naming Conventions
- Python 3.8+; 4‑space indentation; follow PEP8; add type hints for new code.
- Naming: modules/functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Config keys use `lower_snake_case`; keep YAML minimal and documented.
- Prefer relative imports; avoid hard‑coded absolute paths. Set `PYTHONPATH=./` when needed.

## Testing Guidelines
- Functional checks use `main/benchmark.py` with DIV2K/Set5 lists under `Data/list`.
- Add unit tests for utilities where feasible; place beside modules as `test_*.py`.
- Validate metrics via `metrics/psnr.py`, `metrics/ssim.py`; include a small sample in PR if adding new losses/metrics.

## Commit & Pull Request Guidelines
- Commit messages: `type(scope): short summary`.
  - Examples: `feat(train): support 1.5x 3-loop`, `fix(metrics): correct SSIM window`.
- PRs should include:
  - What changed and why; affected configs.
  - Exact run commands and a brief log/metric excerpt (PSNR/SSIM).
  - Paths to outputs under `LOG/...` and any new data list files.
  - Link related issues; add screenshots if UI/figures change.

## Security & Configuration Tips
- Do not commit large datasets, checkpoints, or zips; store under `Data/` and `LOG/` locally and `.gitignore` them.
- Keep secrets/paths out of code; use YAML config and CLI overrides (`opts`).
- Ensure CUDA/GPU visibility is controlled via env vars, not hard‑coded.


