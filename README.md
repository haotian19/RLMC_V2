# RLMC v2

This repository contains the complete experimental pipeline for the RLMC v2 paper:
data preprocessing, model training (three configurations), machine-learning baselines,
count ablation, result aggregation.

## Paper correspondence

| File | Manuscript section |
|---|---|
| `preprocess.py` | 2.2 (data checks; prints prevalence, combination counts) |
| `rlmc.py` | 2.4 (model architecture: base / conv / attn) |
| `utils.py` | 2.5 / 3.3 (early stopping, metrics, attention export) |
| `main.py` | 2.2 / 2.5 / 3.1 / 3.2 (RLMC v2 training) |
| `main_ml.py` | 2.3 / 3.2 (baselines) |
| `evaluate.py` | 3.4 (SHAP computation for Figure 5) |
| `merge_results.py` | aggregates all runs into `paper_data.xlsx` |
| `test.py` | standalone checkpoint re-evaluation (3.2) |
| `run_all.py` | one-command orchestration of the full pipeline |

## Key design points (matching the paper)

1. **Grouped splitting** — `GroupShuffleSplit` by the 12 risk-factor combinations;
   clean 80%/10%/10% train/validation/test over three seeds (22/23/24).
2. **SMOTENC** — categorical-aware oversampling fitted on the training partition only.
3. **Single balancing mechanism** — class-weighted loss is not combined with SMOTENC.
4. **Working early stopping** — persistent counter, patience 10, on validation loss;
   best checkpoint restored (matches Section 2.5).
5. **Validation-based threshold calibration** — `--tune_threshold` selects the F1-maximizing
   threshold on the validation partition per seed; applied to the test partition only.
6. **count ablation** — `--count_mode feature|drop|weight`.
7. Metrics are reported as mean ± SD across seeds; ML baselines additionally export
   bootstrap 95% CIs. (APEI was removed from the paper and from the code.)

## Installation

```bash
pip install torch scikit-learn imblearn xgboost lightgbm openpyxl shap pandas
# R side: install.packages(c("tidyverse", "readxl"))   # optional: ggbeeswarm, gt
```

## One-command run

```bash
python run_all.py --tune_threshold                 # full pipeline: 3 modes + ablation + ML + SHAP
python run_all.py --tune_threshold --skip_ablation # skip the count ablation (faster)
python run_all.py --tune_threshold --skip_shap     # skip SHAP (slowest stage)
python run_all.py --quick                          # smoke test (1 epoch)
```

`run_all.py` flags: `--n_seeds` (default 3), `--num_epochs` (100), `--batch_size` (512),
`--modes "attn base conv"`, `--resume` (resume interrupted seeds from checkpoints),
`--output_dir` (default `output_fixed`).

## Stepwise run (equivalent)

```bash
python preprocess.py                                   # risk.txt -> clean_data.csv (run once)
python main.py    --mode attn --n_seeds 3 --count_mode feature --run_attn_viz --run_shap --tune_threshold
python main.py    --mode base --n_seeds 3 --count_mode feature --tune_threshold
python main.py    --mode conv --n_seeds 3 --count_mode feature --tune_threshold
python main.py    --mode attn --n_seeds 3 --count_mode drop    --tune_threshold
python main.py    --mode attn --n_seeds 3 --count_mode weight  --tune_threshold
python main_ml.py --n_seeds 3 --count_mode feature --tune_threshold
python main_ml.py --n_seeds 3 --count_mode drop    --tune_threshold
python merge_results.py --root output_fixed --out paper_data.xlsx
```

## Outputs

Each run directory (e.g., `output_fixed/attn_feature/`) contains:

- `per_seed_scores.csv`, `summary_mean_std.csv` — Table 2 material
- `training_report_<mode>_seed<s>.csv`, `training_curves.csv` — Figure 2 data
- `roc_points_<mode>_seed<s>.csv` — Figure 3 ROC/PR curve coordinates (small, merged into Excel)
- `test_pred_<mode>_seed<s>.csv` — raw per-sample predictions (kept on disk; not merged)
- `attention_long_<mode>.csv` — Figure 4 data (with `--run_attn_viz`)
- `shap_tidy_<mode>.csv`, `shap_importance_<mode>.csv` — Figure 5 data (with `--run_shap`)
- ML runs additionally: `summary_tidy.csv`, `bootstrap_ci_last_seed.csv`
- `checkpoint_<mode>_<count_mode>_seed<s>.pth` — resumable checkpoints (every epoch)

## Excel workbook (`paper_data.xlsx`, via `merge_results.py`)

Sheets map one-to-one to manuscript items:
`Table2_deep_per_seed`, `Table2_deep_summary`, `Table2_ml_per_seed`, `Table2_ml_summary`,
`Table2_ml_CI`, `Fig2_training_curves`, `Fig3_roc_curves`, `Fig4_attention`,
`Fig5_shap`, `Fig5_shap_importance`, `configs`.
Pass `--include_predictions` to also merge the raw prediction files (large).
