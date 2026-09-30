# SAVE-LEWS: physics-guided prediction from low-cost IoT soil-moisture sensors for landslide early warning

Code, data and stored results for the manuscript

> **Physics-guided prediction from low-cost IoT soil-moisture sensors for landslide early warning: matching the level of physics to what the sensor observes.** [Author 1], [Author 2], [Author 3], Md Eshrat E Alahi. Submitted to *Landslides* (Springer), 2026.

The repository lets anyone regenerate every figure and table of the manuscript from the raw sensor data, and re-run every model, in Google Colab or on a laptop.

## What the study does

**SAVE** (**S**creening–**A**ttribution–**V**alidation for **E**arly warning) is a four-step framework that matches the amount of physics in a prediction model to what a low-cost sensor configuration can observe:

| Step | What it does | Where in this repository |
|---|---|---|
| I. Screening | Green–Ampt travel time, water-balance bound d_max = P/Δθ, lag test, sensor comparability (Se normalisation), using sensor data only | `tables/TableS1_*`, `puerto_rico/p2_lag.py`, `screening_metrics*.py`, `figures/sensor_compare.py` |
| II. Attribution | residual-to-tendency ratio ρ_R, matrix share s_matrix, independent implicit Richards solver, no-physics ablation | `thailand/richards_pinn.py`, `richards_solver.py`, `train_loso.py`, `solver_loso.py` |
| III. Storm-blocked validation | leave-one-storm-out (Thailand) and leave-one-block-out with 24 h purge (Puerto Rico), storm bootstrap, pre-registration | `thailand/run_loso.py`, `bootstrap_table6.py`, `puerto_rico/PREREGISTRATION.md`, `eval_p1.py` |
| IV. Hydrological alert scoring | POD, FAR, CSI, timely detection within ±2 h, calibrated 90 % bands (conformal vs Bayesian) | `thailand/thai_alerts.py`, `h2_compare.py`, `puerto_rico/pr_alerts_rise.py`, `eval_p4.py` |

Models along the physics-guidance ladder: unconstrained MLP (U), bounds-only MLP (B), physics-guided MLP (PG-MLP), Bayesian two-store model (B-TS, NumPyro), Richards PINNs M0–M3 (PyTorch), and a calibrated 1-D Richards solver (R-cal, R-lab, R-cal-Se; JAX).

## Quick start in Google Colab

1. Create the GitHub repository (see *Publishing this repository* below) and edit the line `REPO_URL = ...` in the first code cell of each notebook, or open the notebooks directly from GitHub in Colab (`File → Open notebook → GitHub`).
2. Open **`notebooks/00_reproduce_all_figures_and_tables.ipynb`** and choose `Runtime → Run all`. It clones the repository, installs NumPyro and Optax, and writes all figures to `figures/output/` and all tables to `tables/` in about 2 minutes. Nothing is trained.
3. To re-run the models, open notebooks 01–06. Each loads stored results by default; set the flags in its first cell (`TRAIN`, `RUN_SOLVER`, `RETRAIN`, `RERUN_CHECK`, `RUN_ALL`, `RERUN_*`) to recompute.

Locally:

```bash
git clone https://github.com/YOUR-ACCOUNT/SAVE-LEWS.git && cd SAVE-LEWS
pip install -r requirements.txt
python figures/make_all_figures.py      # Figs 1-15, S1-S3 -> figures/output/
python tables/make_tables.py            # table CSVs -> tables/
python figures/sensor_compare.py        # Table 3 (sensor systems)
```

## Notebooks

| Notebook | Open in Colab | Content | Default run (stored results) | Full re-run |
|---|---|---|---|---|
| `00_reproduce_all_figures_and_tables` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-ACCOUNT/SAVE-LEWS/blob/main/notebooks/00_reproduce_all_figures_and_tables.ipynb) | all figures and tables, checksum and pre-registration checks | ~1 min | – |
| `01_thailand_step1_richards_pinn` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-ACCOUNT/SAVE-LEWS/blob/main/notebooks/01_thailand_step1_richards_pinn.ipynb) | data pipeline (rain counter, temperature correction), water-balance screen, Richards PINN A/B/C, residual budget, independent solver | ~2 min | `TRAIN=True`, ~45 min |
| `02_thailand_step2_consistency_vs_skill` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-ACCOUNT/SAVE-LEWS/blob/main/notebooks/02_thailand_step2_consistency_vs_skill.ipynb) | lateral-flow PINN (M1), event-weighted residual, physics consistency vs skill (Fig. 6) | ~5 min | `TRAIN=True, RUN_SOLVER=True`, ~2 h |
| `03_thailand_step3_loso_attribution` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-ACCOUNT/SAVE-LEWS/blob/main/notebooks/03_thailand_step3_loso_attribution.ipynb) | leave-one-storm-out M0/M2/M2np/M3 vs PG-MLP, attribution (Fig. 7), parameter spread, zero-shot Dev108→Dev107 | ~1 min | `RETRAIN=True` spot check ~25 min; all folds `run_loso.py`, ~6 h |
| `04_thailand_step4_bayesian_two_store` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-ACCOUNT/SAVE-LEWS/blob/main/notebooks/04_thailand_step4_bayesian_two_store.ipynb) | Bayesian two-store model (NumPyro VI and NUTS), conformal vs Bayesian bands, transfer | ~1 min | `run_bts.py 0..6 all`, ~25 min |
| `05_puerto_rico_preregistered_test` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-ACCOUNT/SAVE-LEWS/blob/main/notebooks/05_puerto_rico_preregistered_test.ipynb) | pre-registered tests P1–P4 at Utuado and Toro Negro, lag screening, calibrated Richards solver vs PG-MLP | ~1 min | `RUN_ALL=True`, ~5 h |
| `06_robustness_ablation_alerts_nowcast` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-ACCOUNT/SAVE-LEWS/blob/main/notebooks/06_robustness_ablation_alerts_nowcast.ipynb) | post hoc: physics-guidance ablation, Se robustness and nested selection, Holm correction, alert scoring, nowcast degradation | ~1 min | `RERUN_*=True`, ~5 h |

Run times were measured on a 2-core CPU (similar to free Colab); full re-run times are approximate. All notebooks are stored with their executed outputs, so they can be read on GitHub without running them.

Model training is stochastic; re-runs on different hardware or library versions reproduce the stored results up to seed-level variation. The stored checkpoints and result files are the ones reported in the manuscript.

## Data

| Folder | Content | Source and licence |
|---|---|---|
| `data/thailand/` | Dev108 (1 Nov 2025 – 28 Jan 2026) and Dev107 (1 Nov 2025 – 24 Apr 2026): RK520 FDR soil moisture at 30 cm, tipping-bucket rain, air temperature and humidity, 1-min LoRaWAN records | Installed and operated by the authors, Walailak University. See `data/thailand/README.md` and `DATA_LICENSE.md` |
| `data/puerto_rico/` | Utuado and Toro Negro hillslope stations, 15-min, July 2018 – June 2020: soil moisture at five depths, tensiometers, piezometers, rain; laboratory soil tests | U.S. Geological Survey, public domain. Smith et al. (2020), https://doi.org/10.5066/P9548YK2. See `data/puerto_rico/README.md` |

`data/CHECKSUMS.sha256` lists the SHA-256 of every data file (`sha256sum -c data/CHECKSUMS.sha256`).

## Repository layout

```
data/            raw data (Thailand: authors; Puerto Rico: USGS) + checksums
thailand/        Thai models, runners, stored results (res*.json, checkpoints/, bts_out/, conf_out/), run logs
puerto_rico/     Puerto Rico models, runners, PREREGISTRATION.md (+ SHA-256), DEVIATIONS.md, stored results (out/), run logs
figures/         figure scripts; make_all_figures.py writes figures/output/Fig01.png ... FigS3.png
tables/          make_tables.py writes the CSV versions of Tables 2-12 and S1
notebooks/       Colab notebooks 00-06
MANUSCRIPT_MAP.md  which script and result file produce each figure, table and quoted number
```

## Pre-registration

The Puerto Rico analysis plan (`puerto_rico/PREREGISTRATION.md`) was frozen before any storm catalogue or model result was computed. Its SHA-256 (`7c9ee270a442c49ba002e64a00fde2c77ee2767234b4edf8f3d9281b604393df`) is in `PREREG_SHA256.txt` and is checked by notebooks 00 and 05. All later changes, including the exploratory R-cal-Se model and every post hoc analysis, are listed in `puerto_rico/DEVIATIONS.md`. Superseded results from two corrected solver errors are kept in `puerto_rico/out/superseded_clipbug/` for transparency.

## Citation

Please cite the article (see `CITATION.cff`) and, for the Puerto Rico data, the USGS data release:

> Smith, J.B., Thomas, M.A., Ashland, F., Michel, A., Mirus, B.B., Wayllace, A., 2020. Hillslope hydrologic monitoring data following Hurricane Maria in 2017, Puerto Rico, July 2018 to June 2020. U.S. Geological Survey data release. https://doi.org/10.5066/P9548YK2

## Publishing this repository

```bash
cd SAVE-LEWS
git init && git add . && git commit -m "Code, data and results for the SAVE-LEWS manuscript"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/SAVE-LEWS.git
git push -u origin main
# then replace YOUR-ACCOUNT in the notebooks, README.md and CITATION.cff:
grep -rl "YOUR-ACCOUNT" . | xargs sed -i 's/YOUR-ACCOUNT/<your-github-name>/g'
git commit -am "Set repository URL" && git push
```

All files are below GitHub's 100 MB limit (largest: 17 MB), so Git LFS is not needed. For a permanent DOI to quote in the manuscript's *Code availability* statement, connect the repository to Zenodo (zenodo.org → GitHub → enable the repository) and create a release; Zenodo then archives it and issues a DOI.

## Licence

Code: MIT (`LICENSE`). Data: see `DATA_LICENSE.md`.
