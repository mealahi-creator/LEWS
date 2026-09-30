# Manuscript map: where every figure, table and key number comes from

Figure and table numbers follow the submitted *Landslides* manuscript. "Script" produces the output; "Results" are the stored files it reads (produced by the runners listed in the last column).

## Figures (`python figures/make_all_figures.py` → `figures/output/`)

| Fig. | Content | Script | Results read | Produced by |
|---|---|---|---|---|
| 1 | Sites and instrumentation | `figures/fig_sites.py` | – (schematic) | – |
| 2 | Sensor-system comparison (continuity, dynamic range) | `figures/fig_sensors.py` | raw data | – |
| 3 | SAVE framework | `figures/fig1_save.py` | – (schematic) | – |
| 4 | Utuado lag of deep rise after 27 cm onset (P2) | `puerto_rico/figs_pr.py` | raw data | `p2_lag.py` |
| 5 | Thai training storm, PINNs A/B/C, residual at the probe | `thailand/make_figs_thai.py` | `checkpoints/A_weak.pt, B_strong.pt, C_nophys.pt` | notebook 01 (`TRAIN=True`) |
| 6 | Physics consistency ρ_R vs held-out skill | `figures/make_thai_figs.py` | `thailand/step2_summary.json` ← `res_pinn.json`, `res_B_strong.json`, `res_C_nophys.json`, `res2_M*_s*.json`, `solver_*.json` | notebooks 01–02, `run_step2.py`, `solver_heldout.py` |
| 7 | Signal attribution s_matrix and fast-store share | `thailand/fig_step3.py` | `res3_loso_M*_f*.json` | `run_loso.py` / notebook 03 |
| 8 | Physics-guidance ladder | `figures/fig_ladder.py` | `thailand/ablation_summary.json`, `puerto_rico/out/mlpU_summary.json`, `se_nested_*.json` | `pg_ablation.py`, `run_p1_mlpU.py`, `se_nested.py` |
| 9 | Thai LOSO skill by horizon | `thailand/fig_step3.py` | `res3_pgmlp.json`, `res3_loso_*.json` | `run_pgmlp.py`, `run_loso.py` |
| 10 | Puerto Rico skill by site and depth | `puerto_rico/figs_final_pr.py` | `out/p1_report.json`, `out/explore_72.json` | `run_p1_*.py`, `eval_p1.py`, `explore_72.py` |
| 11 | Storm bootstrap physics − PG-MLP | `puerto_rico/figs_final_pr.py` | `out/p1_report.json` | `eval_p1.py` |
| 12 | Utuado largest held-out storm | `puerto_rico/figs_pr.py` | `out/p1_*_utuado.npz` | `run_p1_*.py` |
| 13 | Hydrological alert performance | `figures/fig_alerts.py` | `thailand/thai_alerts.json`, `puerto_rico/out/pr_alerts_rise.json` | `thai_alerts.py`, `pr_alerts_rise.py` |
| 14 | Warning-band calibration (a Thailand, b Puerto Rico) | `thailand/fig_step4.py` + `puerto_rico/figs_final_pr.py`, composed in `make_all_figures.py` | `bts_out/`, `conf_out/`, `out/p4_report.json` | `run_bts.py`, `conf_preds.py`, `run_p4.py`, `eval_p4.py` |
| 15 | Zero-shot transfer Dev108 → Dev107 | `figures/make_thai_figs.py` | `res3_pgmlp_zeroshot.json`, `res3_loso_M*_fall.json`, `bts_out/res4_all.json` | `run_pgmlp_zeroshot.py`, `run_loso.py all`, `run_bts.py all` |
| S1 | Thai parameter spread across folds | `figures/make_thai_figs.py` | `res3_loso_M*_f*.json` | `run_loso.py` |
| S2 | B-TS posterior intervals per fold | `thailand/fig_step4.py` | `bts_out/res4_*.json` | `run_bts.py` |
| S3 | Example held-out Thai storm with bands | `thailand/fig_step4.py` | `bts_out/1_held_nuts_oracle.npz`, `conf_out/held_1.npz` | `run_bts.py`, `conf_preds.py` |

## Tables (`python tables/make_tables.py`, `python figures/sensor_compare.py` → `tables/`)

| Table | CSV | Source |
|---|---|---|
| 1 Data used | – | README / data READMEs |
| 2 Lab van Genuchten parameters | `Table02_PR_lab_parameters.csv` | `PuertoRico_TestingData.xlsx`, sheet `Summary` |
| 3 Sensor systems and data quality | `Table03_sensor_comparison.csv` | `figures/sensor_compare.py` (raw data) |
| 4 Models compared | – | descriptive |
| 5 Step I screening | `Table05a_PR_lag_screening_P2.csv`, `Table05b_PR_depth_inversions.csv`, `TableS1_*` | `res_p2.json`, `out/screening_metrics.json`, raw Thai data |
| 6 Physics-guidance ladder | `Table06_physics_guidance_ladder.csv` | `ablation_summary.json`, `mlpU_summary.json` |
| 7 Thai LOSO skill | `Table07_Thai_LOSO_skill.csv` | `analyze_step3.summary()` |
| 8 Pre-registered tests | `Table08_PR_preregistered_skill.csv`, `Table08b_PR_P3a_parameters.csv` | `out/p1_report.json` (+ P2: `res_p2.json`, P4: `out/p4_report.json`) |
| 9 Storm bootstrap | `Table09a_Thai_storm_bootstrap.csv`, `Table09b_PR_storm_bootstrap.csv` | `bootstrap_table6.py`, `out/p1_report.json` |
| 10 Robustness of R-cal-Se | `Table10_PR_robustness.csv` | `out/se_robust.json`, `se_nested_*.json`, `se_holm.json` |
| 11 Alert performance | `Table11_alert_performance.csv` | `thai_alerts.json` (rise threshold 0.01 in the manuscript), `out/pr_alerts_rise.json` |
| 12 Warning bands | `Table12a_Thai_band_calibration.csv`, `Table12b_PR_band_calibration_P4.csv` | `h2_compare.py`, `out/p4_report.json` |
| 13 Sensor configurations (design guidance) | – | synthesis of Tables 3, 5, 6 |
| 14 Strength of evidence | – | synthesis |
| S1 Thai water-balance bound | `TableS1_Thai_water_balance_bound.csv` | raw Thai data |
| (text) Nowcast degradation | `Nowcast_degradation.csv` | `thailand/nowcast_degrade.json` ← `nowcast_degrade.py` |

## Key numbers quoted in the Abstract and Conclusions

| Number | File |
|---|---|
| ρ_R = 0.68, network skill 0.59 vs solver 0.17–0.22 | `thailand/step2_summary.json` |
| s_matrix ≈ 0.011–0.013 | `thailand/table_attribution_diagnostics.csv` (median `matrix_share_wetup`) |
| Thai 60-min skill U 0.17–0.37, B 0.32–0.42, PG 0.46–0.54; monotonicity violations 42 % → 0.04 % → 0.02 % | `thailand/ablation_summary.json` |
| R-cal 0.00 vs PG-MLP 0.31–0.39 (Utuado 72 cm); −1.16 vs 0.71–0.73 (Toro Negro 90 cm) | `puerto_rico/out/p1_report.json` |
| R-cal-Se 0.67 at 57 cm; nested 0.71 / 0.68; Holm p = 0.0004 | `out/p1_report.json`, `out/se_nested_utuado.json`, `out/se_holm.json` |
| Alerts: M0 POD 0.06 vs PG-MLP 0.85; R-cal timely 2 % | `thailand/thai_alerts.json`, `out/pr_alerts_rise.json` |
| Nowcast: 0.54 → 0.46 / 0.26 / 0.33 / 0.12; past-only 0.02 | `thailand/nowcast_degrade.json` |
| Pre-registration SHA-256 7c9ee270… | `puerto_rico/PREREG_SHA256.txt` |
