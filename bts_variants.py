import time, numpy as np, json, sys
from train_loso import load_device, storm_catalogue, make_window, loso_windows
import bts
from bts import *
g = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]; S = storm_catalogue(g); TS, TR = 0.55, 0.05
for nu, sstride in [(4.0, 6), (30.0, 6), (4.0, 2), (30.0, 2)]:
    bts.set_nu(nu); row = []
    for fold in (1, 4, 6):
        D = pack(loso_windows(g, S, fold), stride=6, storm_stride=sstride); Phi = features(D, TS, TR)
        post, _ = fit_vi(D, Phi, TS, TR)
        Dh = pack([make_window(g, S.iloc[fold])]); mu, yy = predictive(post, Dh, TS, TR)
        m, _ = interval_metrics(yy, Dh)
        row.append(f"f{fold} skill60 {1-m['storm_60']['mse']/m['storm_60']['msep']:+.2f} cov60 {m['storm_60']['coverage']:.2f} a {np.median(post['a']):.3g} tauF {np.median(post['tau_F']):.3g} c {np.median(post['c']):.3g}")
    print(f'nu={nu} storm_stride={sstride}:', ' | '.join(row), flush=True)
