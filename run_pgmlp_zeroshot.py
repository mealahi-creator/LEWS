import torch, json, numpy as np, pandas as pd; torch.set_num_threads(1)
import pgmlp
from pgmlp import *
from data_pipeline import load_device, storm_catalogue
g8 = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]; S8 = storm_catalogue(g8)
g7 = load_device('../data/thailand/107_pinn.csv'); S7 = storm_catalogue(g7)
TR7, TS7 = 0.188, 0.463
# express Dev107 in Dev108 units through saturation: theta108 = 0.05 + Se107 * 0.50
g7m = g7.copy(); g7m['theta'] = TR + (g7.theta - TR7) / (TS7 - TR7) * (TS - TR)
res = {}
for oracle in (True, False):
    f8 = features(g8, oracle); cols = list(f8.columns)
    tr = [(window_bounds(S8.iloc[i])[0], window_bounds(S8.iloc[i])[1] - pd.Timedelta(minutes=5 * max(HORIZONS)), S8.start[i], S8.end[i], i) for i in range(len(S8))]
    X, Y, ST, _ = samples(g8, f8, tr)
    f7 = features(g7m, oracle)
    for seed in range(3):
        m = fit(X, Y, ST, cols, oracle, seed=seed)
        for si in range(len(S7)):
            a, b = window_bounds(S7.iloc[si])
            Xt, Yt, STt, _ = samples(g7m, f7, [(a, b - pd.Timedelta(minutes=5 * max(HORIZONS)), S7.start[si], S7.end[si], si)])
            with torch.no_grad(): P = m(Xt)
            out = {}
            for j, h in enumerate(HORIZONS):
                s = ~torch.isnan(Yt[:, j]) & STt
                out[f'skill_storm_{h*5}'] = float(1 - ((P[s, j] - Yt[s, j]) ** 2).mean() / (Yt[s, j] ** 2).mean())
                out[f'n_storm_{h*5}'] = int(s.sum())
            res[f'seed{seed}_{"oracle" if oracle else "past"}_s{si}'] = out
            print(seed, oracle, si, round(out['skill_storm_60'], 3), flush=True)
json.dump(res, open('res3_pgmlp_zeroshot.json', 'w')); print('done')
