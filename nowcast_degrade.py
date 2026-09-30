"""Sensitivity of the PG-MLP forecast skill to rain-nowcast error (trained with oracle rain, evaluated with degraded nowcasts)."""
import numpy as np, pandas as pd, torch, json
torch.set_num_threads(2)
from train_loso import load_device, storm_catalogue, HORIZONS
from pgmlp import features as pg_features, samples as pg_samples, fit as pg_fit, window_bounds
g = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]; S = storm_catalogue(g)
rng = np.random.default_rng(0)
r0 = g.rain.fillna(0.0)
def fut(r):
    return {h: r[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1).fillna(0) for h in HORIZONS}
SCEN = {'oracle': None, 'amount error, factor ~1.6 (sigma 0.5)': ('amp', 0.5), 'amount error, factor ~2.7 (sigma 1.0)': ('amp', 1.0),
        'timing error +30 min': ('shift', 6), 'timing error +60 min': ('shift', 12), 'missed 30% of rain': ('miss', 0.3)}
res = {k: {'se': 0.0, 'ref': 0.0} for k in SCEN}; res['past-only model'] = {'se': 0.0, 'ref': 0.0}
for held in [0, 1, 3, 4, 5, 6]:
    fs = {}
    for orc in (True, False):
        f = pg_features(g, orc); cols = list(f.columns)
        a, b = window_bounds(S.iloc[held]); p0, p1 = a - pd.Timedelta(hours=24), b + pd.Timedelta(hours=24); tr = []
        for i in range(len(S)):
            if i == held: continue
            wa, wb = window_bounds(S.iloc[i]); s0, s1 = S.start[i], S.end[i]
            if p1 > wa and p0 < wb:
                if p1 <= s0: wa = max(wa, p1)
                elif p0 >= s1: wb = min(wb, p0)
                else: continue
            tr.append((wa, wb - pd.Timedelta(minutes=120), s0, s1, i))
        X, Y, ST, _ = pg_samples(g, f, tr); m = pg_fit(X, Y, ST, cols, orc, seed=0); fs[orc] = (f, m)
    w = g.loc[a:b]; s0, s1 = S.start[held], S.end[held]
    storm = (w.index >= s0 - pd.Timedelta(hours=1)) & (w.index <= s1 + pd.Timedelta(hours=6))
    pos = g.index.get_indexer(w.index); th = g.theta.values
    y = th[np.clip(pos + 12, 0, len(g) - 1)] - th[pos]
    def pred(f, m):
        x = f.loc[w.index].values; ok = ~np.isnan(x).any(1)
        out = np.full(len(w), np.nan)
        with torch.no_grad(): out[ok] = m(torch.tensor(x[ok], dtype=torch.float32)).numpy()[:, 1]
        return out
    for name, sc in SCEN.items():
        reps = 1 if sc is None or sc[0] == 'shift' else 10
        for rep in range(reps):
            f = fs[True][0].copy()
            if sc is not None:
                if sc[0] == 'amp':
                    e = np.exp(rng.normal(-sc[1] ** 2 / 2, sc[1], len(r0)))
                    for h, v in fut(r0).items(): f[f'fr{h}'] = v * e
                    r = None
                elif sc[0] == 'shift': r = r0.shift(sc[1]).fillna(0)
                else: r = r0 * (rng.random(len(r0)) > sc[1])
                if r is not None:
                    for h, v in fut(r).items(): f[f'fr{h}'] = v
            p = pred(f, fs[True][1]); ok = storm & ~np.isnan(p) & ~np.isnan(y)
            res[name]['se'] += np.sum((p[ok] - y[ok]) ** 2) / reps; res[name]['ref'] += np.sum(y[ok] ** 2) / reps
    p = pred(*fs[False]); ok = storm & ~np.isnan(p) & ~np.isnan(y)
    res['past-only model']['se'] += np.sum((p[ok] - y[ok]) ** 2); res['past-only model']['ref'] += np.sum(y[ok] ** 2)
    print('fold', held, flush=True)
out = {k: 1 - v['se'] / v['ref'] for k, v in res.items()}
print(json.dumps(out, indent=1)); json.dump(out, open('nowcast_degrade.json', 'w'), indent=1)
