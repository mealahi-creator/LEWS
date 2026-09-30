"""Physics-guidance ablation of the PG-MLP: U = unconstrained MLP; B = storage-bounded output only;
PG = full physics guidance (bounded output + no-wetting-without-rain + monotone-in-rain penalties + saturation gate).
usage: python pg_ablation.py VARIANT"""
import sys, json, time, torch, numpy as np, pandas as pd; torch.set_num_threads(1)
import torch.nn as nn
from pgmlp import features, samples, window_bounds, HORIZONS, TR, TS
from data_pipeline import load_device, storm_catalogue
V = sys.argv[1]
class Net(nn.Module):
    def __init__(self, d, mu, sd):
        super().__init__(); self.mu, self.sd = mu, sd
        self.net = nn.Sequential(nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2 * len(HORIZONS)))
    def forward(self, x):
        a = self.net((x - self.mu) / self.sd).view(-1, len(HORIZONS), 2)
        if V == 'U':
            return 0.05 * a[..., 0]
        th = x[:, 0:1]; se = x[:, 1:2]
        d = (TS - th) * torch.sigmoid(a[..., 0]) - (th - TR) * torch.sigmoid(a[..., 1])
        if V == 'PG':
            d = torch.where(se >= 0.90, torch.zeros_like(d), d)
        return d
def fit(X, Y, ST, cols, seed):
    torch.manual_seed(seed); mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    m = Net(X.shape[1], mu, sd); opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-5)
    okY = ~torch.isnan(Y); Y0 = torch.nan_to_num(Y)
    pers = torch.stack([(Y0[:, j][okY[:, j]] ** 2).mean() for j in range(Y.shape[1])]); w = torch.where(ST, 4.0, 1.0)
    ic = {c: i for i, c in enumerate(cols)}; fut = [ic[c] for c in cols if c.startswith('fr')]; past3h = ic['r36']
    for it in range(3000):
        b = torch.multinomial(w, 512, replacement=True); x, y, ok = X[b], Y0[b], okY[b]; p = m(x)
        loss = (((p - y) ** 2 * ok) / pers).sum() / ok.sum()
        if V == 'PG':
            dry = x[:, past3h] <= 0
            if fut: dry = dry & (x[:, fut].sum(1) <= 0)
            pen = (torch.relu(p[dry]) ** 2).mean() / pers.mean() if dry.any() else 0.0
            if fut:
                x2 = x.clone(); x2[:, fut] = x2[:, fut] + 1.0; pen = pen + (torch.relu(p - m(x2)) ** 2).mean() / pers.mean()
            loss = loss + pen
        opt.zero_grad(); loss.backward(); opt.step()
    return m
def evaluate(m, Xt, Yt, STt, cols, oracle):
    ic = {c: i for i, c in enumerate(cols)}; fut = [ic[c] for c in cols if c.startswith('fr')]
    with torch.no_grad():
        P = m(Xt)
        th = Xt[:, 0:1]; out = {}
        for j, h in enumerate(HORIZONS):
            s = ~torch.isnan(Yt[:, j]) & STt
            out[f'se_{h*5}'] = float(((P[s, j] - Yt[s, j]) ** 2).sum()); out[f'ref_{h*5}'] = float((Yt[s, j] ** 2).sum()); out[f'n_{h*5}'] = int(s.sum())
        tot = th + P
        out['viol_bounds'] = float(((tot > TS + 1e-4) | (tot < TR - 1e-4)).any(1).float().mean())
        dry = Xt[:, ic['r36']] <= 0
        if fut: dry = dry & (Xt[:, fut].sum(1) <= 0)
        out['n_dry'] = int(dry.sum()); out['viol_wet_dry'] = float((P[dry][:, 1] > 0.002).float().mean()) if dry.any() else float('nan')
        if fut:
            X2 = Xt.clone(); X2[:, fut] = X2[:, fut] + 1.0; P2 = m(X2)
            out['viol_monotone'] = float((P2[:, 1] < P[:, 1] - 0.002).float().mean())
    return out
g8 = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]; S8 = storm_catalogue(g8)
g7 = load_device('../data/thailand/107_pinn.csv'); S7 = storm_catalogue(g7); TR7, TS7 = 0.188, 0.463
g7m = g7.copy(); g7m['theta'] = TR + (g7.theta - TR7) / (TS7 - TR7) * (TS - TR)
res = {}; t0 = time.time()
for oracle in (True, False):
    mode = 'oracle' if oracle else 'past'
    f8 = features(g8, oracle); cols = list(f8.columns)
    for seed in range(3):
        for held in range(len(S8)):
            a, b = window_bounds(S8.iloc[held]); p0, p1 = a - pd.Timedelta(hours=24), b + pd.Timedelta(hours=24); tr = []
            for i in range(len(S8)):
                if i == held: continue
                wa, wb = window_bounds(S8.iloc[i]); s0, s1 = S8.start[i], S8.end[i]
                if p1 > wa and p0 < wb:
                    if p1 <= s0: wa = max(wa, p1)
                    elif p0 >= s1: wb = min(wb, p0)
                    else: continue
                tr.append((wa, wb - pd.Timedelta(minutes=5 * max(HORIZONS)), s0, s1, i))
            X, Y, ST, _ = samples(g8, f8, tr); m = fit(X, Y, ST, cols, seed)
            Xt, Yt, STt, _ = samples(g8, f8, [(a, b - pd.Timedelta(minutes=5 * max(HORIZONS)), S8.start[held], S8.end[held], held)])
            res[f'loso_{mode}_seed{seed}_s{held}'] = evaluate(m, Xt, Yt, STt, cols, oracle)
        # zero-shot: train on all Dev108 storms
        tr = [(window_bounds(S8.iloc[i])[0], window_bounds(S8.iloc[i])[1] - pd.Timedelta(minutes=5 * max(HORIZONS)), S8.start[i], S8.end[i], i) for i in range(len(S8))]
        X, Y, ST, _ = samples(g8, f8, tr); m = fit(X, Y, ST, cols, seed); f7 = features(g7m, oracle)
        for si in range(len(S7)):
            a, b = window_bounds(S7.iloc[si])
            Xt, Yt, STt, _ = samples(g7m, f7, [(a, b - pd.Timedelta(minutes=5 * max(HORIZONS)), S7.start[si], S7.end[si], si)])
            res[f'zs_{mode}_seed{seed}_s{si}'] = evaluate(m, Xt, Yt, STt, cols, oracle)
        print(V, mode, seed, f'{time.time()-t0:.0f}s', flush=True)
json.dump(res, open(f'res_ablation_{V}.json', 'w')); print('done', V)
