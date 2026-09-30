"""Physics-guided MLP forecaster (constraint-level physics guidance, no governing equation):
storage-bounded Delta-theta at 30/60/120 min, rain-monotonicity and no-wetting-without-rain penalties,
Se >= 0.90 gate. Oracle variant sees future rain sums; past-only variant does not."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from train_loso import HORIZONS, window_bounds

TR, TS = 0.05, 0.55
PAST = [1, 3, 6, 12, 36, 72]          # 5, 15, 30, 60, 180, 360 min

def features(g, oracle):
    th = g.theta; r = g.rain.fillna(0.0)
    f = pd.DataFrame(index=g.index)
    f['theta'] = th; f['Se'] = ((th - TR) / (TS - TR)).clip(0, 1)
    for L in (1, 3, 6):
        f[f'dth{L}'] = th - th.shift(L)
    for L in PAST:
        f[f'r{L}'] = r.rolling(L, min_periods=1).sum()
    # storm rain to date: rain since the last >= 6 h dry spell
    wet = (r > 0).astype(int); dry6 = r.rolling(72, min_periods=1).sum() == 0
    grp = dry6.cumsum(); f['storm_rain'] = r.groupby(grp).cumsum()
    if oracle:
        for h in HORIZONS:
            f[f'fr{h}'] = r[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1).fillna(0)
    return f

def samples(g, f, wins_times):
    """rows at origins inside the given (start,end,storm_start,storm_end) windows with all targets present."""
    X, Y, M, ST, KEY = [], [], [], [], []
    th = g.theta
    for (a, b, s0, s1, tag) in wins_times:
        idx = g.loc[a:b].index
        pos = g.index.get_indexer(idx)
        for h in HORIZONS:
            pass
        tgt = np.stack([th.values[np.clip(pos + h, 0, len(g) - 1)] - th.values[pos] for h in HORIZONS], 1)
        tgt[(pos[:, None] + np.array(HORIZONS)[None]) >= len(g)] = np.nan
        x = f.iloc[pos].values
        ok = ~np.isnan(x).any(1) & ~np.isnan(th.values[pos])
        storm = (idx >= s0 - pd.Timedelta(hours=1)) & (idx <= s1 + pd.Timedelta(hours=6))
        X.append(x[ok]); Y.append(tgt[ok]); ST.append(storm[ok]); KEY += [tag] * int(ok.sum())
    return (torch.tensor(np.concatenate(X), dtype=torch.float32), torch.tensor(np.concatenate(Y), dtype=torch.float32),
            torch.tensor(np.concatenate(ST)), np.array(KEY))

class PGMLP(nn.Module):
    def __init__(self, d, oracle, mu, sd):
        super().__init__()
        self.mu, self.sd, self.oracle = mu, sd, oracle
        self.net = nn.Sequential(nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2 * len(HORIZONS)))
        self.cols = None

    def forward(self, x):
        a = self.net((x - self.mu) / self.sd).view(-1, len(HORIZONS), 2)
        th = x[:, 0:1]; se = x[:, 1:2]
        d = (TS - th) * torch.sigmoid(a[..., 0]) - (th - TR) * torch.sigmoid(a[..., 1])
        return torch.where(se >= 0.90, torch.zeros_like(d), d)           # saturation gate

def fit(X, Y, ST, cols, oracle, seed=0, steps=3000, lam_pen=1.0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    m = PGMLP(X.shape[1], oracle, mu, sd); m.cols = cols
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-5)
    okY = ~torch.isnan(Y); Y0 = torch.nan_to_num(Y)
    pers = torch.stack([(Y0[:, j][okY[:, j]] ** 2).mean() for j in range(Y.shape[1])])
    w = torch.where(ST, 4.0, 1.0)
    ic = {c: i for i, c in enumerate(cols)}
    fut = [ic[c] for c in cols if c.startswith('fr')]
    past3h = ic['r36']
    for it in range(steps):
        b = torch.multinomial(w, 512, replacement=True)
        x, y, ok = X[b], Y0[b], okY[b]
        p = m(x)
        l = (((p - y) ** 2 * ok) / pers).sum() / ok.sum()
        # no wetting without rain (no rain in last 3 h and, if known, none coming)
        dry = x[:, past3h] <= 0
        if fut:
            dry = dry & (x[:, fut].sum(1) <= 0)
        pen = (torch.relu(p[dry]) ** 2).mean() / pers.mean() if dry.any() else 0.0
        # monotone in future rain
        if fut:
            x2 = x.clone(); x2[:, fut] = x2[:, fut] + 1.0
            pen = pen + (torch.relu(p - m(x2)) ** 2).mean() / pers.mean()
        loss = l + lam_pen * pen
        opt.zero_grad(); loss.backward(); opt.step()
    return m

def run_fold(g, S, held, oracle, seed=0):
    f = features(g, oracle); cols = list(f.columns)
    a, b = window_bounds(S.iloc[held])
    p0, p1 = a - pd.Timedelta(hours=24), b + pd.Timedelta(hours=24)
    tr = []
    for i in range(len(S)):
        if i == held:
            continue
        wa, wb = window_bounds(S.iloc[i]); s0, s1 = S.start[i], S.end[i]
        if p1 > wa and p0 < wb:
            if p1 <= s0: wa = max(wa, p1)
            elif p0 >= s1: wb = min(wb, p0)
            else: continue
        tr.append((wa, wb - pd.Timedelta(minutes=5 * max(HORIZONS)), s0, s1, i))
    X, Y, ST, _ = samples(g, f, tr)
    m = fit(X, Y, ST, cols, oracle, seed=seed)
    Xt, Yt, STt, _ = samples(g, f, [(a, b - pd.Timedelta(minutes=5 * max(HORIZONS)), S.start[held], S.end[held], held)])
    with torch.no_grad():
        P = m(Xt)
    out = {}
    for j, h in enumerate(HORIZONS):
        ok = ~torch.isnan(Yt[:, j]); s = ok & STt
        e = (P[:, j] - Yt[:, j]); out[f'mse_storm_{h*5}'] = float((e[s] ** 2).mean())
        out[f'msep_storm_{h*5}'] = float((Yt[s, j] ** 2).mean()); out[f'n_storm_{h*5}'] = int(s.sum())
        out[f'skill_storm_{h*5}'] = 1 - out[f'mse_storm_{h*5}'] / out[f'msep_storm_{h*5}']
        out[f'mse_all_{h*5}'] = float((e[ok] ** 2).mean()); out[f'msep_all_{h*5}'] = float((Yt[ok, j] ** 2).mean())
        out[f'skill_all_{h*5}'] = 1 - out[f'mse_all_{h*5}'] / out[f'msep_all_{h*5}']
    return out
