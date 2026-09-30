"""Held-out per-origin predictions for persistence, the M2 PINN (LOSO checkpoints) and the PG-MLP (seed 0),
on the same origin set as the B-TS evaluation, for cross-conformal interval baselines."""
import os, numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from train_loso import (load_device, storm_catalogue, make_window, Physics, Stepper, stack, analysis, origins,
                        future_drive, rollout, pred_sensor_delta, HORIZONS)
import pgmlp
from pgmlp import features as pg_features, samples as pg_samples, fit as pg_fit, window_bounds
from bts import pack

g = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]; S = storm_catalogue(g)
os.makedirs('conf_out', exist_ok=True)

def pgmlp_fold(held, oracle, seed=0):
    f = pg_features(g, oracle); cols = list(f.columns)
    a, b = window_bounds(S.iloc[held]); p0, p1 = a - pd.Timedelta(hours=24), b + pd.Timedelta(hours=24)
    tr = []
    for i in range(len(S)):
        if i == held: continue
        wa, wb = window_bounds(S.iloc[i]); s0, s1 = S.start[i], S.end[i]
        if p1 > wa and p0 < wb:
            if p1 <= s0: wa = max(wa, p1)
            elif p0 >= s1: wb = min(wb, p0)
            else: continue
        tr.append((wa, wb - pd.Timedelta(minutes=120), s0, s1, i))
    X, Y, ST, _ = pg_samples(g, f, tr)
    m = pg_fit(X, Y, ST, cols, oracle, seed=seed)
    # predict at every held-out timestamp where features exist
    w = g.loc[a:b]; x = f.loc[w.index].values; ok = ~np.isnan(x).any(1)
    pred = np.full((len(w), 3), np.nan, np.float32)
    with torch.no_grad():
        pred[ok] = m(torch.tensor(x[ok], dtype=torch.float32)).numpy()
    return pd.DataFrame(pred, index=w.index, columns=[30, 60, 120])

for held in range(7):
    w = make_window(g, S.iloc[held]); D = pack([w])
    ts_index = w['t'][D['ki']]
    rec = {'t': np.array(ts_index.astype('int64')), 'y': D['y'], 'ym': D['ym'], 'storm': D['storm']}
    for mode, orc in [('oracle', True), ('past', False)]:
        rec[f'persistence_{mode}'] = np.zeros_like(D['y'])
        # M2 PINN
        m = Stepper(Physics(fast=True)); m.load_state_dict(torch.load(f'checkpoints/loso_M2_f{held}.pt'))
        with torch.no_grad():
            P = stack([w]); drv = m.phys.drive(P['rain']); A = analysis(m, P, drv)
            wi, ki = origins(P); fd = future_drive(m.phys, drv, wi, ki, orc)
            traj, _ = rollout(m, A[wi, ki], fd)
            pm = np.stack([pred_sensor_delta(m.phys, traj, fd, P['obs'][wi, ki], h).numpy() for h in HORIZONS], 1)
        assert np.array_equal(ki.numpy(), D['ki']), 'origin sets differ'
        rec[f'M2_{mode}'] = pm
        # PG-MLP (seed 0), aligned by timestamp; origins without features get NaN
        pg = pgmlp_fold(held, orc)
        rec[f'PGMLP_{mode}'] = pg.reindex(ts_index).values
    np.savez_compressed(f'conf_out/held_{held}.npz', **rec)
    print('fold', held, 'origins', len(D['ki']), 'PG-MLP missing', int(np.isnan(rec['PGMLP_oracle'][:, 0]).sum()), flush=True)
print('done')
