"""Step 2: does a convergent lateral inflow / drainage term close the sensor water budget,
and does it beat the 1-D model on held-out storms?  usage: python run_step2.py TAG LATERAL(0/1) SEED"""
import torch, sys, json, time
torch.set_num_threads(1)
from train_utils import *
from richards_solver import solver_forecast

tag, lateral, seed = sys.argv[1], bool(int(sys.argv[2])), int(sys.argv[3])
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
g108 = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]
g107 = load_device('../data/thailand/107_pinn.csv')
S108, S107 = storm_catalogue(g108), storm_catalogue(g107)
win0 = make_window(g108, S108.iloc[0])

m, hist = train(win0, lam=1.0, r_scale=1e-6, ramp=0.4, event_weight=20.0,
                lateral=lateral, iters=ITERS, seed=seed)
torch.save(m.state_dict(), f'checkpoints/{tag}.pt')
ph = m.phys
res = {'tag': tag, 'lateral': lateral, 'seed': seed, 'params': ph.values(), 'hist': hist.tolist()}

# --- sensor-cell budget on wet-up steps of the training storm
with torch.no_grad():
    eff = eff_rain(ph, win0['rain']); A = analysis(m, win0, eff)
    th1, q0, S = m(A[:-1], eff[:-1]); R, dthdt, div, Ss = residual(ph, A[:-1], th1, q0, S)
    lat_sink = (ph.p('lam_out') * ph.K(th1)) if lateral else torch.zeros_like(th1)
    rise = (win0['obs'][1:] - win0['obs'][:-1]) > 0.002
    fall = (win0['obs'][1:] - win0['obs'][:-1]) < -0.002
    i = I_OBS
    for nm, sel in [('wetup', rise), ('falling', fall)]:
        res[f'budget_{nm}'] = {
            'n_steps': int(sel.sum()),
            'obs_dthdt': float(((win0['obs'][1:] - win0['obs'][:-1])[sel] / DT).mean()),
            'model_dthdt': float(dthdt[sel, i].mean()), 'matrix_div': float(div[sel, i].mean()),
            'source_S': float(Ss[sel, i].mean()), 'lateral_sink': float(-lat_sink[sel, i].mean()),
            'mean_abs_R': float(R[sel, i].abs().mean()),
            'ratio_R_dthdt': float(R[sel, i].abs().mean() / dthdt[sel, i].abs().mean().clamp_min(1e-20))}
    # water delivered: rain vs lateral inflow over the storm (mm)
    res['water_mm'] = {'rain': float(win0['rain'].sum()), 'effective_rain': float(eff[:, 0].sum()),
                       'lateral_inflow': float(eff[:, 1].sum())}

# --- independent solver with the same parameters, storm-time origins, oracle rain
    idx = origins_and_targets(win0, 24); idx = idx[win0['in_storm'][idx]]
    fe = future_eff(eff, idx, 24)
    ts = solver_forecast(ph, A[idx], fe); tn, _ = forecast(m, A[idx], fe, 24)
    res['solver_check'] = {}
    for h in HORIZONS:
        targ = win0['obs'][idx + h] - win0['obs'][idx]
        sk = lambda tr: float(1 - ((tr[:, h - 1, I_OBS] - A[idx, I_OBS] - targ) ** 2).mean() / (targ ** 2).mean())
        res['solver_check'][h * 5] = {'net': sk(tn), 'solver': sk(ts),
                                      'profile_rms': float(((ts[:, h - 1] - tn[:, h - 1]) ** 2).mean().sqrt())}

# --- skill: all Dev108 storms, then Dev107 zero-shot (theta_r/theta_s anchored to Dev107 min/max)
for dev, g, S, tr, tsat in [(108, g108, S108, 0.05, 0.55), (107, g107, S107, 0.188, 0.463)]:
    ph.tr, ph.ts = tr, tsat
    for si in range(len(S)):
        w = make_window(g, S.iloc[si])
        for orc in (True, False):
            r = evaluate(m, w, oracle=orc)
            res[f'd{dev}_s{si}_{"oracle" if orc else "past"}'] = {k: v for k, v in r.items() if isinstance(v, float)}
ph.tr, ph.ts = 0.05, 0.55
json.dump(res, open(f'res2_{tag}.json', 'w'))
print('done', tag, json.dumps({k: round(v, 4) for k, v in res['params'].items()}), flush=True)
