"""Physics-only held-out test: implicit Richards solver with each run's learned parameters,
its own real-time assimilation pass (solver + nudging), oracle rain, Dev108 storms.
usage: python solver_heldout.py TAG [apriori]"""
import torch, sys, json
torch.set_num_threads(1)
from train_utils import *
from richards_solver import be_step, solver_forecast

tag = sys.argv[1]
if tag == 'apriori':
    ph = Physics()
else:
    lateral = tag.startswith('M1')
    ph = Physics(lateral=lateral); m = Stepper(ph)
    m.load_state_dict(torch.load(f'checkpoints/{tag}.pt'))
g = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]; S = storm_catalogue(g)

@torch.no_grad()
def solver_analysis(win, drv):
    obs, ok = win['obs'], win['obs_ok']; G = nudge_gain()
    th = torch.full((1, NZ), float(obs[ok][0])); out = [th]
    for k in range(len(obs) - 1):
        th = be_step(ph, th, drv[k:k + 1])
        if ok[k + 1]:
            th = (th + G * (obs[k + 1] - th[:, I_OBS])).clamp(ph.tr + 1e-3, ph.ts - 1e-3)
        out.append(th)
    return torch.cat(out)

res = {}
with torch.no_grad():
    for si in range(len(S)):
        w = make_window(g, S.iloc[si]); drv = ph.drive(w['rain'])
        A = solver_analysis(w, drv)
        idx = origins_and_targets(w, 24)
        for orc in (True, False):
            tr = solver_forecast(ph, A[idx], future_eff(drv, idx, 24, orc))
            for h in HORIZONS:
                ok = w['obs_ok'][idx + h]
                targ = w['obs'][idx + h] - w['obs'][idx]; pred = tr[:, h - 1, I_OBS] - A[idx, I_OBS]
                sel = ok & w['in_storm'][idx]
                res[f's{si}_{"oracle" if orc else "past"}_{h*5}'] = float(1 - ((pred - targ)[sel] ** 2).mean() / (targ[sel] ** 2).mean())
        print(tag, si, round(res[f's{si}_oracle_60'], 3), flush=True)
json.dump(res, open(f'solver_{tag}.json', 'w'))
print('done', tag)
