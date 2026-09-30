"""Physics-only LOSO check: implicit Richards solver (+ fast store / probe operator) with each fold's
learned parameters, own assimilation pass, oracle rain, held-out storm.  usage: python solver_loso.py MODEL"""
import torch, sys, json, os
torch.set_num_threads(1)
from train_loso import *
from richards_solver import be_step

CFG = {'M0': dict(), 'M2': dict(fast=True), 'M3': dict(fast=True, bounded=True)}
model = sys.argv[1]
g = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]; S = storm_catalogue(g)
out = {}
for fold in range(7):
    ph = Physics(**CFG[model]); m = Stepper(ph)
    m.load_state_dict(torch.load(f'checkpoints/loso_{model}_f{fold}.pt'))
    w = make_window(g, S.iloc[fold]); P = stack([w])
    with torch.no_grad():
        drv = ph.drive(P['rain']); G = nudge_gain()
        obs, ok = P['obs'][0], P['ok'][0]
        th = torch.full((1, NZ), float(obs[ok][0])); A = [th]
        for k in range(len(obs) - 1):
            th = be_step(ph, th, drv[0, k:k + 1])
            if ok[k + 1]:
                tgt = ph.sensor_inverse(obs[k + 1], drv[0, k + 1, 3]) if ph.fast else obs[k + 1]
                th = (th + G * (tgt - th[:, I_OBS])).clamp(ph.tr + 1e-3, ph.ts - 1e-3)
            A.append(th)
        A = torch.cat(A)
        wi, ki = origins(P)
        fd = future_drive(ph, drv, wi, ki, oracle=True)
        th = A[ki]; traj = []
        for j in range(H):
            th = be_step(ph, th, fd[:, j]); traj.append(th)
        traj = torch.stack(traj, 1)
        st = P['storm'][0]
        for h in HORIZONS:
            sel = ok[ki + h] & st[ki]
            pred = pred_sensor_delta(ph, traj, fd, obs[ki], h); targ = obs[ki + h] - obs[ki]
            out[f'f{fold}_{h*5}'] = dict(mse=float(((pred - targ)[sel] ** 2).mean()), msep=float((targ[sel] ** 2).mean()),
                                        n=int(sel.sum()))
    print(model, fold, round(1 - out[f'f{fold}_60']['mse'] / out[f'f{fold}_60']['msep'], 3), flush=True)
json.dump(out, open(f'res3_solver_{model}.json', 'w')); print('done')
