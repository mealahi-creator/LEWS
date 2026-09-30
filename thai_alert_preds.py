"""Held-out per-origin 60-min predictions for M0, M2np and M3 (oracle rain) on the conf_out origin set."""
import numpy as np, torch
torch.set_num_threads(2)
from train_loso import (load_device, storm_catalogue, make_window, Physics, Stepper, stack, analysis, origins,
                        future_drive, rollout, pred_sensor_delta, HORIZONS)
from bts import pack
g = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]; S = storm_catalogue(g)
CFG = {'M0': dict(), 'M2np': dict(fast=True), 'M3': dict(fast=True, bounded=True)}
for held in range(7):
    w = make_window(g, S.iloc[held]); D = pack([w]); rec = {}
    for tag, kw in CFG.items():
        m = Stepper(Physics(**kw)); m.load_state_dict(torch.load(f'checkpoints/loso_{tag}_f{held}.pt'))
        with torch.no_grad():
            P = stack([w]); drv = m.phys.drive(P['rain']); A = analysis(m, P, drv)
            wi, ki = origins(P); fd = future_drive(m.phys, drv, wi, ki, True)
            traj, _ = rollout(m, A[wi, ki], fd)
            rec[tag] = np.stack([pred_sensor_delta(m.phys, traj, fd, P['obs'][wi, ki], h).numpy() for h in HORIZONS], 1)
        assert np.array_equal(ki.numpy(), D['ki'])
    np.savez_compressed(f'conf_out/extra_{held}.npz', **rec); print('fold', held, flush=True)
