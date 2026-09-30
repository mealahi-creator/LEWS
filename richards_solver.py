"""Independent backward-Euler Richards solver (Newton, dense Jacobian per column) with the
same parameterisation/forcing as the PINN. Used to (i) verify that the PINN's profile is a
Richards solution and (ii) as a physics-only baseline."""
import torch
from torch.func import jacrev, vmap
from richards_pinn import DT, NZ, I_OBS

@torch.no_grad()
def be_step(phys, th, eff_mm, nsub=2, newton=8):
    """th (B,NZ) -> th_new (B,NZ); forcing frozen at start of step as in the PINN."""
    q0, S = phys.forcing(th, eff_mm)
    dt = DT / nsub
    lo, hi = phys.tr + 1e-4, phys.ts - 1e-4
    for _ in range(nsub):
        old = th
        def G(x, xo, q0_, S_):
            f, _, _ = phys.rhs(x[None], q0_[None], S_[None])
            return x - xo - dt * f[0]
        x = old.clone()
        for _ in range(newton):
            r = vmap(G)(x, old, q0, S)
            J = vmap(jacrev(G))(x, old, q0, S)
            dx = torch.linalg.solve(J, -r.unsqueeze(-1)).squeeze(-1)
            x = (x + dx.clamp(-0.05, 0.05)).clamp(lo, hi)
        th = x
    return th

@torch.no_grad()
def solver_forecast(phys, th0, eff_seq):
    th = th0; traj = []
    for j in range(eff_seq.shape[1]):
        th = be_step(phys, th, eff_seq[:, j]); traj.append(th)
    return torch.stack(traj, 1)
