"""usage: python run_loso.py MODEL FOLD   (FOLD = held-out storm index, or 'all' = train on all 7)"""
import torch, json, sys, time; torch.set_num_threads(1)
from train_loso import *
CFG = {'M0': dict(fast=False, lam=1.0), 'M2': dict(fast=True, lam=1.0), 'M2np': dict(fast=True, lam=0.0), 'M3': dict(fast=True, bounded=True, lam=1.0)}
model_tag, fold = sys.argv[1], sys.argv[2]
g = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]; S = storm_catalogue(g)
if fold == 'all':
    wins = [make_window(g, S.iloc[i]) for i in range(len(S))]
else:
    wins = loso_windows(g, S, int(fold))
t0 = time.time()
m, hist = train(wins, **CFG[model_tag])
tag = f'{model_tag}_f{fold}'
torch.save(m.state_dict(), f'checkpoints/loso_{tag}.pt')
res = {'params': m.phys.values(), 'train_s': time.time() - t0, 'hist_last': hist[-50:].mean(0).tolist()}
strip = lambda r: {k: v for k, v in r.items() if isinstance(v, (int, float))}
if fold == 'all':
    g7 = load_device('../data/thailand/107_pinn.csv'); S7 = storm_catalogue(g7)
    m.phys.tr, m.phys.ts = 0.188, 0.463
    for si in range(len(S7)):
        for orc in (True, False):
            res[f'd107_s{si}_{"oracle" if orc else "past"}'] = strip(evaluate(m, make_window(g7, S7.iloc[si]), oracle=orc))
else:
    w = make_window(g, S.iloc[int(fold)])
    for orc in (True, False):
        res[f'held_{"oracle" if orc else "past"}'] = strip(evaluate(m, w, oracle=orc))
json.dump(res, open(f'res3_loso_{tag}.json', 'w')); print('done', tag, flush=True)
