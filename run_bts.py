"""Step-4 LOSO runner for the Bayesian two-store model.  usage: python run_bts.py FOLD   (FOLD = 0..6 or all)"""
import os, sys, time, json
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import numpy as np
from train_loso import load_device, storm_catalogue, make_window, loso_windows
import bts
from bts import *
bts.set_nu(30.0)                       # chosen on folds 1/4/6 among 4 variants (variant check in bts_variants.py)
STRIDE, STORM_STRIDE = 6, 6
fold = sys.argv[1]
g = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]; S = storm_catalogue(g); TS, TR = 0.55, 0.05
wins = [make_window(g, S.iloc[i]) for i in range(len(S))] if fold == 'all' else loso_windows(g, S, int(fold))
D = pack(wins, stride=STRIDE, storm_stride=STORM_STRIDE); Phi = features(D, TS, TR)
out = {'fold': fold, 'n_train_origins': int(len(D['wi']))}
os.makedirs('bts_out', exist_ok=True)

def summ(post):
    return {k: dict(q05=float(np.quantile(post[k], .05)), q50=float(np.median(post[k])), q95=float(np.quantile(post[k], .95)),
                    sd_log=float(np.std(np.log(post[k])))) for k in ['c', 'tau_c', 'a', 'tau_F']}

posts = {}
t = time.time()
for disc in (True, False):
    best = None
    for seed in range(4):
        post, losses = fit_vi(D, Phi, TS, TR, discrepancy=disc, seed=seed)
        L = float(np.mean(losses[-200:]))
        if best is None or L < best[1]:
            best = (post, L, seed)
    posts['vi' if disc else 'vi_nodisc'] = best[0]
    out['vi' if disc else 'vi_nodisc'] = dict(final_loss=best[1], seed=best[2], post=summ(best[0]))
out['vi_seconds'] = time.time() - t
t = time.time()
post_h, diag = fit_nuts(D, Phi, TS, TR, warmup=500, samples=500, chains=4, seed=0)
posts['nuts'] = post_h
out['nuts'] = dict(diag=diag, post=summ(post_h), seconds=time.time() - t)
# per-chain medians (to show modes)
nch = 4; per = len(post_h['a']) // nch
out['nuts']['chain_medians'] = {k: [float(np.median(post_h[k][i * per:(i + 1) * per])) for i in range(nch)] for k in ['c', 'tau_c', 'a', 'tau_F']}

def evaluate(Dh, ts, tr, tag):
    for name, post in posts.items():
        disc = name != 'vi_nodisc'
        for orc in (True, False):
            mu, yy = predictive(post, Dh, ts, tr, oracle=orc, discrepancy=disc)
            m, q = interval_metrics(yy, Dh)
            key = f'{tag}_{name}_{"oracle" if orc else "past"}'
            out[key] = m
            np.savez_compressed(f'bts_out/{fold}_{key}.npz', y=Dh['y'], ym=Dh['ym'], storm=Dh['storm'], **q)

if fold == 'all':
    g7 = load_device('../data/thailand/107_pinn.csv'); S7 = storm_catalogue(g7)
    for si in range(len(S7)):
        evaluate(pack([make_window(g7, S7.iloc[si])]), 0.463, 0.188, f'd107s{si}')
else:
    evaluate(pack([make_window(g, S.iloc[int(fold)])]), TS, TR, 'held')
json.dump(out, open(f'bts_out/res4_{fold}.json', 'w'))
print('done', fold, round(out['vi_seconds']), round(out['nuts']['seconds']), out['nuts']['diag'], flush=True)
