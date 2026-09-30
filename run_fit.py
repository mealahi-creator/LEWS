import torch, sys, json, pickle
torch.set_num_threads(1)
from train_utils import *
lam = float(sys.argv[1]); iters = int(sys.argv[2]); tag = sys.argv[3]; seed = 0; rs = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
g = load_device('../data/thailand/pinn_108_new.csv'); g = g.loc['2025-11-01 10:40':]
s = storm_catalogue(g)
win = make_window(g, s.iloc[0])
m, h = train(win, lam=lam, iters=iters, seed=seed, r_scale=rs)
torch.save(m.state_dict(), f'fit_{tag}.pt')
res = {}
for si in range(len(s)):
    w = make_window(g, s.iloc[si])
    for orc in (True, False):
        r = evaluate(m, w, oracle=orc)
        res[f's{si}_{"oracle" if orc else "past"}'] = {k: v for k, v in r.items() if isinstance(v, float)}
res['params'] = m.phys.values(); res['hist'] = h.tolist()
json.dump(res, open(f'res_{tag}.json', 'w'))
print('done', json.dumps(res['params']))
