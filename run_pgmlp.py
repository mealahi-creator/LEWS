import torch, json, time; torch.set_num_threads(1)
from pgmlp import *
from data_pipeline import load_device, storm_catalogue
g = load_device('../data/thailand/pinn_108_new.csv').loc['2025-11-01 10:40':]; S = storm_catalogue(g)
res = {}
for seed in range(3):
    for oracle in (True, False):
        for held in range(len(S)):
            t = time.time(); r = run_fold(g, S, held, oracle, seed)
            res[f'seed{seed}_{"oracle" if oracle else "past"}_s{held}'] = r
            print(seed, oracle, held, round(r['skill_storm_30'], 3), round(r['skill_storm_60'], 3), round(r['skill_storm_120'], 3), f'{time.time()-t:.0f}s', flush=True)
json.dump(res, open('res3_pgmlp.json', 'w')); print('done')
