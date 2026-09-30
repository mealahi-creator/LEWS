import numpy as np, json
H = 1  # 60 min
rows = {}
def load(k):
    c = np.load(f'conf_out/held_{k}.npz'); e = np.load(f'conf_out/extra_{k}.npz'); b = np.load(f'bts_out/{k}_held_nuts_oracle.npz')
    return dict(y=c['y'][:, H], ok=c['ym'][:, H], st=c['storm'], P={'Persistence': np.zeros(len(c['y'])), 'PG-MLP': c['PGMLP_oracle'][:, H],
               'M0': e['M0'][:, H], 'M2': c['M2_oracle'][:, H], 'M2np': e['M2np'][:, H], 'M3': e['M3'][:, H], 'B-TS': b['med'][:, H]})
F = [load(k) for k in range(7)]
def pooled(folds):
    out = {}
    for m in F[0]['P']:
        se = sp = 0
        for k in folds:
            f = F[k]; s = f['st'] & f['ok'] & ~np.isnan(f['P'][m]) & ~np.isnan(f['P']['PG-MLP'])
            se += np.sum((f['P'][m][s] - f['y'][s]) ** 2); sp += np.sum(f['y'][s] ** 2)
        out[m] = 1 - se / sp
    return out
print('skill all 7', {k: round(v, 2) for k, v in pooled(range(7)).items()})
print('skill ex2', {k: round(v, 2) for k, v in pooled([0, 1, 3, 4, 5, 6]).items()})
res = {}
for d in (0.005, 0.01, 0.02):
    res[d] = {}
    for m in F[0]['P']:
        if m == 'Persistence': continue
        CA = MA = FA = TN = 0
        for k in [0, 1, 3, 4, 5, 6]:
            f = F[k]; s = f['st'] & f['ok'] & ~np.isnan(f['P'][m]) & ~np.isnan(f['P']['PG-MLP'])
            o = f['y'][s] >= d; p = f['P'][m][s] >= d
            CA += np.sum(o & p); MA += np.sum(o & ~p); FA += np.sum(~o & p); TN += np.sum(~o & ~p)
        N = CA + MA + FA + TN
        res[d][m] = dict(events=int(CA + MA), N=int(N), POD=CA / max(CA + MA, 1), FAR=FA / max(CA + FA, 1), CSI=CA / max(CA + MA + FA, 1),
                         EI=(CA + TN) / N, MAR=MA / max(CA + MA, 1))
    print(d, {m: {k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for m, r in res[d].items()})
json.dump({str(k): v for k, v in res.items()}, open('thai_alerts.json', 'w'), indent=1, default=float)
