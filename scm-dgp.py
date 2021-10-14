from models.scm import *
import os

np.random.seed(0)
params = [np.round(np.random.uniform(0.1, 0.9, 4),decimals=1) for _ in range(5)]
print(f'Random Parameterizations: {params}')
scms = [BackdoorSCM]
base_dir = './datasets/SCMs/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
for scm in scms:
    for selected in range(len(params)):
        scm1 = scm(U_params=params[selected])
        if selected == 0:
            print(f'>>> Starting with {scm1.name}\n '
                  f'- Adjacency:\n{scm1.adj}')
        n_samples = 10000
        scm1.sample(n_samples)
        ate, l2x1, l2x0 = scm1.ate(n_samples)
        base_name = f'{scm1.name}_{params[selected]}p_N{int(n_samples)}'
        scm1.l1.to_csv(os.path.join(base_dir, base_name + '_L1' + '.csv'))
        l2x1.to_csv(os.path.join(base_dir, base_name + '_doX1' + '.csv'))
        l2x0.to_csv(os.path.join(base_dir, base_name + '_doX0' + '.csv'))
        pd.DataFrame(np.round(np.array([ate])[:,np.newaxis],decimals=2)).to_csv(os.path.join(base_dir, base_name + '_ATE' + '.csv'))
        pd.DataFrame(scm1.adj).to_csv(os.path.join(base_dir, base_name + '_adj' + '.csv'))