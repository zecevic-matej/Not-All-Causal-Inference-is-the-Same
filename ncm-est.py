from models.ncm import NCMConfounder
import torch
import numpy as np
import pandas as pd
import time, os
import itertools
import matplotlib.pyplot as plt
import random

class Config():
    def __init__(self):
        self.max_iterations = 3000#0
        self.log_interval = 50#0
        self.batch_size = 1
        self.loss_running_window = 50
        self.seeds = [4]#[0, 4, 304, 606]
        self.samples = 1000
        self.load_model = True#False
        self.save_viz = False#True
        self.animate = False
        self.dir_exp = "./experiments/NCM/"


cfg = Config()
print(f'Config = {cfg.__dict__}\n')

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def compute_gt_marginals(meta, distr='L1'):
    gt_marginals = {}
    for i in range(meta[distr].shape[1]):
        p1 = (meta[distr][:,i].sum() / meta[distr].shape[0]).item()
        gt_marginals.update({i: (1-p1, p1)})
    return gt_marginals

def load_dataset(seed, p, data_dir="./datasets/SCMs/"):
    desc_m, U_params_str, N_str = p.split("_")
    U_params = [float(x) for x in U_params_str[1:-2].split(" ")]
    N = int(N_str[1:])
    p = os.path.join(data_dir, p)

    adj = np.array(pd.read_csv(p+'_adj.csv'))[:,1:]
    ATE = np.array(pd.read_csv(p+'_ATE.csv'))[:,1:][0][0]
    L1 = torch.tensor(np.array(pd.read_csv(p+'_L1.csv'))[:,1:],dtype=torch.float)
    doX0 = torch.tensor(np.array(pd.read_csv(p+'_doX0.csv'))[:,1:],dtype=torch.float)
    doX1 = torch.tensor(np.array(pd.read_csv(p+'_doX1.csv'))[:,1:],dtype=torch.float)

    n_train = int(len(L1) - 0.2 * len(L1))
    n_other = int(len(L1) - n_train)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_set, val_set, test_set = torch.utils.data.random_split(L1, [n_train, int(n_other / 2), int(n_other / 2)])
    train_data = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valid_data = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    meta = {'Model': desc_m, 'N': N, 'U_Params': U_params, 'adj': adj, 'ATE': ATE, 'L1': L1, 'L2_doX0': doX0, 'L2_doX1': doX1}
    print(f'Train Size = {len(train_set)}\t Valid Size = {len(val_set)}\t Test Size = {len(test_set)}')
    print(f'Meta Data; Model {desc_m}\t U_params {U_params}\t GT ATE {ATE}')
    return meta, train_data, valid_data, test_data

def plot_marginals(pred, gt, seed, figsize=(13,9), running_losses=None, animate=False, save=None):
    plt.clf()
    N = len(gt)
    s = 3 if running_losses is not None else 2
    fig, axs = plt.subplots(s,N, figsize=figsize)
    for ind, a in enumerate(axs.flatten()):
        if running_losses is not None:
            if ind >= 2*N:
                for t in running_losses:
                    a.plot(range(len(t[0])), t[0], label=t[1])
                a.legend()
                break
        if ind >= N:
            marginals = gt
            color = 'black'
        else:
            marginals = pred
            color = 'blue'
        a.bar([0,1], marginals[ind % N], facecolor=color)
        a.set_title(f'Var:{ind} (p0={marginals[ind % N][0]:.2f},p1={marginals[ind % N][1]:.2f})')
        a.set_xlim(0,1)
        a.set_xticks([0,1])
        a.set_ylim(0,1)
    plt.suptitle(f'NCM (Seed {seed})')
    plt.tight_layout()
    if animate:
        plt.pause(0.001)
        #input("Press [enter] to continue.")
    elif save is not None:
        plt.savefig(os.path.join(save[0], f'Viz-seed-{seed}{save[1]}.png'))
    else:
        plt.show()

#p = "ChainSCM_[0.4 0.6 0.5 0.8]p_N10000"
p = "ConfounderSCM_[0.1 0.8 0.7 0.8]p_N10000"

scm_model = p.split("SCM_")[0]
cfg.dir_exp = os.path.join(cfg.dir_exp, scm_model)
if not os.path.exists(cfg.dir_exp):
    os.makedirs(cfg.dir_exp)

ncm_func = {
    #'Chain': NCMChain,
    'Confounder': NCMConfounder
}

for seed in cfg.seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    meta, train_data, valid_data, test_data = load_dataset(seed, p)

    ncm = ncm_func[scm_model](adj=meta['adj'])

    gt_marginals = compute_gt_marginals(meta, 'L1')

    if not cfg.load_model:

        optimizer = torch.optim.Adam(ncm.params())
        loss_int_weight = 1.

        train_ds = cycle(train_data)
        valid_val = 100000
        best_valid_val = valid_val + 1

        plt.clf()
        plt.close()
        animate = cfg.animate

        loss_per_step = []
        loss_components_per_step = []
        running_losses_tr = []
        running_losses_va = []
        running_losses_components_tr = []
        running_losses_components_va = []
        for step in range(cfg.max_iterations):
            t0 = time.time()

            batch = next(train_ds)
            for V in ncm.V:
                ncm.S[V].zero_grad()

            pV_est = ncm.forward(batch, samples=cfg.samples, doX=torch.tensor([-1.]*cfg.samples).unsqueeze(1)) # -1 means no intervention, clean up!!
            # pV_int_mass = torch.zeros((1,))
            # for i in torch.where((train_data.dataset.dataset == batch).all(axis=1))[0]: # specifically do(X=0)
            #     pV_int_est = ncm.forward(batch, doX=torch.tensor([0.]*1000).unsqueeze(1)) # TODO: clean the doX interface
            #     pV_int_mass += pV_int_est
            #     #print(f'    {i}/{len(train_data)}    ', end="\r", flush=True)
            # this is the special case where y:=v and thus we only need to count how often we have the v in the data
            reps = len(torch.where((train_data.dataset.dataset == batch).all(axis=1))[0])
            pV_int_est = ncm.forward(batch, samples=cfg.samples, doX=torch.tensor([0.] * cfg.samples).unsqueeze(1))  # TODO: clean the doX interface
            pV_int_mass = reps * pV_int_est
            pV_int_mass = torch.where(pV_int_mass == 0., torch.tensor(0.)+1e-6, pV_int_mass)

            nll_l1 = -1 * torch.log(pV_est)
            nll_l2 = -1 * torch.log(pV_int_mass)
            if torch.isnan(nll_l1) or torch.isnan(nll_l2):
                import pdb; pdb.set_trace()
            loss = nll_l1 + 0. * nll_l2
            #print(f"Reps: {reps}\t Weight: {loss_int_weight}")
            loss.backward()
            optimizer.step()

            loss_int_weight = 1/np.log(step + 3) # 1/log(3) = .9
            # not introducing stopping, since it probably won't matter
            if loss_int_weight < .001:
                loss_int_weight = .001

            loss_per_step.append(loss.item())
            loss_components_per_step.append((nll_l1.item(), nll_l2.item()))
            if step % cfg.log_interval == 0:
                t1 = time.time()
                examples_per_sec = cfg.log_interval * cfg.batch_size / (t1 - t0)
                with torch.no_grad():
                    valid_ds = cycle(valid_data)
                    valid_vals = []
                    valid_components_vals = []
                    # TODO: repetition essentially of above, isolate
                    for _ in range(cfg.loss_running_window): # TODO: make this a true validation on all val data
                        val_batch = next(valid_ds)
                        pV_est_val = ncm.forward(val_batch, samples=cfg.samples, doX=torch.tensor([-1.]*cfg.samples).unsqueeze(1))

                        # pV_int_mass_val = torch.zeros((1,))
                        # val_ds_int = cycle(valid_data)
                        # batch_val = next(val_ds_int)
                        # for i in torch.where((valid_data.dataset.dataset == batch_val).all(axis=1))[0]: # TODO: note this is the same, since dataset.dataset is everything together
                        #     pV_int_est_val = ncm.forward(batch_val, doX=torch.tensor([0.]*1000).unsqueeze(1))
                        #     pV_int_mass_val += pV_int_est_val
                        val_ds_int = cycle(valid_data)
                        batch_val = next(val_ds_int)
                        reps_val = len(torch.where((valid_data.dataset.dataset == batch_val).all(axis=1))[0])
                        pV_int_est_val = ncm.forward(batch_val, samples=cfg.samples, doX=torch.tensor([0.] * cfg.samples).unsqueeze(1))  # TODO: clean the doX interface
                        pV_int_mass_val = reps_val * pV_int_est_val
                        pV_int_mass_val = torch.where(pV_int_mass_val == 0., torch.tensor(0.)+1e-6, pV_int_mass_val)

                        val_nll_l1 = -1 * torch.log(pV_est_val)
                        val_nll_l2 = -1 * torch.log(pV_int_mass_val)
                        val_loss = val_nll_l1 + 0. * val_nll_l2
                        valid_vals.append(val_loss)
                        valid_components_vals.append((val_nll_l1.item(), val_nll_l2.item()))

                    if animate:
                        #pred_marginals = ncm.compute_marginals(samples=1000)
                        pred_marginals = ncm.compute_marginals(samples=cfg.samples, doX=0.)
                        gt_marginals = compute_gt_marginals(meta, 'L2_doX0')


                running_loss_tr = np.mean(loss_per_step[-cfg.loss_running_window:])
                running_loss_va = np.mean(valid_vals[-cfg.loss_running_window:])
                running_loss_components_tr = np.mean(np.array(loss_components_per_step)[-cfg.loss_running_window:,:],axis=0)
                running_loss_components_va = np.mean(np.array(valid_components_vals)[-cfg.loss_running_window:,:],axis=0)
                running_losses_components_tr.append((running_loss_components_tr[0],running_loss_components_tr[1]))
                running_losses_components_va.append((running_loss_components_va[0],running_loss_components_va[1]))

                if running_loss_va < best_valid_val:
                    # uncomment this for non-valid # best_valid_elbo = train_elbo
                    best_valid_val = running_loss_va
                    #suffix = ''
                    suffix = f' [Saved Model.]'
                    states = {
                        "MLPs": [s.state_dict() for s in ncm.S.values()],
                    }
                    torch.save(states, os.path.join(cfg.dir_exp, f'NCM-seed-{seed}'))
                else:
                    suffix = ""

                if animate:
                    plot_marginals(pred_marginals, gt_marginals, seed, running_losses=[(running_losses_tr,'Train'), (running_losses_va,'Valid')], animate=True)

                print(
                    f"Step {step:<10d}\t"
                    f"Train NLL: {running_loss_tr:<5.3f} (L1 {running_loss_components_tr[0]:<5.3f}, L2 {running_loss_components_tr[1]:<5.3f})\t"
                    f"Validation NLL: {running_loss_va:<5.3f} (L1 {running_loss_components_va[0]:<5.3f}, L2 {running_loss_components_va[1]:<5.3f})\t"
                    f"Speed: {examples_per_sec:<5.2e} examples/s" + suffix
                )
                t0 = t1
        print(f'Training Complete.')
        pred_marginals = ncm.compute_marginals(samples=cfg.samples)
        plot_marginals(pred_marginals, gt_marginals, seed,
                       running_losses=[(np.array(running_losses_components_tr)[:,0],'Train L1'), (np.array(running_losses_components_tr)[:,1],'Train L2')], #[(running_losses_tr,'Train'), (running_losses_va,'Valid')],
                       save=(cfg.dir_exp, '-end'))

    # for i in range(100):
    #     plt.close()

    checkpoint = torch.load(os.path.join(cfg.dir_exp,f'NCM-seed-{seed}'))
    for ind, s in enumerate(ncm.S.values()):
        s.load_state_dict(checkpoint["MLPs"][ind])
    print(f'Loaded best model.')

    pred_marginals = ncm.compute_marginals(samples=cfg.samples)
    plot_marginals(pred_marginals, gt_marginals, seed, save=(cfg.dir_exp, '-best') if cfg.save_viz or not cfg.load_model else None)
    print(f'Computed Marginals for L1 distribution.')

    # intervention
    pred_marginals = ncm.compute_marginals(samples=cfg.samples, doX=1.)
    plot_marginals(pred_marginals, compute_gt_marginals(meta, 'L2_doX1'), seed, save=(cfg.dir_exp, '-doX1') if cfg.save_viz or not cfg.load_model else None)
    print(f'Computed Marginals for do(X=1).')

    # intervention
    pred_marginals = ncm.compute_marginals(samples=cfg.samples, doX=0.)
    plot_marginals(pred_marginals, compute_gt_marginals(meta, 'L2_doX0'), seed, save=(cfg.dir_exp, '-doX0') if cfg.save_viz or not cfg.load_model else None)
    print(f'Computed Marginals for do(X=0).')




# Code below was used to Debug the Positivity Violation of Confounder SCM
# which rendered the task impossible due to non-identifiability for any parameterization of the SCM with that
# specific type of structural equations (was due to the X = and(U_X, Z), also same for = or(..), but fine for xor(..))
ps = ["ConfounderSCM_[0.1 0.8 0.7 0.8]p_N10000",
    # "ConfounderSCM_[0.4 0.6 0.5 0.8]p_N10000",
    # "ConfounderSCM_[0.5 0.7 0.6 0.5]p_N10000",
    # "ConfounderSCM_[0.6 0.8 0.2 0.2]p_N10000",
    # "ConfounderSCM_[0.9 0.4 0.7 0.5]p_N10000"
      ]
for p in ps:
    meta, train_data, valid_data, test_data = load_dataset(seed, p)
    l1 = compute_gt_marginals(meta, 'L1')
    l2dx0 = compute_gt_marginals(meta, 'L2_doX0')
    M = meta['L1'].shape[0]

    # positivity violation check for Confounder SCM
    # l1pz = lambda i: l1[2][i]
    # l1pxz = lambda x,z: len(torch.where((meta['L1'][:,[0,2]] == torch.tensor([x,z])).all(axis=1))[0])/M
    # l1px0gz = [l1pxz(0.,0.) / l1pz(0), l1pxz(0.,1.) / l1pz(1)]
    # l1px1gz = [l1pxz(1.,0.) / l1pz(0), l1pxz(1.,1.) / l1pz(1)]
    # print(p)
    # print(f'p(X=0|Z)={l1px0gz}\t p(X=1|Z)={l1px1gz}\t Sums={[l1px0gz[0]+l1px1gz[0], l1px0gz[1]+l1px1gz[1]]}')

    # backdoor adjustment validation via statistical terms, vs. conditional
    l1pxy = lambda x,y: len(torch.where((meta['L1'][:,[0,1]] == torch.tensor([x,y])).all(axis=1))[0])/M
    l1pxz = lambda x,z: len(torch.where((meta['L1'][:,[0,2]] == torch.tensor([x,z])).all(axis=1))[0])/M
    l1pygxz = lambda y, x,z: (len(torch.where((meta['L1'][:,[0,1,2]] == torch.tensor([x,y,z])).all(axis=1))[0])/M) / l1pxz(x,z)
    l1px = lambda x: len(torch.where((meta['L1'][:,[0]] == torch.tensor([x])).all(axis=1))[0])/M
    l1pzgx = lambda z, x: (len(torch.where((meta['L1'][:,[0,2]] == torch.tensor([x,z])).all(axis=1))[0])/M) / l1px(x)
    l1pz = lambda z: len(torch.where((meta['L1'][:,[2]] == torch.tensor([z])).all(axis=1))[0])/M

    l1py0gx0z = [l1pygxz(0., 0., 0.),l1pygxz(0., 0., 1.)]
    l1py1gx0z = [l1pygxz(1., 0., 0.),l1pygxz(1., 0., 1.)]
    l1pzgx0 = [l1pzgx(0., 0.), l1pzgx(1., 0.)]
    l1pz_ = [l1pz(0.), l1pz(1.)]

    print(f'Conditional:\n'
          f'p(Y=0|X=0) = p(X=0,Y=0)/p(X=0) = {l1pxy(0.,0.)}/{l1px(0.)} = {l1pxy(0.,0.)/l1px(0.)}\n'
          f'           = Sum_Z( p(Y=0|X=0,Z)p(Z|X=0)\n'
          f'           = Sum({l1py0gx0z}, {l1pzgx0})\n'
          f'           = {sum(np.array(l1py0gx0z)*np.array(l1pzgx0))}\n'
          f'p(Y=1|X=0) = p(X=0,Y=1)/p(X=0) = {l1pxy(0.,1.)}/{l1px(0.)} = {l1pxy(0.,1.)/l1px(0.)}\n'
          f'           = Sum_Z( p(Y=1|X=0,Z)p(Z|X=0)\n'
          f'           = Sum({l1py1gx0z}, {l1pzgx0})\n'
          f'           = {sum(np.array(l1py1gx0z)*np.array(l1pzgx0))}\n')

    print(f'Interventional:\n'
          f'p(Y=0|do(X=0)) = Ground Truth Counting in L2 distr. {l2dx0[1][0]}\n'
          f'               = Sum_Z( p(Y=0|X=0,Z)p(Z)\n'
          f'               = Sum({l1py0gx0z}, {l1pz_})\n'
          f'               = {sum(np.array(l1py0gx0z)*np.array(l1pz_))}\n'
          f'p(Y=1|do(X=0)) = Ground Truth Counting in L2 distr. {l2dx0[1][1]}\n'
          f'               = Sum_Z( p(Y=1|X=0,Z)p(Z)\n'
          f'               = Sum({l1py1gx0z}, {l1pz_})\n'
          f'               = {sum(np.array(l1py1gx0z)*np.array(l1pz_))}\n')



# Visualize the Structural Equations learned by the NCM
# and compare to the original SCM!
samples = 1 # if bernoulli random variables, then this is not necessary anymore since only {0,1} opposed to [0,1)
Z0 = ncm.S['Z'](torch.zeros((samples, 1)))
Z1 = ncm.S['Z'](torch.ones((samples, 1)))
print(f'Z = f_Z(U_Z=0) = {Z0}\n'
      f'    f_Z(U_Z=1) = {Z1}')
X00 = ncm.S['X'](torch.cat((torch.zeros((samples, 1)),torch.zeros((samples, 1))),axis=1))
X01 = ncm.S['X'](torch.cat((torch.zeros((samples, 1)),torch.ones((samples, 1))),axis=1))
X10 = ncm.S['X'](torch.cat((torch.ones((samples, 1)),torch.zeros((samples, 1))),axis=1))
X11 = ncm.S['X'](torch.cat((torch.ones((samples, 1)),torch.ones((samples, 1))),axis=1))
print(f'X = f_X(U_X=0, Z=0) = {torch.mean(X00)}\n'
      f'  = f_X(U_X=0, Z=1) = {torch.mean(X01)}\n'
      f'  = f_X(U_X=1, Z=0) = {torch.mean(X10)}\n'
      f'  = f_X(U_X=1, Z=1) = {torch.mean(X11)}')

# Replacing the actual NN learned with the ground truth structural equation of the SCM
# to see whether everything is basically resolved by that - whether that MLP is the single problem
# ncm.S['X'] = lambda t: torch.where((t == torch.tensor([0., 0.])).all(axis=1), torch.tensor(-100.), torch.where((t == torch.tensor([1., 1.])).all(axis=1), torch.tensor(-100.), torch.tensor(100.))).unsqueeze(1)
#
# pred_marginals = ncm.compute_marginals(samples=cfg.samples)
# plot_marginals(pred_marginals, gt_marginals, seed)
# pred_marginals = ncm.compute_marginals(samples=cfg.samples, doX=1.)
# plot_marginals(pred_marginals, compute_gt_marginals(meta, 'L2_doX1'), seed)
# pred_marginals = ncm.compute_marginals(samples=cfg.samples, doX=0.)
# plot_marginals(pred_marginals, compute_gt_marginals(meta, 'L2_doX0'), seed)