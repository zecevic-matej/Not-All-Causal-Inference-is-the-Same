from models.tncm import TNCM
import torch
import numpy as np
import pandas as pd
import time, os, sys
import itertools
import matplotlib.pyplot as plt
import random
from aux.helper import Logger
from datetime import datetime
import pickle

class Config():
    def __init__(self):
        self.max_epochs = 3
        self.batch_size = 1
        self.loss_running_window = 1000
        self.seeds = [0, 4, 304, 606]
        self.samples = 200
        self.loss_int_weight = 0.
        self.loss_int_weight_decay = False

        self.load_model = None
        self.save_viz = True
        self.animate = False
        self.dir_exp = "./experiments/TNCM/"


cfg = Config()

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
    train_data = torch.utils.data.DataLoader(L1, batch_size=cfg.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    meta = {'Model': desc_m, 'N': N, 'U_Params': U_params, 'adj': adj, 'ATE': ATE, 'L1': L1, 'L2_doX0': doX0, 'L2_doX1': doX1}
    return meta, train_data

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

spn_type = "Regular" #"EinSum"

ps = [

    "ChainSCM_[0.1 0.8 0.7 0.8]p_N10000",
    "ChainSCM_[0.4 0.6 0.5 0.8]p_N10000",
    "ChainSCM_[0.5 0.7 0.6 0.5]p_N10000",

    "BackdoorSCM_[0.1 0.8 0.7 0.8]p_N10000",
    "BackdoorSCM_[0.4 0.6 0.5 0.8]p_N10000",
    "BackdoorSCM_[0.5 0.7 0.6 0.5]p_N10000",

    "ColliderSCM_[0.1 0.8 0.7 0.8]p_N10000",
    "ColliderSCM_[0.4 0.6 0.5 0.8]p_N10000",
    "ColliderSCM_[0.5 0.7 0.6 0.5]p_N10000",

    "ConfounderSCM_[0.1 0.8 0.7 0.8]p_N10000",
    "ConfounderSCM_[0.4 0.6 0.5 0.8]p_N10000",
    "ConfounderSCM_[0.5 0.7 0.6 0.5]p_N10000",

      ]
cfg.dir_exp = os.path.join(cfg.dir_exp, datetime.now().strftime("%Y%m%d-%H%M%S") + "_TNCM")
for p in ps:

    scm_model = p.split("SCM_")[0]
    dir_exp = os.path.join(cfg.dir_exp, p)

    for seed in cfg.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        meta, train_data = load_dataset(seed, p)

        ncm = TNCM(adj=meta['adj'], spn_type=spn_type)

        ncm.print_graph()
        print(f'Topological Order: {ncm.indices_to_names(ncm.topology)}')

        gt_marginals = compute_gt_marginals(meta, 'L1')

        if cfg.load_model is None:

            dir_seed = os.path.join(dir_exp, f"seed-{seed}")
            dir_seed_models = os.path.join(dir_seed, 'models')
            txt_meta = os.path.join(dir_seed, f'Experiment-Log-seed-{seed}.txt')
            if not os.path.exists(dir_seed_models):
                os.makedirs(dir_seed_models)
            sys.stdout = Logger(txt_meta)

            print(f'Config = {cfg.__dict__}\n')
            print(f'Data Set Size = {len(train_data)}')
            print(f'Meta Data; Model {meta["Model"]}\t U_params {meta["U_Params"]}\t GT ATE {meta["ATE"]}')

            #optimizer = torch.optim.SGD(ncm.params(), lr=0.2, weight_decay=0.0)
            if spn_type != "EinSum":
                optimizer = torch.optim.Adam(ncm.params())

            train_ds = cycle(train_data)
            best_loss_tr =  -100000. if spn_type == "EinSum" else 100000 # TODO: make EiNet also minimize #100000
            loss_int_weight = cfg.loss_int_weight

            plt.clf()
            plt.close()
            animate = cfg.animate

            loss_per_step = []
            loss_components_per_step = []
            running_losses_tr = []
            running_losses_va = []
            running_losses_components_tr = []
            running_losses_components_va = []
            for step in range(cfg.max_epochs): # epochs now
                t0 = time.time()

                for i in range(len(train_data)):

                    batch = next(train_ds)

                    if spn_type != "EinSum":
                        for V in ncm.V:
                            ncm.S[V].zero_grad()

                    pV_est = ncm.forward(batch, samples=cfg.samples, doX=torch.tensor([-1.]*cfg.samples).unsqueeze(1), Xi=-1) # -1 means no intervention, clean up!!
                    # this is the special case where y:=v and thus we only need to count how often we have the v in the data
                    # reps = len(torch.where((train_data.dataset == datapoint).all(axis=1))[0])
                    # pV_int_est = ncm.forward(datapoint, samples=cfg.samples, doX=torch.tensor([0.] * cfg.samples).unsqueeze(1), Xi=0)  # TODO: clean the doX interface
                    # pV_int_mass = reps * pV_int_est
                    # pV_int_mass = torch.where(pV_int_mass == 0., torch.tensor(0.)+1e-6, pV_int_mass)

                    if spn_type == "EinSum":
                        nll_l1 = torch.log(pV_est) # TODO: correct this, it is not nll but ll, but the other code forces it to be named like this for the moment
                        nll_l2 = -1 * torch.zeros(1)
                        loss = nll_l1
                    else:
                        nll_l1 = -1 * torch.log(pV_est)
                        nll_l2 = -1 * torch.zeros(1)#* torch.log(pV_int_mass)
                        if torch.isnan(nll_l1) or torch.isnan(nll_l2):
                            import pdb; pdb.set_trace()
                        loss = nll_l1 + loss_int_weight * nll_l2
                        #print(f"Reps: {reps}\t Weight: {loss_int_weight}")
                    loss.backward()

                    if spn_type == "EinSum":
                        for V in ncm.V:
                            ncm.S[V].em_process_batch()
                    else:
                        optimizer.step()

                    if cfg.loss_int_weight_decay:
                        loss_int_weight = 1/np.log(step + 3) # 1/log(3) = .9
                        # not introducing stopping, since it probably won't matter
                        if loss_int_weight < .001:
                            loss_int_weight = .001

                    loss_per_step.append(loss.item())
                    loss_components_per_step.append((nll_l1.item(), nll_l2.item()))

                    if i!= 0 and int(i * cfg.batch_size) % cfg.loss_running_window == 0:
                        t1 = time.time()

                        running_loss_tr = np.mean(loss_per_step[-cfg.loss_running_window:])
                        running_loss_components_tr = np.mean(np.array(loss_components_per_step)[-cfg.loss_running_window:,:],axis=0)
                        running_losses_components_tr.append((running_loss_components_tr[0],running_loss_components_tr[1]))

                        if spn_type == "EinSum":
                            condition = lambda x,y: x > y
                        else:
                            condition = lambda x,y: x < y

                        if condition(running_loss_tr,best_loss_tr): # TODO: make EiNet also minimize
                            # uncomment this for non-valid # best_valid_elbo = train_elbo
                            best_loss_tr = running_loss_tr
                            #suffix = ''
                            suffix = f' [Saved Model.]'
                            states = {
                                "SPNs": [s.state_dict() for s in ncm.S.values()],
                            }
                            best_model_name = f'NCM-seed-{seed}-ep-{step}-i-{i}'
                            torch.save(states, os.path.join(dir_seed_models, best_model_name))
                        else:
                            suffix = ""

                        print(
                            f"Epoch {step:<10d}/{cfg.max_epochs} I {i}/{len(train_data)}\t"
                            f"Train NLL: {running_loss_tr:<5.3f} (L1 {running_loss_components_tr[0]:<5.3f}, L2 {running_loss_components_tr[1]:<5.3f})\t"
                            f"" + suffix
                        )
                        t0 = t1

                if spn_type == "EinSum":
                    for V in ncm.V:
                        ncm.S[V].em_update()

        checkpoint = torch.load(os.path.join(dir_seed_models,best_model_name) if cfg.load_model is None else cfg.load_model)
        for ind, s in enumerate(ncm.S.values()):
            s.load_state_dict(checkpoint["SPNs"][ind])
        print(f'Loaded best model.')

        marginals = {'Model': os.path.join(dir_seed_models,best_model_name)}

        pred_marginals = ncm.compute_marginals(samples=cfg.samples)
        plot_marginals(pred_marginals, gt_marginals, seed, save=(dir_seed, '-best') if cfg.save_viz or cfg.load_model is None else None)
        marginals.update({'L1': {'GT': gt_marginals, 'TNCM': pred_marginals}})
        print(f'Computed Marginals for L1 distribution.')

        # intervention
        pred_marginals = ncm.compute_marginals(samples=cfg.samples, doX=1., Xi=0)
        plot_marginals(pred_marginals, compute_gt_marginals(meta, 'L2_doX1'), seed, save=(dir_seed, '-doX1') if cfg.save_viz or cfg.load_model is None else None)
        marginals.update({'L2_doX1': {'GT': gt_marginals, 'TNCM': pred_marginals}})
        print(f'Computed Marginals for do(X=1).')

        # intervention
        pred_marginals = ncm.compute_marginals(samples=cfg.samples, doX=0., Xi=0)
        plot_marginals(pred_marginals, compute_gt_marginals(meta, 'L2_doX0'), seed, save=(dir_seed, '-doX0') if cfg.save_viz or cfg.load_model is None else None)
        marginals.update({'L2_doX0': {'GT': gt_marginals, 'TNCM': pred_marginals}})
        print(f'Computed Marginals for do(X=0).')

        with open(os.path.join(dir_seed, "Marginals.pkl"), "wb") as f:
            pickle.dump(marginals, f)
            print('Saved Marginals.')

        if cfg.load_model is not None:
            break