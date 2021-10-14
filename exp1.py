# Experiment on Tractability

from models.ncm import NCM
from models.tncm import TNCM
from models.ncm import MLP
from models.tncm import SPN
import numpy as np
import torch
import time, itertools
import matplotlib.pyplot as plt
import seaborn as sns


def create_chain_adj(N):
    adj = np.zeros((N,N))
    for i in range(N-1):
        adj[i,i+1] = 1.
    return adj

def create_chain_adj_with_increasing_complexity(N):
    adj = np.zeros((N,N))
    for i in range(N-1):
        adj[:i+1,i+1] = 1.
    return adj

def normalize(ys):
    ymin = np.min(ys)
    ymax = np.max(ys)
    return [(y-ymin)/(ymax - ymin) for y in ys]

def timed_marginal_inference(ncm, samples=1, doX=-1, Xi=-1, debug=False, restrict=False):
    t0 = time.time()
    N = len(ncm.V)
    ind_d = N-1
    vals = []
    for val in [0, 1]:
        domains = [[0, 1]] * (N - 1)
        domains.insert(ind_d, [val])
        combinations = np.stack([x for x in itertools.product(*domains)])
        p_comb = []
        for ind, c in enumerate(combinations):
            c = torch.tensor(c, dtype=torch.float).unsqueeze(0)
            pC = ncm.forward(c, torch.tensor([doX] * samples).unsqueeze(1), Xi, samples, debug)
            p_comb.append(pC)
            if restrict:
                break
        vals.append(sum(p_comb).item())
    t = time.time() - t0
    return t, vals


seed_list = [0]#, 4, 304]#, 606]

exp1_data = {'NCM': {}, 'TNCM': {}}
N_start = 3
# N_end = 12
# for N in range(N_start, N_end):
#     exp1_data['NCM'].update({N: {}})
#     exp1_data['TNCM'].update({N: {}})
#     for seed in seed_list:
#         adj = create_chain_adj(N)
#         ncm = NCM(adj=adj)
#         t_ncm, _ = timed_marginal_inference(ncm)
#         exp1_data['NCM'][N].update({seed: t_ncm})
#         tncm = TNCM(adj=adj, spn_type="Regular")
#         t_tncm, _ = timed_marginal_inference(tncm)
#         exp1_data['TNCM'][N].update({seed: t_tncm})
#     print(f'Exp1  {N+1}/{N_end}    ', end="\r", flush=True)
#
# # visualize intractability of parameterized SCM
# sns.reset_orig()
# fig = plt.figure(figsize=(10,7))
# for ncm_type in ['NCM', 'TNCM']:
#     y = [np.mean(list(exp1_data[ncm_type][i].values())) for i in range(N_start, N_end)]
#     yerr = [np.std(list(exp1_data[ncm_type][i].values())) for i in range(N_start, N_end)]
#     x = range(N_start, N_end)
#     #plt.plot(x, y, label=ncm_type)
#     plt.errorbar(x, y, yerr=yerr, label=ncm_type)
#     plt.gca().set_yscale('log')
# plt.legend()
# plt.grid(True, which="both", ls="-")
# plt.show()



exp2_data = {'NCM': {}, 'TNCM': {}}
# adj = create_chain_adj(N_start)
# N_end = 300#1000
# for N in range(N_start, N_end):
#     exp2_data['NCM'].update({N: {}})
#     exp2_data['TNCM'].update({N: {}})
#     for seed in seed_list:
#         #adj = create_chain_adj_with_increasing_complexity(N)
#         ncm = NCM(adj=adj, scale=N)#True)
#         t_ncm, _ = timed_marginal_inference(ncm, restrict=True)
#         exp2_data['NCM'][N].update({seed: t_ncm})
#         tncm = TNCM(adj=adj, spn_type="Regular", scale=N)#True)
#         t_tncm, _ = timed_marginal_inference(tncm, restrict=True)
#         exp2_data['TNCM'][N].update({seed: t_tncm})
#     print(f'Exp2  {N+1}/{N_end}    ', end="\r", flush=True)

# visualize the relativitity of the intractability
# sns.set_theme()
# fig = plt.figure(figsize=(10,7))
# ys = []
# for ncm_type in ['NCM', 'TNCM']:
#     y = np.array([np.mean(list(exp2_data[ncm_type][i].values())) for i in range(N_start, N_end)])
#     ys.append(y)
#     yerr = np.array([np.std(list(exp2_data[ncm_type][i].values())) for i in range(N_start, N_end)])
#     x = range(N_start, N_end)
#     p = plt.plot(x, y, label=ncm_type)
#     #plt.errorbar(x, y, yerr=yerr, label=ncm_type)
#     pc = p[0].get_color()
#     plt.fill_between(x, y - yerr, y + yerr, alpha=0.5, facecolor=pc)
# lin = lambda x: 0.00001 * np.array(x)
# y = lin(x)
# ys.append(y)
# plt.plot(x, lin(x), color="black")
# maxes = np.array([np.max(g) for g in ys])
# minofmax = np.min(maxes)
# secondmaxofmax = np.partition(maxes.flatten(), -2)[-2]
# cand = secondmaxofmax
# plt.gca().set_ylim(0 - 0.05 * cand, cand + 0.05 * cand)
# plt.gca().set_xlim(0 - 0.05 * len(x), len(x) + 0.05 * len(x))
# #plt.gca().set_aspect('equal', adjustable='datalim')
# plt.gca().set_aspect(1.0/plt.gca().get_data_ratio(), adjustable='box')
# plt.legend()
# plt.grid(True, which="both", ls="-")
# plt.show()






# visualize the quadratic nature of MLP
N_start = 3
N_end = 500
window = 30
seed_list = [0, 4, 304]#, 606]
exp2_data.update({'MLP': {}})
ts = []
for N in range(N_start, N_end):
    exp2_data['MLP'].update({N: {}})
    for seed in seed_list:
        adj = create_chain_adj_with_increasing_complexity(N)
        #ncm = NCM(adj=adj, scale=True)
        mlp = MLP(len(adj), 1, [10*len(adj) for _ in range(3)])
        t0 = time.time()
        #ncm.forward(torch.ones(1,len(adj)), torch.tensor([-1] * 1).unsqueeze(1), -1, 1, False)
        mlp(torch.ones(1,len(adj)))
        t1 = time.time() - t0
        exp2_data['MLP'][N].update({seed: t1})
        ts.append(t1)
    if len(ts) > window:
        avg_time = np.mean(ts[-window:])
    else:
        avg_time = -1
    print(f'Exp MLP  {N+1}/{N_end}     {avg_time:.4f}', end="\r", flush=True)

if False:
    sns.set_theme()
    ncm_type = "MLP"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = [np.mean(list(exp2_data[ncm_type][i].values())) for i in range(N_start, N_end)]
    yerr = [np.std(list(exp2_data[ncm_type][i].values())) for i in range(N_start, N_end)]
    x = range(N_start, N_end)
    plt.plot(x, y, label=ncm_type)
    #lin = lambda x: np.min(y)*x
    #plt.plot(x, lin(x))
    plt.legend()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.show()

# visualize the linear nature of SPN
from models.EinsumNetwork import Graph, EinsumNetwork
exp2_data.update({'SPN': {}})
ts = []
for N in range(N_start, N_end):
    exp2_data['SPN'].update({N: {}})
    for seed in seed_list:
        adj = create_chain_adj_with_increasing_complexity(N)
        spn = SPN(D=len(adj), C=10+len(adj), K=len(adj)) # note the plus here opposed to times since the C internally squares! so with plus we have linear scaling of network size
        # spn = EinsumNetwork.EinsumNetwork(
        #     Graph.random_binary_trees(num_var=len(adj), depth=1,
        #                               num_repetitions=3),
        #     EinsumNetwork.Args(
        #         num_classes=1,
        #         num_input_distributions=5*len(adj),
        #         exponential_family=EinsumNetwork.CategoricalArray,
        #         exponential_family_args={'K': 2},
        #         num_sums=2*len(adj),
        #         num_var=len(adj),
        #         online_em_frequency=1,
        #         online_em_stepsize=0.05)
        # )
        # spn.initialize()
        t0 = time.time()
        spn(torch.ones(1,len(adj)))
        t1 = time.time() - t0
        exp2_data['SPN'][N].update({seed: t1})
        ts.append(t1)
    if len(ts) > window:
        avg_time = np.mean(ts[-window:])
    else:
        avg_time = -1
    print(f'Exp SPN  {N+1}/{N_end}     {avg_time:.4f}', end="\r", flush=True)

if False:
    sns.set_theme()
    ncm_type = "SPN"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = [np.mean(list(exp2_data[ncm_type][i].values())) for i in range(N_start, N_end)]
    yerr = [np.std(list(exp2_data[ncm_type][i].values())) for i in range(N_start, N_end)]
    x = range(N_start, N_end)
    plt.plot(x, y, label=ncm_type)
    #plt.errorbar(x, y, yerr=yerr, label=ncm_type)
    #lin = lambda x: np.min(y)*x
    #plt.plot(x, lin(x))
    plt.legend()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.show()


types = ['MLP', 'SPN', 'Lin']
colors = ['blue', 'orange', 'black']
for type in types[:-1]:
    y = [np.mean(list(exp2_data[type][i].values())) for i in range(N_start, N_end)]
    yerr = [np.std(list(exp2_data[type][i].values())) for i in range(N_start, N_end)]
    exp2_data[type].update({'Mean': y, 'Std': yerr})
sns.set_theme()
fig,axs = plt.subplots(1,4, figsize=(18,6))
lin = lambda x: 0.00001 * np.array(x) #np.mean([exp2_data['MLP']['Mean'][0]/N_start,exp2_data['SPN']['Mean'][0]/N_start]) * x
for ind, a in enumerate(axs):
    x = range(N_start, N_end)
    if ind == len(axs)-1:
        for i in range(3):
            #x = normalize(x)
            y = exp2_data[types[i]]['Mean'] if i != 2 else lin(x)
            std = np.array(exp2_data[types[i]]['Std']) if i != 2 else None
            # = normalize(y)
            p = a.plot(x, y, label=types[i], color=colors[i])
            if i != 2:
                pc = p[0].get_color()
                a.fill_between(x, y-std, y+std, alpha=0.5, facecolor=pc)
            a.set_xlim(0, len(x)+.05*len(x))
            maxes = np.array([np.max(g) for g in [exp2_data[t]['Mean'] for t in types[:-1]] + lin(x)])
            minofmax = np.min(maxes)
            secondmaxofmax = np.partition(maxes.flatten(), -2)[-2]
            cand = secondmaxofmax
            #a.set_ylim(0 - .05*cand, cand+.05*cand)
            #a.set_xlim(0 - .05*len(x), len(x)+.05*len(x))
            #a.set_aspect(1.0/a.get_data_ratio(), adjustable='box')
            a.set_aspect(1.0/a.get_data_ratio(), adjustable='box')
    elif ind == len(axs)-2:
        #x = normalize(x)
        a.plot(x, lin(x), label=types[ind], color=colors[ind])
        #a.set_ylim(0 - .05*len(x), len(x)+.05*len(x))
        #a.set_xlim(0 - .05*len(x), len(x)+.05*len(x))
        #a.set_aspect(1.0/a.get_data_ratio(), adjustable='box')
        a.set_aspect(1.0/a.get_data_ratio(), adjustable='box')
    else:
        y = exp2_data[types[ind]]['Mean']
        std = np.array(exp2_data[types[ind]]['Std'])
        #y = normalize(y)
        #x = normalize(x)
        p = a.plot(x, y, label=types[ind], color=colors[ind])
        pc = p[0].get_color()
        a.fill_between(x, y-std, y+std, alpha=0.5, facecolor=pc)
        cand = np.max(y)
        #a.set_aspect('equal', adjustable='datalim')
        a.set_aspect(1.0/a.get_data_ratio(), adjustable='box')
    a.legend()
#plt.tight_layout()
plt.show()

ind = 0
a = plt.gca()
y = exp2_data[types[ind]]['Mean']
std = np.array(exp2_data[types[ind]]['Std'])
p = a.plot(x, y, label=types[ind], color=colors[ind])
pc = p[0].get_color()
a.fill_between(x, y - std, y + std, alpha=0.5, facecolor=pc)
cand = np.max(y)
a.set_aspect(1.0 / a.get_data_ratio(), adjustable='box')
#a.set_aspect('equal', adjustable='box')
plt.show()


# visualize base time complexity comparison
# base_speed = 0.1
# lin = lambda x: base_speed * x
# qua = lambda x: base_speed * x**2
# cub = lambda x: base_speed * x**3
# exp = lambda x: base_speed * 2 ** x
# n = 10
# x = np.arange(1,n)
# names = ['Linear', 'Quadratic', 'Cubic', 'Exponential']
# for ind, f in enumerate([lin, qua, cub, exp]):
#     plt.plot(x, f(x), label=names[ind])
#     #plt.gca().set_aspect('equal', adjustable='datalim')
# plt.xlim(0,n)
# plt.legend()
# plt.show()