# ATE Estimation and JSD

from glob import glob
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

class D(dict):
    def __missing__(self, key):
        self[key] = D()
        return self[key]

p_ncm = "./experiments/NCM/20211012-210344/*/*/Marginals.pkl"
p_tncm = "./experiments/TNCM/20211012-210317_TNCM/*/*/Marginals.pkl"
p_gt_ate = "./datasets/SCMs/*_ATE.csv"
p_gt_l1 = "./datasets/SCMs/*_L1.csv"
p_gt_l2_dX0 = "./datasets/SCMs/*_doX0.csv"
p_gt_l2_dX1 = "./datasets/SCMs/*_doX1.csv"

d = D()
for ind, p in enumerate(glob(p_ncm) + glob(p_tncm)):
    with open(p, "rb") as f:
        M = pickle.load(f)
        ncm_type = "TNCM" if "TNCM" in p else "NCM"
        scm_type = p.split("/")[-3].split("SCM")[0]
        params = p.split("/")[-3].split("_")[1].split("p")[0]
        seed = p.split("/")[-2].split("-")[1]
        d[ncm_type][scm_type][params][seed] = M
    print(f' {ind+1}/{len(p_ncm)+len(p_tncm)}   ', end="\r", flush=True)
d_gt_ate = D()
for p in glob(p_gt_ate):
    scm_type = p.split("SCM")[1].split("s/")[1]
    params = p.split("p")[0].split("_")[1]
    d_gt_ate[scm_type][params] = pd.read_csv(p).iloc[0,1]

def get_gt_distr(p_gt):
    d_gt = D()
    for p in glob(p_gt):
        scm_type = p.split("SCM")[1].split("s/")[1]
        params = p.split("p")[0].split("_")[1]
        d_gt[scm_type][params] = np.array(pd.read_csv(p))[:,1:]
    return d_gt
d_gt_l1 = get_gt_distr(p_gt_l1)
d_gt_l2_dX0 = get_gt_distr(p_gt_l2_dX0)
d_gt_l2_dX1 = get_gt_distr(p_gt_l2_dX1)



# violin plots for ATE
#tips = sns.load_dataset("tips")
#ax = sns.violinplot(x=tips["total_bill"])
fig, axs = plt.subplots(1,2,figsize=(10,5))
d_ate = D()
for ind, ncm_type in enumerate(d.keys()):
    df = pd.DataFrame(columns=["SCM", "ATE Error"])
    err_means = []
    for scm_type in d[ncm_type].keys():
        errs = []
        for params in d[ncm_type][scm_type].keys():
            tncm_pY1dX0 = np.mean([d[ncm_type][scm_type][params][seed]['L2_doX0'][ncm_type][1][1] for seed in d[ncm_type][scm_type][params].keys()])
            tncm_pY1dX1 = np.mean([d[ncm_type][scm_type][params][seed]['L2_doX1'][ncm_type][1][1] for seed in d[ncm_type][scm_type][params].keys()])
            tncm_ate = tncm_pY1dX1 - tncm_pY1dX0
            gt_ate = d_gt_ate[scm_type][params]
            err_ate = abs(gt_ate - tncm_ate)
            errs.append([scm_type, err_ate])
        err_means.append(np.mean([t[1] for t in errs]))
        df = df.append(pd.DataFrame(errs, columns=["SCM", "ATE Error"]))
    d_ate[ncm_type] = df

    sns.violinplot(x="SCM", y="ATE Error", data=df, cut=0, ax=axs[ind])
    maxes = np.array([np.max(g) for g in err_means])
    minofmax = np.min(maxes)
    secondmaxofmax = np.partition(maxes.flatten(), -2)[-2]
axs[0].set_ylim(0., .1)
axs[1].set_ylim(0., .1)
plt.show()


# JSD table
#distance.jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0)
d_distr = D()
distr_cols = ["SCM", "JSD L1", "JSD L2 do(X=0)", "JSD L2 do(X=1)"]
for ind, ncm_type in enumerate(d.keys()):
    df = pd.DataFrame(columns=distr_cols)
    for scm_type in d[ncm_type].keys():
        errs = []
        for params in d[ncm_type][scm_type].keys():
            l1 = [d[ncm_type][scm_type][params][seed]['L1'][ncm_type] for seed in d[ncm_type][scm_type][params].keys()]
            l1 = np.hstack((np.mean(np.array(pd.DataFrame(l1).applymap(lambda x: x[0])),axis=0),np.mean(np.array(pd.DataFrame(l1).applymap(lambda x: x[1])),axis=0))) # prob vector with x1=0, x2=0...x1=1,...xn=1
            l2_dX0 = [d[ncm_type][scm_type][params][seed]['L2_doX0'][ncm_type] for seed in d[ncm_type][scm_type][params].keys()]
            l2_dX0 = np.hstack((np.mean(np.array(pd.DataFrame(l2_dX0).applymap(lambda x: x[0])),axis=0),np.mean(np.array(pd.DataFrame(l2_dX0).applymap(lambda x: x[1])),axis=0)))
            l2_dX1 = [d[ncm_type][scm_type][params][seed]['L2_doX1'][ncm_type] for seed in d[ncm_type][scm_type][params].keys()]
            l2_dX1 = np.hstack((np.mean(np.array(pd.DataFrame(l2_dX1).applymap(lambda x: x[0])),axis=0),np.mean(np.array(pd.DataFrame(l2_dX1).applymap(lambda x: x[1])),axis=0)))

            gt_l1 = np.mean(d_gt_l1[scm_type][params],axis=0)
            gt_l1 = np.hstack((1-gt_l1, gt_l1))
            gt_l2_dX0 = np.mean(d_gt_l2_dX0[scm_type][params],axis=0)
            gt_l2_dX0 = np.hstack((1-gt_l2_dX0, gt_l2_dX0))
            gt_l2_dX1 = np.mean(d_gt_l2_dX1[scm_type][params],axis=0)
            gt_l2_dX1 = np.hstack((1-gt_l2_dX1, gt_l2_dX1))

            err_distr_l1 = distance.jensenshannon(l1, gt_l1, 2.0)
            err_distr_l2_dX0 = distance.jensenshannon(l2_dX0, gt_l2_dX0, 2.0)
            err_distr_l2_dX1 = distance.jensenshannon(l2_dX1, gt_l2_dX1, 2.0)
            errs.append([scm_type, err_distr_l1, err_distr_l2_dX0, err_distr_l2_dX1])
        df = df.append(pd.DataFrame(errs, columns=distr_cols))
    d_distr[ncm_type] = df
for ncm_type in d.keys():
    for scm_type in d[ncm_type].keys():
        print(f"NCM: {ncm_type}\t SCM: {scm_type}\t\nMeans:\n{np.mean(d_distr[ncm_type].loc[d_distr[ncm_type]['SCM'] == scm_type])}")