import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 13,
})
def normalize_bpa_from_triplet(triplet):
    T, I, F = triplet; s = T + I + F
    if s <= 0: return {"G":0.0, "B":0.0, "Θ":1.0}
    return {"G": T/s, "B": F/s, "Θ": I/s}
def dst_combine_two(m1, m2):
    inter = {("G","G"): "G", ("G","Θ"): "G", ("Θ","G"): "G",
             ("B","B"): "B", ("B","Θ"): "B", ("Θ","B"): "B",
             ("Θ","Θ"): "Θ", ("G","B"): None, ("B","G"): None}
    K = 0.0; m = {"G":0.0, "B":0.0, "Θ":0.0}
    for a, ma in m1.items():
        for b, mb in m2.items():
            s = inter.get((a,b), None)
            if s is None: K += ma*mb
            else: m[s] += ma*mb
    if K >= 1.0 - 1e-12:
        return {"G": m1["G"], "B": m1["B"], "Θ": m1["Θ"]}
    scale = 1.0/(1.0 - K)
    for k in m: m[k] *= scale
    return m
def dst_fuse_triplets(triplets_list):
    bp = normalize_bpa_from_triplet(triplets_list[0])
    for t in triplets_list[1:]:
        bp = dst_combine_two(bp, normalize_bpa_from_triplet(t))
    return (bp["G"], bp["Θ"], bp["B"]), bp
def inflate_to_ivns(triplet, delta):
    T, I, F = triplet
    def clip_pair(x):
        a = max(0.0, x - delta); b = min(1.0, x + delta)
        if b < a: a, b = b, a
        return (a, b)
    return (clip_pair(T), clip_pair(I), clip_pair(F))
def normalize_ivns_matrix(ivns_mat):
    n_alt, n_crit = ivns_mat.shape[:2]
    out = np.empty((n_alt, n_crit), dtype=object)
    for j in range(n_crit):
        T_l = np.array([ivns_mat[i,j][0][0] for i in range(n_alt)])
        T_h = np.array([ivns_mat[i,j][0][1] for i in range(n_alt)])
        I_l = np.array([ivns_mat[i,j][1][0] for i in range(n_alt)])
        I_h = np.array([ivns_mat[i,j][1][1] for i in range(n_alt)])
        F_l = np.array([ivns_mat[i,j][2][0] for i in range(n_alt)])
        F_h = np.array([ivns_mat[i,j][2][1] for i in range(n_alt)])
        Tmn, Tmx = T_l.min(), T_h.max()
        Imn, Imx = I_l.min(), I_h.max()
        Fmn, Fmx = F_l.min(), F_h.max()
        Trng = (Tmx - Tmn) if Tmx > Tmn else 1.0
        Irng = (Imx - Imn) if Imx > Imn else 1.0
        Frng = (Fmx - Fmn) if Fmx > Fmn else 1.0
        for i in range(n_alt):
            (Tlo,Thi),(Ilo,Ihi),(Flo,Fhi) = ivns_mat[i,j]
            Ta, Tb = (Tlo - Tmn)/Trng, (Thi - Tmn)/Trng
            T_pair = (min(Ta,Tb), max(Ta,Tb))
            Ia, Ib = (Imx - Ilo)/Irng, (Imx - Ihi)/Irng
            I_pair = (min(Ia,Ib), max(Ia,Ib))
            Fa, Fb = (Fmx - Flo)/Frng, (Fmx - Fhi)/Frng
            F_pair = (min(Fa,Fb), max(Fa,Fb))
            out[i,j] = (T_pair, I_pair, F_pair)
    return out
def topsis_rc_from_ivns_benefit(ivns_norm, weights):
    n_alt, n_crit = ivns_norm.shape[:2]
    w = np.asarray(weights, float); w = w / w.sum()
    mid = np.zeros((n_alt, n_crit, 3))
    for i in range(n_alt):
        for j in range(n_crit):
            (Tlo,Thi),(Ilo,Ihi),(Flo,Fhi) = ivns_norm[i,j]
            mid[i,j,0] = 0.5*(Tlo+Thi); mid[i,j,1] = 0.5*(Ilo+Ihi); mid[i,j,2] = 0.5*(Flo+Fhi)
    Tc, Ic, Fc = mid[:,:,0], mid[:,:,1], mid[:,:,2]
    Tp, Tm = Tc.max(axis=0), Tc.min(axis=0)
    Ip, Im = Ic.max(axis=0), Ic.min(axis=0)
    Fp, Fm = Fc.max(axis=0), Fc.min(axis=0)
    Dp = np.zeros(n_alt); Dm = np.zeros(n_alt)
    for i in range(n_alt):
        d2p = 0.0; d2m = 0.0
        for j in range(n_crit):
            d2p += w[j]*((mid[i,j,0]-Tp[j])**2 + (mid[i,j,1]-Ip[j])**2 + (mid[i,j,2]-Fp[j])**2)
            d2m += w[j]*((mid[i,j,0]-Tm[j])**2 + (mid[i,j,1]-Im[j])**2 + (mid[i,j,2]-Fm[j])**2)
        Dp[i] = np.sqrt(d2p); Dm[i] = np.sqrt(d2m)
    return Dm / (Dm + Dp + 1e-12)
def run_pipeline(experts_triplets, alternatives, criteria, delta=0.05, weights=None):
    base = experts_triplets[0].copy(); n_alt, n_crit = base.shape
    if weights is None: weights = np.ones(n_crit)/n_crit
    fused = np.empty((3,5), dtype=object)
    for i in range(3):
        for j in range(5):
            (T,I,F), _ = dst_fuse_triplets([E[i,j] for E in experts_triplets])
            fused[i,j] = (T,I,F)
    ivns = np.empty((3,5), dtype=object)
    for i in range(3):
        for j in range(5):
            ivns[i,j] = inflate_to_ivns(fused[i,j], delta)
    rc = topsis_rc_from_ivns_benefit(normalize_ivns_matrix(ivns), weights)
    return rc
def perturb_triplet(tif, sigma, rng):
    T,I,F = tif
    return (float(np.clip(T + rng.normal(0, sigma), 0, 1)),
            float(np.clip(I + rng.normal(0, sigma), 0, 1)),
            float(np.clip(F + rng.normal(0, sigma), 0, 1)))
def make_experts(baseline, p, sigma, rng):
    experts = []
    for _ in range(p):
        arr = np.empty((3,5), dtype=object)
        for i in range(3):
            for j in range(5): arr[i,j] = perturb_triplet(baseline[i,j], sigma, rng)
        experts.append(arr)
    return experts
alternatives = ["Treatment A", "Treatment B", "Treatment C"]
criteria = ["Effectiveness", "Side Effects", "Affordability", "Recovery Time", "Patient Satisfaction"]
ndm = np.empty((3,5), dtype=object)
data = [[(0.8,0.1,0.1),(0.6,0.2,0.2),(0.5,0.3,0.2),(0.7,0.2,0.1),(0.9,0.05,0.05)],
        [(0.9,0.05,0.05),(0.8,0.1,0.1),(0.6,0.2,0.2),(0.85,0.1,0.05),(0.85,0.1,0.05)],
        [(0.95,0.02,0.03),(0.9,0.05,0.05),(0.7,0.15,0.15),(0.9,0.05,0.05),(0.95,0.02,0.03)]]
for i in range(3):
    for j in range(5): ndm[i,j] = data[i][j]
num_sim = 1000; rng = np.random.default_rng(123)
ranks_data = np.zeros((num_sim, 3), dtype=int)
for s in range(num_sim):
    rc = run_pipeline(make_experts(ndm, 3, 0.05, rng), alternatives, criteria)
    ranks_data[s, np.argsort(-rc)] = np.arange(1, 4)
freq_matrix = np.zeros((3, 3))
for a in range(3): freq_matrix[a, :] = pd.Series(ranks_data[:, a]).value_counts().reindex([1,2,3], fill_value=0).values
fig, ax = plt.subplots(figsize=(10.5, 6.2))
colors = ["#440154", "#21918C", "#FDE725"]
for i in range(3): ax.bar(np.array([1,2,3]) + (i-1)*0.2, freq_matrix[i, :], 0.18, label=alternatives[i], color=colors[i], edgecolor="black")
ax.set_xlabel("Rank"); ax.set_ylabel("Frequency"); ax.set_xlim(0.5, 3.5)
ax.set_xticks(np.arange(0.5, 4.0, 0.5)); ax.set_ylim(0, 1000); ax.set_yticks(np.arange(0, 1200, 200))
ax.legend(title="Alternatives")
plt.tight_layout(); plt.savefig("figures/png/MC_RankFreq_1000_v2.png", dpi=300)
