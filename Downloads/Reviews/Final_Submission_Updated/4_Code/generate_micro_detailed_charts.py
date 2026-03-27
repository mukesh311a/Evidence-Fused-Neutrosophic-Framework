import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.rcParams.update({"font.family": "serif", "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"], "figure.dpi": 300, "savefig.dpi": 300, "font.size": 12})
def normalize_bpa_from_triplet(triplet):
    T, I, F = triplet; s = T + I + F
    if s <= 0: return {"G":0.0, "B":0.0, "Θ":1.0}
    return {"G": T/s, "B": F/s, "Θ": I/s}
def dst_combine_two(m1, m2):
    inter = {("G","G"): "G", ("G","Θ"): "G", ("Θ","G"): "G", ("B","B"): "B", ("B","Θ"): "B", ("Θ","B"): "B", ("Θ","Θ"): "Θ", ("G","B"): None, ("B","G"): None}
    K = 0.0; m = {"G":0.0, "B":0.0, "Θ":0.0}
    for a, ma in m1.items():
        for b, mb in m2.items():
            s = inter.get((a,b), None)
            if s is None: K += ma*mb
            else: m[s] += ma*mb
    if K >= 1.0 - 1e-12: return {"G": m1["G"], "B": m1["B"], "Θ": m1["Θ"]}
    scale = 1.0/(1.0 - K)
    for k in m: m[k] *= scale
    return m
def dst_fuse_triplets(triplets_list):
    bp = normalize_bpa_from_triplet(triplets_list[0])
    for t in triplets_list[1:]: bp = dst_combine_two(bp, normalize_bpa_from_triplet(t))
    return (bp["G"], bp["Θ"], bp["B"]), bp
def inflate_to_ivns(triplet, delta):
    T, I, F = triplet
    def clip_pair(x):
        a = max(0.0, x - delta); b = min(1.0, x + delta)
        return (min(a,b), max(a,b))
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
        Trng, Irng, Frng = max(Tmx-Tmn,1e-9), max(Imx-Imn,1e-9), max(Fmx-Fmn,1e-9)
        for i in range(n_alt):
            (Tlo,Thi),(Ilo,Ihi),(Flo,Fhi) = ivns_mat[i,j]
            out[i,j] = (((Tlo-Tmn)/Trng, (Thi-Tmn)/Trng), ((Imx-Ilo)/Irng, (Imx-Ihi)/Irng), ((Fmx-Flo)/Frng, (Fmx-Fhi)/Frng))
    return out
def topsis_rc(ivns_norm, weights):
    n_alt, n_crit = ivns_norm.shape[:2]; w = np.asarray(weights, float); w = w / w.sum()
    mid = np.zeros((n_alt, n_crit, 3))
    for i in range(n_alt):
        for j in range(n_crit):
            (Tlo,Thi),(Ilo,Ihi),(Flo,Fhi) = ivns_norm[i,j]
            mid[i,j,0], mid[i,j,1], mid[i,j,2] = 0.5*(Tlo+Thi), 0.5*(Ilo+Ihi), 0.5*(Flo+Fhi)
    Tp, Tm = mid[:,:,0].max(axis=0), mid[:,:,0].min(axis=0)
    Ip, Im = mid[:,:,1].max(axis=0), mid[:,:,1].min(axis=0)
    Fp, Fm = mid[:,:,2].max(axis=0), mid[:,:,2].min(axis=0)
    Dp, Dm = np.zeros(n_alt), np.zeros(n_alt)
    for i in range(n_alt):
        d2p, d2m = 0.0, 0.0
        for j in range(n_crit):
            d2p += w[j]*((mid[i,j,0]-Tp[j])**2+(mid[i,j,1]-Ip[j])**2+(mid[i,j,2]-Fp[j])**2)
            d2m += w[j]*((mid[i,j,0]-Tm[j])**2+(mid[i,j,1]-Im[j])**2+(mid[i,j,2]-Fm[j])**2)
        Dp[i], Dm[i] = np.sqrt(d2p), np.sqrt(d2m)
    return Dm / (Dm + Dp + 1e-12)
def run_pipe(ndm_mat, weights):
    n_alt, n_crit = ndm_mat.shape[:2]
    ivns = np.empty((n_alt, n_crit), dtype=object)
    for i in range(n_alt):
        for j in range(n_crit): ivns[i,j] = inflate_to_ivns(ndm_mat[i,j], 0.05)
    return topsis_rc(normalize_ivns_matrix(ivns), weights)
alternatives = ["Treatment A", "Treatment B", "Treatment C"]
criteria = ["Effectiveness", "Side Effects", "Affordability", "Recovery Time", "Patient Satisfaction"]
data = [[(0.8,0.1,0.1),(0.6,0.2,0.2),(0.5,0.3,0.2),(0.7,0.2,0.1),(0.9,0.05,0.05)],
        [(0.9,0.05,0.05),(0.8,0.1,0.1),(0.6,0.2,0.2),(0.85,0.1,0.05),(0.85,0.1,0.05)],
        [(0.95,0.02,0.03),(0.9,0.05,0.05),(0.7,0.15,0.15),(0.9,0.05,0.05),(0.95,0.02,0.03)]]
ndm = np.empty((3, 5), dtype=object)
for i in range(3):
    for j in range(5): ndm[i,j]=data[i][j]
num_sim = 1000; rng = np.random.default_rng(123); w_eq = np.ones(5)/5
f_rc = np.zeros((num_sim, 3))
for s in range(num_sim):
    exps = []
    for _ in range(3):
        arr = np.empty((3,5), dtype=object)
        for i in range(3):
            for j in range(5):
                T,I,F = ndm[i,j]; arr[i,j]=(float(np.clip(T+rng.normal(0,0.05),0,1)), float(np.clip(I+rng.normal(0,0.05),0,1)), float(np.clip(F+rng.normal(0,0.05),0,1)))
        exps.append(arr)
    fused = np.empty((3,5), dtype=object)
    for i in range(3):
        for j in range(5):
            (T,I,F), _ = dst_fuse_triplets([Ex[i,j] for Ex in exps])
            fused[i,j] = (T,I,F)
    f_rc[s, :] = run_pipe(fused, w_eq)
bins = np.linspace(0, 1.0, 11); labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]
b_df = pd.DataFrame(index=labels)
for i, alt in enumerate(alternatives): b_df[alt] = pd.cut(f_rc[:, i], bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
fig, ax = plt.subplots(figsize=(12, 7))
colors = ["#440154", "#21918C", "#FDE725"]
for i, alt in enumerate(alternatives):
    rects = ax.bar(np.arange(10) + (i-1)*0.25, b_df[alt], 0.25, label=alt, color=colors[i], edgecolor="black")
    for r in rects:
        h = r.get_height()
        if h > 0: ax.annotate(f'{int(h)}', xy=(r.get_x()+r.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xlabel("RC Score Range"); ax.set_ylabel("Frequency"); ax.set_xticks(np.arange(10)); ax.set_xticklabels(labels, rotation=45); ax.legend(title="Alternatives")
plt.tight_layout(); os.makedirs("figures/png", exist_ok=True); plt.savefig("figures/png/MC_Score_Grouped_Bar.png", dpi=300)
