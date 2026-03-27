import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "axes.titlecolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

def save_figure(fig, name):
    output_dir_pdf = 'figures/pdf'
    output_dir_svg = 'figures/svg'
    output_dir_png = 'figures/png'
    os.makedirs(output_dir_pdf, exist_ok=True)
    os.makedirs(output_dir_svg, exist_ok=True)
    os.makedirs(output_dir_png, exist_ok=True)
    fig.savefig(os.path.join(output_dir_pdf, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir_svg, f"{name}.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir_png, f"{name}.png"), bbox_inches="tight")

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
            mid[i,j,0] = 0.5*(Tlo+Thi)
            mid[i,j,1] = 0.5*(Ilo+Ihi)
            mid[i,j,2] = 0.5*(Flo+Fhi)
    Tcol, Icol, Fcol = mid[:,:,0], mid[:,:,1], mid[:,:,2]
    T_plus,  T_minus = Tcol.max(axis=0), Tcol.min(axis=0)
    I_plus,  I_minus = Icol.max(axis=0), Icol.min(axis=0)
    F_plus,  F_minus = Fcol.max(axis=0), Fcol.min(axis=0)
    Dp = np.zeros(n_alt); Dm = np.zeros(n_alt)
    for i in range(n_alt):
        d2p = 0.0; d2m = 0.0
        for j in range(n_crit):
            d2p += w[j]*((mid[i,j,0]-T_plus[j])**2 + (mid[i,j,1]-I_plus[j])**2 + (mid[i,j,2]-F_plus[j])**2)
            d2m += w[j]*((mid[i,j,0]-T_minus[j])**2 + (mid[i,j,1]-I_minus[j])**2 + (mid[i,j,2]-F_minus[j])**2)
        Dp[i] = np.sqrt(d2p); Dm[i] = np.sqrt(d2m)
    RC = Dm / (Dm + Dp + 1e-12)
    return RC

def perturb_triplet(tif, sigma, rng):
    T,I,F = tif
    return (float(np.clip(T + rng.normal(0, sigma), 0, 1)),
            float(np.clip(I + rng.normal(0, sigma), 0, 1)),
            float(np.clip(F + rng.normal(0, sigma), 0, 1)))

def make_experts_from_baseline(baseline_ndm, p, sigma, rng):
    n_alt, n_crit = baseline_ndm.shape[:2]
    experts = []
    for _ in range(p):
        arr = np.empty((n_alt, n_crit), dtype=object)
        for i in range(n_alt):
            for j in range(n_crit):
                arr[i,j] = perturb_triplet(baseline_ndm[i,j], sigma, rng)
        experts.append(arr)
    return experts

def run_pipeline_for_single_ndm(ndm_mat, weights, delta=0.05):
    n_alt, n_crit = ndm_mat.shape[:2]
    ivns = np.empty((n_alt, n_crit), dtype=object)
    for i in range(n_alt):
        for j in range(n_crit):
            ivns[i,j] = inflate_to_ivns(ndm_mat[i,j], delta)
    ivns_norm = normalize_ivns_matrix(ivns)
    return topsis_rc_from_ivns_benefit(ivns_norm, weights)

alternatives = ["Treatment A", "Treatment B", "Treatment C"]
criteria = ["Effectiveness", "Side Effects", "Affordability", "Recovery Time", "Patient Satisfaction"]
data = [
    [(0.8, 0.1, 0.1), (0.6, 0.2, 0.2), (0.5, 0.3, 0.2), (0.7, 0.2, 0.1), (0.9, 0.05, 0.05)],
    [(0.9, 0.05, 0.05), (0.8, 0.1, 0.1), (0.6, 0.2, 0.2), (0.85, 0.1, 0.05), (0.85, 0.1, 0.05)],
    [(0.95, 0.02, 0.03), (0.9, 0.05, 0.05), (0.7, 0.15, 0.15), (0.9, 0.05, 0.05), (0.95, 0.02, 0.03)],
]
ndm = np.empty((3, 5), dtype=object)
for i in range(3):
    for j in range(5):
        ndm[i,j] = data[i][j]

num_sim = 1000
rng = np.random.default_rng(123)
weights_equal = np.ones(len(criteria))/len(criteria)

fused_rc_store = np.zeros((num_sim, 3))
expert_ranks_store = np.zeros((num_sim, 3, 3), dtype=int)
final_ranks = np.zeros((num_sim, 3), dtype=int)

for s in range(num_sim):
    experts_s = make_experts_from_baseline(ndm, 3, 0.05, rng)
    for e_idx, E in enumerate(experts_s):
        rc_e = run_pipeline_for_single_ndm(E, weights_equal)
        order_e = np.argsort(-rc_e)
        expert_ranks_store[s, e_idx, order_e] = np.arange(1, 4)
    fused_ndm = np.empty((3, 5), dtype=object)
    for i in range(3):
        for j in range(5):
            (T,I,F), _ = dst_fuse_triplets([Ex[i,j] for Ex in experts_s])
            fused_ndm[i,j] = (T,I,F)
    rc_fused = run_pipeline_for_single_ndm(fused_ndm, weights_equal)
    fused_rc_store[s, :] = rc_fused
    order_f = np.argsort(-rc_fused)
    final_ranks[s, order_f] = np.arange(1, 4)

print("Visualizing Results...")

fig1, ax1 = plt.subplots(figsize=(10, 6))
rank_levels = [1, 2, 3]
final_freqs = np.zeros((3, 3))
for a_idx in range(3):
    final_freqs[a_idx, :] = pd.Series(final_ranks[:, a_idx]).value_counts().reindex(rank_levels, fill_value=0).values
x_ranks = np.array(rank_levels)
width = 0.2
colors = ["#440154", "#21918C", "#FDE725"]
for a_idx, alt in enumerate(alternatives):
    ax1.bar(x_ranks + (a_idx - 1) * width, final_freqs[a_idx, :], width, label=alt, color=colors[a_idx], edgecolor="black")
ax1.set_xlabel("Rank", fontweight="bold")
ax1.set_ylabel("Frequency", fontweight="bold")
ax1.set_xlim(0.5, 3.5)
ax1.set_xticks(np.arange(0.5, 4.0, 0.5))
ax1.set_ylim(0, 1000)
ax1.set_yticks(np.arange(0, 1200, 200))
ax1.legend(title="Alternatives")
save_figure(fig1, "Final_Rank_Frequency_Grouped")

bins = np.linspace(0, 1.0, 11)
bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]
df_rc = pd.DataFrame(fused_rc_store, columns=alternatives)
binned_counts = pd.DataFrame(index=bin_labels)
for alt in alternatives:
    binned_counts[alt] = pd.cut(df_rc[alt], bins=bins, labels=bin_labels, include_lowest=True).value_counts().reindex(bin_labels, fill_value=0)
fig2, ax2 = plt.subplots(figsize=(12, 7))
x = np.arange(len(bin_labels))
for i, alt in enumerate(alternatives):
    rects = ax2.bar(x + (i - 1) * 0.25, binned_counts[alt], 0.25, label=alt, color=colors[i], edgecolor="black")
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax2.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_xlabel("Relative Closeness (RC) Score Range", fontweight="bold")
ax2.set_ylabel("Frequency", fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(bin_labels, rotation=45)
ax2.legend(title="Alternatives")
plt.tight_layout()
save_figure(fig2, "Final_Binned_Score_Grouped")

plt.figure(figsize=(10, 6))
for a_idx, alt in enumerate(alternatives):
    sns.histplot(fused_rc_store[:, a_idx], bins=50, kde=True, label=alt, color=colors[a_idx], alpha=0.5, element="step")
plt.xlabel("RC Score")
plt.ylabel("Density")
plt.legend(title="Alternatives")
sns.despine()
plt.tight_layout()
save_figure(plt.gcf(), "Final_Score_Histogram")

print("All results saved and visualizations organized.")