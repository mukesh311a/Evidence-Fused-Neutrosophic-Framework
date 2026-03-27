import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set font and HD resolution
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# Core Logic Functions (Reused from original script)

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
    n_alt, n_crit = ivns_mat.shape
    out = np.empty_like(ivns_mat, dtype=object)
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
    n_alt, n_crit = ivns_norm.shape
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

def run_pipeline_expertfusion_to_rank(experts_triplets, alternatives, criteria, delta=0.05, weights=None):
    base = experts_triplets[0].copy()
    n_alt, n_crit = base.shape
    if weights is None: weights = np.ones(n_crit)/n_crit
    fused = np.empty_like(base, dtype=object)
    for i in range(n_alt):
        for j in range(n_crit):
            (T,I,F), _ = dst_fuse_triplets([E[i,j] for E in experts_triplets])
            fused[i,j] = (T,I,F)
    ivns = np.empty_like(fused, dtype=object)
    for i in range(n_alt):
        for j in range(n_crit):
            ivns[i,j] = inflate_to_ivns(fused[i,j], delta)
    ivns_norm = normalize_ivns_matrix(ivns)
    rc = topsis_rc_from_ivns_benefit(ivns_norm, weights)
    return rc

def perturb_triplet(tif, sigma, rng):
    T,I,F = tif
    return (float(np.clip(T + rng.normal(0, sigma), 0, 1)),
            float(np.clip(I + rng.normal(0, sigma), 0, 1)),
            float(np.clip(F + rng.normal(0, sigma), 0, 1)))

def make_experts_from_baseline(baseline_ndm, alternatives, criteria, p=3, sigma=0.03, rng=None):
    if rng is None: rng = np.random.default_rng()
    n_alt = len(alternatives); n_crit = len(criteria)
    experts = []
    for _ in range(p):
        arr = np.empty((n_alt, n_crit), dtype=object)
        for i in range(n_alt):
            for j in range(n_crit):
                arr[i,j] = perturb_triplet(baseline_ndm[i,j], sigma, rng)
        experts.append(arr)
    return experts

# --- Main Parameters ---

alternatives = ["Treatment A", "Treatment B", "Treatment C"]
criteria = ["Effectiveness", "Side Effects", "Affordability", "Recovery Time", "Patient Satisfaction"]
ndm = np.array([
    [(0.8, 0.1, 0.1), (0.6, 0.2, 0.2), (0.5, 0.3, 0.2), (0.7, 0.2, 0.1), (0.9, 0.05, 0.05)],
    [(0.9, 0.05, 0.05), (0.8, 0.1, 0.1), (0.6, 0.2, 0.2), (0.85, 0.1, 0.05), (0.85, 0.1, 0.05)],
    [(0.95, 0.02, 0.03), (0.9, 0.05, 0.05), (0.7, 0.15, 0.15), (0.9, 0.05, 0.05), (0.95, 0.02, 0.03)]
], dtype=object)

num_sim = 100
num_experts = 3
sigma_mc = 0.05
rng = np.random.default_rng(123)
weights_equal = np.ones(len(criteria))/len(criteria)

ranks_data = np.zeros((num_sim, len(alternatives)), dtype=int)

for s in range(num_sim):
    experts_s = make_experts_from_baseline(ndm, alternatives, criteria, num_experts, sigma_mc, rng)
    rc = run_pipeline_expertfusion_to_rank(experts_s, alternatives, criteria, delta=0.05, weights=weights_equal)
    order = np.argsort(-rc)
    ranks_data[s, order] = np.arange(1, len(alternatives)+1)

# Frequency Calculation
rank_levels = [1, 2, 3]
freq_matrix = np.zeros((len(alternatives), len(rank_levels)))

for a, alt in enumerate(alternatives):
    counts = pd.Series(ranks_data[:, a]).value_counts().reindex(rank_levels, fill_value=0)
    freq_matrix[a, :] = counts.values

# --- Visualization: Grouped Bar Chart ---

x = np.arange(len(alternatives))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

colors = ["#0072B2", "#E69F00", "#009E73"]
for i, r in enumerate(rank_levels):
    ax.bar(x + (i - 1) * width, freq_matrix[:, i], width, label=f"Rank {r}", color=colors[i], edgecolor="black")

ax.set_xlabel("Alternative", fontname="Cambria")
ax.set_ylabel("Frequency (N=100)", fontname="Cambria")
ax.set_title("Rank Frequency Distribution (100 Monte Carlo Trials)", fontname="Cambria", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(alternatives, fontname="Cambria")
ax.legend(title="Ranks", prop={"family": "Cambria"})

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save HD PNG
output_path = "figures/png/MC_RankFreq_100_Grouped.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"Chart saved to {output_path}")

# Optional show for debug
# plt.show()
