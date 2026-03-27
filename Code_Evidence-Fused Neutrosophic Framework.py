import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import matplotlib.font_manager as fm
import os


if os.path.exists('cambria.ttc'):
    fm.fontManager.addfont('cambria.ttc')
    cam_font = 'Cambria'
elif os.path.exists('cambria.ttf'):
    fm.fontManager.addfont('cambria.ttf')
    cam_font = 'Cambria'
else:
    print("I cannot find the Cambria font file.")
    print("Please make sure you uploaded 'cambria.ttc' or 'cambria.ttf' to the folder icon on the left!")
    cam_font = 'serif'


plt.rcParams.update({
    "font.family": cam_font,
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

def set_color_style():
    sns.set_theme(context="talk", style="whitegrid", rc={"font.family": cam_font})
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, # Enforcing HD mode
        "axes.linewidth": 1.0, "grid.color": "#000000", "grid.linewidth": 0.6,
        "font.family": cam_font, "font.size": 14,
        "axes.labelsize": 14, "axes.titlesize": 15,
        "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "axes.grid": True
    })

def fancy_3d(ax, draw_bbox=True, elev=22, azim=28):
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white"); pane.set_edgecolor("black"); pane.set_alpha(1.0)
    ax.set_proj_type('persp'); ax.view_init(elev=elev, azim=azim)
    try: ax.set_box_aspect((1.2, 1.0, 0.9))
    except: pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.85
        except: pass
    ax.tick_params(pad=14, length=6, width=1.2)
    ax.xaxis.labelpad = 22; ax.yaxis.labelpad = 26; ax.zaxis.labelpad = 26
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    if draw_bbox:
        x0, x1 = ax.get_xlim3d(); y0, y1 = ax.get_ylim3d(); z0, z1 = ax.get_zlim3d()
        corners = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                            [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        segs = [(corners[a], corners[b]) for a,b in edges]
        ax.add_collection3d(Line3DCollection(segs, colors='black', linewidths=1.2))

RESULTS_DIR = "results"
OUTPUTS_DIR = "outputs"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

import builtins
from IPython.display import display as _ipy_display

_text_log = []
_original_print = builtins.print
_original_display = _ipy_display


def print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    msg = sep.join(str(a) for a in args) + end
    _text_log.append(msg)
    _original_print(*args, **kwargs)


def display(obj):
    try:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            _text_log.append(obj.to_string() + "\n")
        else:
            _text_log.append(str(obj) + "\n")
    except Exception:
        pass
    return _original_display(obj)


def save_figure(fig, name):
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS_DIR, f"{name}.{fmt}"), bbox_inches="tight", dpi=300)


def write_text_outputs(path=os.path.join(OUTPUTS_DIR, "run_output.txt")):
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(_text_log))
    _original_print(f"Text output saved to: {path}")


alternatives = ["Treatment A", "Treatment B", "Treatment C"]
criteria = ["Effectiveness", "Side Effects", "Affordability", "Recovery Time", "Patient Satisfaction"]
ALT_SHORT = ["A", "B", "C"]; ALT_MAP = dict(zip(ALT_SHORT, alternatives))
ALT_ORDER = alternatives

ndm = np.array([
    [(0.8, 0.1, 0.1), (0.6, 0.2, 0.2), (0.5, 0.3, 0.2), (0.7, 0.2, 0.1), (0.9, 0.05, 0.05)],
    [(0.9, 0.05, 0.05), (0.8, 0.1, 0.1), (0.6, 0.2, 0.2), (0.85, 0.1, 0.05), (0.85, 0.1, 0.05)],
    [(0.95, 0.02, 0.03), (0.9, 0.05, 0.05), (0.7, 0.15, 0.15), (0.9, 0.05, 0.05), (0.95, 0.02, 0.03)]
], dtype=object)

df_ndm = pd.DataFrame(ndm.tolist(), columns=criteria, index=alternatives)
print("\nStep 1: Neutrosophic Decision Matrix (T,I,F)")
display(df_ndm)

df_T = df_ndm.map(lambda x: float(x[0]))

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    df_T, annot=True, fmt=".2f", cmap="viridis",
    linewidths=0.5, square=True, cbar_kws={"shrink": 0.8}, ax=ax,
    annot_kws={"fontfamily": cam_font}
)

ax.set_xlabel("", fontfamily=cam_font, fontweight="normal", color="#000000")
ax.set_ylabel("", fontfamily=cam_font, fontweight="normal", color="#000000")

for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontfamily(cam_font)
    t.set_fontweight("bold")
    t.set_color("#000000")

ax.set_title("", fontfamily=cam_font, fontweight="bold", color="#000000")

cbar = ax.collections[0].colorbar
for lbl in cbar.ax.get_yticklabels():
    lbl.set_fontfamily(cam_font)

plt.tight_layout()
save_figure(fig, "T_heatmap_quick")
plt.show()


# Second Heatmap definition
fig, ax = plt.subplots(figsize=(9.5, 6.0))

hm = sns.heatmap(
    df_T,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    linewidths=0.8,
    linecolor="white",
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax,
    annot_kws={
        "fontsize": 14,
        "fontweight": "bold",
        "color": "black",
        "fontfamily": cam_font
    }
)

ax.set_title("", fontsize=16, fontweight="bold", fontfamily=cam_font, pad=35)
ax.set_xlabel("", fontsize=14, fontfamily=cam_font, fontweight="normal")
ax.set_ylabel("", fontsize=14, fontfamily=cam_font, fontweight="normal")

ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=13, fontfamily=cam_font)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13, fontfamily=cam_font)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=13)
for lbl in cbar.ax.get_yticklabels():
    lbl.set_fontfamily(cam_font)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

plt.tight_layout()
save_figure(fig, "NDM_T_heatmap_cambria")
plt.show()


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

def iwns_scores_benefit(ivns_norm, weights, alpha=1.0, beta=1.0, gamma=1.0):
    n_alt, n_crit = ivns_norm.shape
    w = np.asarray(weights, float); w = w / w.sum()
    denom = (alpha + beta + gamma)
    out = np.zeros(n_alt)
    for i in range(n_alt):
        s = 0.0
        for j in range(n_crit):
            (Tlo,Thi),(Ilo,Ihi),(Flo,Fhi) = ivns_norm[i,j]
            Tm = 0.5*(Tlo+Thi); Im = 0.5*(Ilo+Ihi); Fm = 0.5*(Flo+Fhi)
            s += w[j] * (alpha*Tm + beta*Im + gamma*Fm) / denom
        out[i] = s
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

def run_pipeline_expertfusion_to_rank(experts_triplets, delta=0.05, weights=None,
                                      alpha=1.0, beta=1.0, gamma=1.0):
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

    iwns = iwns_scores_benefit(ivns_norm, weights, alpha, beta, gamma)

    rc = topsis_rc_from_ivns_benefit(ivns_norm, weights)

    return {"fused": fused, "ivns": ivns, "ivns_norm": ivns_norm, "iwns": iwns, "rc": rc}

rng = np.random.default_rng(0)
def perturb_triplet(tif, sigma=0.03):
    T,I,F = tif
    return (float(np.clip(T + rng.normal(0, sigma), 0, 1)),
            float(np.clip(I + rng.normal(0, sigma), 0, 1)),
            float(np.clip(F + rng.normal(0, sigma), 0, 1)))

def make_experts_from_baseline(baseline_ndm, p=3, sigma=0.03):
    n_alt = len(alternatives); n_crit = len(criteria)
    experts = []
    for _ in range(p):
        arr = np.empty((n_alt, n_crit), dtype=object)
        for i in range(n_alt):
            for j in range(n_crit):
                arr[i,j] = perturb_triplet(baseline_ndm[i,j], sigma)
        experts.append(arr)
    return experts

num_experts = 3; sigma_expert = 0.03
experts = make_experts_from_baseline(ndm, num_experts, sigma_expert)

weights_equal = np.ones(len(criteria))/len(criteria)
res = run_pipeline_expertfusion_to_rank(experts, delta=0.05, weights=weights_equal)
print("\nSingle-run IWNS (benefit, non-negative) and TOPSIS RC")
print(pd.Series(res["iwns"], index=alternatives).to_frame("IWNS (0–1)"))
print(pd.Series(res["rc"], index=alternatives).to_frame("RC (0–1)"))

weight_sets = np.array([
    [0.4, 0.1, 0.1, 0.2, 0.2],
    [0.2, 0.3, 0.1, 0.2, 0.2],
    [0.2, 0.1, 0.4, 0.2, 0.1],
    [0.2, 0.1, 0.1, 0.4, 0.2],
    [0.1, 0.1, 0.1, 0.2, 0.5],
], dtype=float)

sens_rows = []
for cid, w in enumerate(weight_sets, start=1):
    rc = topsis_rc_from_ivns_benefit(res["ivns_norm"], w)
    for i, alt in enumerate(alternatives):
        sens_rows.append({"Config": cid, "Alternative": alt, "RC": float(rc[i])})
df_sens_long = pd.DataFrame(sens_rows)
print("\nSensitivity (TOPSIS RC across weight sets)")
display(df_sens_long.pivot(index="Config", columns="Alternative", values="RC"))

plt.figure(figsize=(8,5))
sns.boxplot(data=df_sens_long, x="Alternative", y="RC")
plt.title("Sensitivity: TOPSIS Relative Closeness by Alternative", fontfamily=cam_font, fontweight="bold", pad=15)
plt.ylabel("RC (0–1)", fontweight='normal', color='black', fontfamily=cam_font)
plt.xticks(fontfamily=cam_font)
plt.yticks(fontfamily=cam_font)
plt.tight_layout()
save_figure(plt.gcf(), "Sensitivity_TOPSIS_boxplot")
plt.show()

num_sim = 1000; delta_mc = 0.05; sigma_expert_mc = 0.05
rng = np.random.default_rng(123)
scores_store = np.zeros((num_sim, len(alternatives)))
ranks = np.zeros((num_sim, len(alternatives)), dtype=int)

for s in range(num_sim):
    experts_s = make_experts_from_baseline(ndm, num_experts, sigma_expert_mc)
    out = run_pipeline_expertfusion_to_rank(experts_s, delta=delta_mc, weights=weights_equal)
    rc = out["rc"]; scores_store[s,:] = rc
    order = np.argsort(-rc); ranks[s, order] = np.arange(1, len(alternatives)+1)

df_scores_mc = pd.DataFrame(scores_store, columns=alternatives)
df_mc = pd.DataFrame(ranks, columns=alternatives)
summary = pd.DataFrame({"RC_mean": df_scores_mc.mean(), "RC_std": df_scores_mc.std(),
                        "Rank1_freq": (df_mc.eq(1).sum() / num_sim)})
print("\nMonte Carlo — TOPSIS RC summaries (full pipeline)"); display(summary)

# Binned RC frequency table for command prompt/report text.
bins = np.linspace(0, 1, 11)
labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]
df_binned_rc = pd.DataFrame({
    alt: pd.cut(df_scores_mc[alt], bins=bins, labels=labels, include_lowest=True)
          .value_counts().reindex(labels, fill_value=0)
    for alt in alternatives
})
print("\nBinned RC frequencies (N=1000)")
print(df_binned_rc.astype(int).to_string())

fig_binned, ax_binned = plt.subplots(figsize=(12, 7))
df_binned_rc.plot(kind='bar', width=0.85, ax=ax_binned,
                  color=["#440154", "#21918C", "#FDE725"], edgecolor="black")
for p in ax_binned.patches:
    if p.get_height() > 0:
        ax_binned.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width()/2, p.get_height()),
            xytext=(0,3), textcoords="offset points",
            ha='center', fontsize=9, fontweight='bold', fontfamily=cam_font
        )
ax_binned.set_xlabel("Relative Closeness (RC) Score Range", fontfamily=cam_font, fontweight='bold')
ax_binned.set_ylabel("Frequency (N=1000 trials)", fontfamily=cam_font, fontweight='bold')
ax_binned.legend(title="Alternatives", prop={'family': cam_font, 'size': 12})
plt.xticks(rotation=45, fontfamily=cam_font)
plt.yticks(fontfamily=cam_font)
sns.despine(); plt.tight_layout()
save_figure(fig_binned, "MC_Score_Grouped_Bar")
plt.show()

plt.figure(figsize=(12, 7))
for alt in alternatives:
    srt = np.sort(df_scores_mc[alt])
    cdf = np.arange(1, len(srt)+1)/len(srt)
    plt.plot(srt, cdf, label=alt, linewidth=1.5)
plt.xlabel("Relative Closeness (RC)", color="#000000", fontweight="bold", fontfamily=cam_font, fontsize=14)
plt.ylabel("Cumulative Probability", color="#000000", fontweight="bold", fontfamily=cam_font, fontsize=14)
plt.xticks(fontfamily=cam_font)
plt.yticks(fontfamily=cam_font)

leg = plt.legend(prop={'family': cam_font, 'size': 12})
for text in leg.get_texts():
    text.set_color("black")

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
save_figure(plt.gcf(), "MonteCarlo_RC_CDF")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=df_mc)
plt.ylabel("Rank (lower is better)", fontweight='normal', color='black', fontfamily=cam_font)
plt.xticks(fontfamily=cam_font)
plt.yticks(fontfamily=cam_font)
plt.tight_layout()
save_figure(plt.gcf(), "MonteCarlo_Rank_Distribution")
plt.show()

M_CRISP = "Crisp TOPSIS"; M_FUZZY = "Fuzzy TOPSIS"; M_IFS = "IFS–TOPSIS"
M_NEUTRO = "Neutrosophic TOPSIS (without DST)"; M_PROP = "Proposed Method"

def _minmax_norm_cols(X):
    X = np.asarray(X, float); mn = X.min(axis=0); mx = X.max(axis=0)
    rng = np.where(mx > mn, mx - mn, 1.0); out = (X - mn)/rng
    out[:, mx == mn] = 0.0; return out

def _classical_topsis_rc(X, weights, benefit_mask=None):
    X = np.asarray(X, float); w = np.asarray(weights, float); w = w/w.sum()
    if benefit_mask is None: benefit_mask = np.ones(X.shape[1], dtype=bool)
    V = X * w
    P_plus  = np.where(benefit_mask, V.max(axis=0), V.min(axis=0))
    P_minus = np.where(benefit_mask, V.min(axis=0), V.max(axis=0))
    D_plus  = np.sqrt(((V - P_plus )**2).sum(axis=1))
    D_minus = np.sqrt(((V - P_minus)**2).sum(axis=1))
    return D_minus / (D_minus + D_plus + 1e-12)

def crisp_topsis_from_svns(ndm_mat, weights):
    m, n = len(alternatives), len(criteria)
    S = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            T,I,F = ndm_mat[i,j]; S[i,j] = T - I - F
    X = _minmax_norm_cols(S)
    return _classical_topsis_rc(X, weights, benefit_mask=np.ones(n, dtype=bool))

def fuzzy_topsis_from_svns(ndm_mat, weights, alpha=1.0, beta=1.0):
    m, n = len(alternatives), len(criteria)
    MU = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            T,I,F = ndm_mat[i,j]
            mu_low  = max(0.0, T - alpha*I - beta*F)
            mu_high = min(1.0, T + alpha*I + beta*F)
            MU[i,j] = 0.5*(mu_low + mu_high)
    X = _minmax_norm_cols(MU)
    return _classical_topsis_rc(X, weights, benefit_mask=np.ones(n, dtype=bool))

def ifs_topsis(ndm_mat, weights):
    m, n = len(alternatives), len(criteria)
    MU = np.zeros((m, n)); NU = np.zeros((m, n)); PI = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            T,I,F = ndm_mat[i,j]; MU[i,j], NU[i,j], PI[i,j] = T, F, I
    MU_n = _minmax_norm_cols(MU); NU_n = _minmax_norm_cols(NU); PI_n = _minmax_norm_cols(PI)
    MU_plus, MU_minus = MU_n.max(axis=0), MU_n.min(axis=0)
    NU_plus, NU_minus = NU_n.min(axis=0), NU_n.max(axis=0)
    PI_plus, PI_minus = PI_n.min(axis=0), PI_n.max(axis=0)
    w = np.asarray(weights)/np.sum(weights)
    mvals = len(alternatives); Dp = np.zeros(mvals); Dm = np.zeros(mvals)
    for i in range(mvals):
        d2p = 0.0; d2m = 0.0
        for j in range(n):
            d2p += w[j]*((MU_n[i,j]-MU_plus[j])**2 + (NU_n[i,j]-NU_plus[j])**2 + (PI_n[i,j]-PI_plus[j])**2)
            d2m += w[j]*((MU_n[i,j]-MU_minus[j])**2 + (NU_n[i,j]-NU_minus[j])**2 + (PI_n[i,j]-PI_minus[j])**2)
        Dp[i] = np.sqrt(d2p); Dm[i] = np.sqrt(d2m)
    return Dm/(Dm+Dp+1e-12)

def proposed_from_fused(ndm_mat, weights, delta=0.05):
    m, n = len(alternatives), len(criteria)
    ivns = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            ivns[i,j] = inflate_to_ivns(ndm_mat[i,j], delta)
    ivns_norm = normalize_ivns_matrix(ivns)
    return topsis_rc_from_ivns_benefit(ivns_norm, weights)


def neutrosophic_topsis_no_dst(ndm_mat, weights, delta=0.05):

    m, n = len(alternatives), len(criteria)
    ivns = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            ivns[i,j] = inflate_to_ivns(ndm_mat[i,j], delta)
    ivns_norm = normalize_ivns_matrix(ivns)
    return topsis_rc_from_ivns_benefit(ivns_norm, weights)

print("\n=== Comparison A: Single NDM (no fusion) ===")
rc_crisp_A = crisp_topsis_from_svns(ndm, weights_equal)
rc_fuzzy_A = fuzzy_topsis_from_svns(ndm, weights_equal)
rc_ifs_A   = ifs_topsis(ndm, weights_equal)
rc_neutro_A = neutrosophic_topsis_no_dst(ndm, weights_equal, delta=0.05)
rc_prop_A  = proposed_from_fused(ndm, weights_equal, delta=0.05)
compA = pd.DataFrame({M_CRISP: rc_crisp_A, M_FUZZY: rc_fuzzy_A, M_IFS: rc_ifs_A, M_NEUTRO: rc_neutro_A, M_PROP: rc_prop_A},
                     index=alternatives)
ranksA = compA.rank(ascending=False, method="min").astype(int)
print(compA.round(4)); print("\nRanks (1=best):"); print(ranksA)

print("\n=== Comparison B: Multi-expert (baselines: mean; proposed: DST) ===")
def mean_aggregate_experts(experts_list):
    m, n = experts_list[0].shape
    avg = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            Ts = [E[i,j][0] for E in experts_list]
            Is = [E[i,j][1] for E in experts_list]
            Fs = [E[i,j][2] for E in experts_list]
            avg[i,j] = (float(np.mean(Ts)), float(np.mean(Is)), float(np.mean(Fs)))
    return avg

avg_ndm = mean_aggregate_experts(experts)
rc_crisp_B = crisp_topsis_from_svns(avg_ndm, weights_equal)
rc_fuzzy_B = fuzzy_topsis_from_svns(avg_ndm, weights_equal)
rc_ifs_B   = ifs_topsis(avg_ndm, weights_equal)
rc_neutro_B = neutrosophic_topsis_no_dst(avg_ndm, weights_equal, delta=0.05)
res_B = run_pipeline_expertfusion_to_rank(experts, delta=0.05, weights=weights_equal)
rc_prop_B = res_B["rc"]
compB = pd.DataFrame({M_CRISP: rc_crisp_B, M_FUZZY: rc_fuzzy_B, M_IFS: rc_ifs_B, M_NEUTRO: rc_neutro_B, M_PROP: rc_prop_B},
                     index=alternatives)
ranksB = compB.rank(ascending=False, method="min").astype(int)
print(compB.round(4)); print("\nRanks (1=best):"); print(ranksB)


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

METHOD_ORDER = [M_CRISP, M_FUZZY, M_IFS, M_NEUTRO, M_PROP]

PALETTE = {
    M_CRISP: "#0072B2",
    M_FUZZY: "#E69F00",
    M_IFS:   "#009E73",
    M_NEUTRO: "#CC79A7",
    M_PROP:  "#D55E00",
}
HATCHES = {
    M_CRISP: "",
    M_FUZZY: "//",
    M_IFS:   "..",
    M_NEUTRO: "\\\\",
    M_PROP:  "xx",
}

def _pub_style():
    sns.set_theme(context="talk", style="whitegrid", rc={"font.family": cam_font})
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "font.family": cam_font,
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })


def _annotate_bars(ax, bars):
    by_x = {}
    for b in bars:
        x = b.get_x() + b.get_width()/2
        by_x.setdefault(round(x, 6), []).append(b)

    for x, group in by_x.items():
        tops = [g.get_height() for g in group]
        for g in group:
            h = g.get_height()
            ax.annotate(f"{h:.2f}",
                        (g.get_x()+g.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, color="black", fontfamily=cam_font)
        k = int(np.argmax(tops))
        g_star = group[k]
        ax.annotate("",
                    (g_star.get_x()+g_star.get_width()/2, g_star.get_height()),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=14, color="black", fontfamily=cam_font)

def plot_rc_bars(comp_df, scenario_label, fname, highlight=M_PROP):
    _pub_style()
    df = comp_df.loc[ALT_ORDER, METHOD_ORDER]
    A, M = df.shape
    x = np.arange(A)
    width = 0.14
    gap = 0.03

    fig, ax = plt.subplots(figsize=(10, 5))
    all_bars = []
    for k, method in enumerate(METHOD_ORDER):
        offs = (k - (M - 1) / 2.0) * (width + gap)
        vals = df[method].to_numpy()
        bars = ax.bar(
            x + offs, vals, width,
            label=method,
            color=PALETTE[method],
            edgecolor="black",
            linewidth=0.9 if method != highlight else 1.4,
            hatch=HATCHES[method],
            zorder=3,
            alpha=0.95 if method != highlight else 1.0
        )
        all_bars.extend(bars)

        ax.plot(x + offs, vals, "o", ms=2.5, color="black", zorder=4)

    _annotate_bars(ax, all_bars)

    ax.set_title("", fontfamily=cam_font, fontweight="bold", pad=45)
    ax.set_ylabel("Relative Closeness (RC)", fontweight='bold', color='black', fontsize=14, fontfamily=cam_font)

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(ALT_ORDER, fontsize=14, fontweight='bold', color='black', fontfamily=cam_font)

    for tick in ax.get_yticklabels():
        tick.set_fontfamily(cam_font)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    leg = ax.legend(
        title="",
        ncols=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
        prop={'family': cam_font, 'size': 12}
    )
    for text in leg.get_texts():
        text.set_color("black")
    leg.get_title().set_color("black")


    plt.tight_layout()
    save_figure(fig, fname)
    plt.show()

def plot_rank_heatmap(ranks_df, scenario_label, fname):
    _pub_style()
    R = ranks_df.loc[ALT_ORDER, METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    vmax = int(np.nanmax(R.to_numpy()))
    cmap = sns.color_palette("light:navy", as_cmap=True)
    im = sns.heatmap(
        R, ax=ax,
        annot=True, fmt=".0f",
        cmap="YlGnBu_r",
        vmin=1, vmax=vmax,
        cbar_kws={"shrink": 0.8, "label": "Rank (1 = best)"},
        linewidths=0.5, linecolor="#FFFFFF",
        annot_kws={"fontfamily": cam_font}
    )
    ax.set_title(f"Ranks by Method — {scenario_label}", fontfamily=cam_font, fontweight="bold", pad=15)
    ax.set_xlabel("Method", fontfamily=cam_font)
    ax.set_ylabel("Alternative", fontweight='normal', color='black', fontfamily=cam_font)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontfamily=cam_font)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontfamily=cam_font)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_fontfamily(cam_font)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontfamily(cam_font)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    save_figure(fig, fname)
    plt.show()

def plot_rc_panels(compA, compB, fname, highlight=M_PROP):
    _pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, comp_df in zip(
        axes,
        [compA.loc[ALT_ORDER, METHOD_ORDER], compB.loc[ALT_ORDER, METHOD_ORDER]]
    ):
        A, M = comp_df.shape
        x = np.arange(A)
        width = 0.14
        gap = 0.03
        all_bars = []
        for k, method in enumerate(METHOD_ORDER):
            offs = (k - (M - 1) / 2.0) * (width + gap)
            vals = comp_df[method].to_numpy()
            bars = ax.bar(
                x + offs, vals, width,
                label=method,
                color=PALETTE[method],
                edgecolor="black",
                linewidth=0.9 if method != highlight else 1.4,
                hatch=HATCHES[method],
                zorder=3,
                alpha=0.95 if method != highlight else 1.0
            )
            all_bars.extend(bars)
            ax.plot(x + offs, vals, "o", ms=2.5, color="black", zorder=4)
        _annotate_bars(ax, all_bars)

        ax.set_title("", fontfamily=cam_font, fontweight="bold", pad=40)
        ax.set_xlabel("Alternative", fontfamily=cam_font)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(ALT_ORDER, fontfamily=cam_font, fontweight="bold")
        for tick in ax.get_yticklabels():
            tick.set_fontfamily(cam_font)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("Relative Closeness (RC)", fontweight='normal', color='black', fontfamily=cam_font)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Method",
        ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.15),
        frameon=False, handlelength=1.4, columnspacing=1.0, handletextpad=0.4,
        prop={'family': cam_font, 'size': 12}
    )
    plt.tight_layout()
    save_figure(fig, fname)
    plt.show()

plot_rc_bars(compA, "Single NDM (no fusion)",        fname="Fig1A_RC_q1")
plot_rc_bars(compB, "Multi-expert (mean vs DST)",    fname="Fig1B_RC_q1")
plot_rank_heatmap(ranksA, "Single NDM (no fusion)",  fname="Fig2A_Ranks_q1")
plot_rank_heatmap(ranksB, "Multi-expert (mean vs DST)", fname="Fig2B_Ranks_q1")

plot_rc_panels(compA, compB, fname="Fig_RC_Panels_q1")

delta_report = 0.05
ivfs_from_ndm = np.empty_like(ndm, dtype=object)
for i in range(len(alternatives)):
    for j in range(len(criteria)):
        ivfs_from_ndm[i, j] = inflate_to_ivns(ndm[i, j], delta_report)
df_ivfs = pd.DataFrame(ivfs_from_ndm.tolist(), columns=criteria, index=alternatives)

ivns_after_dst = res["ivns"]
dst_T_only = np.empty((len(alternatives), len(criteria)), dtype=object)
for i in range(len(alternatives)):
    for j in range(len(criteria)):
        dst_T_only[i, j] = ivns_after_dst[i, j][0]
df_dst = pd.DataFrame(dst_T_only.tolist(), columns=criteria, index=alternatives)

ivns_norm = res["ivns_norm"]
norm_T_only = np.empty((len(alternatives), len(criteria)), dtype=object)
for i in range(len(alternatives)):
    for j in range(len(criteria)):
        norm_T_only[i, j] = ivns_norm[i, j][0]
df_norm = pd.DataFrame(norm_T_only.tolist(), columns=criteria, index=alternatives)

df_iwns = pd.DataFrame({"IWNS": res["iwns"]}, index=alternatives)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

print("df_ndm:")
display(df_ndm)

print("\ndf_ivfs:")
display(df_ivfs)

print("\ndf_dst: Representing Truth values Lower and upper bounds after DST aggretaion for each alternatives")
display(df_dst)

print("\ndf_norm:Representing Normalized Truth values Lower and upper bounds after Normalization of DTS aggreated values")
display(df_norm)

print("\ndf_iwns:")
display(df_iwns)

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

rank_levels = [1,2,3]
counts = np.zeros((len(alternatives), len(rank_levels)), dtype=int)
for a, alt in enumerate(alternatives):
    vc = df_mc[alt].value_counts().reindex(rank_levels, fill_value=0).sort_index()
    counts[a, :] = vc.to_numpy()

fig = plt.figure(figsize=(10.2, 6.8))
ax  = fig.add_subplot(111, projection="3d")

dx = 0.6; dy = 0.6
xs = []; ys = []; zs = []; dzs = []; cols = []

rank_colors = {1:"#0072B2", 2:"#E69F00", 3:"#009E73"}
for a in range(len(alternatives)):
    for r_idx, r in enumerate(rank_levels):
        xs.append(a - dx/2)
        ys.append(r - dy/2)
        zs.append(0)
        dzs.append(counts[a, r_idx])
        cols.append(rank_colors[r])

ax.bar3d(xs, ys, zs, dx, dy, dzs, color=cols, edgecolor="#000000", linewidth=0.6, alpha=0.95)

ax.set_title("Monte Carlo Rank Frequencies (N = 1000)", fontsize=16, fontweight="bold", fontfamily=cam_font, pad=20)
ax.set_xlabel("Alternative", fontsize=13, fontfamily=cam_font)
ax.set_ylabel("Rank", fontsize=13, fontfamily=cam_font)
ax.set_zlabel("Frequency", fontsize=13, fontfamily=cam_font)

ax.set_xticks(range(len(alternatives)))
ax.set_xticklabels(alternatives, fontsize=12, fontfamily=cam_font)
ax.set_yticks(rank_levels)
ax.set_yticklabels(rank_levels, fontfamily=cam_font)
for t in ax.zaxis.get_major_ticks():
    t.label1.set_fontfamily(cam_font)

import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=rank_colors[r], label=f"Rank {r}") for r in rank_levels]
leg = ax.legend(handles=handles, title="Ranks", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, prop={'family': cam_font, 'size': 12})
leg.get_frame().set_edgecolor("#000000"); leg.get_frame().set_linewidth(0.8)
leg.get_title().set_fontfamily(cam_font)

plt.tight_layout()
save_figure(fig, "Fig_3D_Rank_Frequencies")
plt.show()

write_text_outputs()
