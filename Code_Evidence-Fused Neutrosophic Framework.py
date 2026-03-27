import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Cambria", "Times New Roman"],
        "font.size": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    }
)


def save_figure(fig, name):
    for fmt in ["png", "pdf"]:
        folder = f"figures/{fmt}"
        os.makedirs(folder, exist_ok=True)
        fig.savefig(os.path.join(folder, f"{name}.{fmt}"), bbox_inches="tight", dpi=300)


alternatives = ["Treatment A", "Treatment B", "Treatment C"]
criteria = ["Effectiveness", "Side Effects", "Affordability", "Recovery Time", "Patient Satisfaction"]
weights_equal = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

ndm = np.array(
    [
        [(0.8, 0.1, 0.1), (0.6, 0.2, 0.2), (0.5, 0.3, 0.2), (0.7, 0.2, 0.1), (0.9, 0.05, 0.05)],
        [(0.9, 0.05, 0.05), (0.8, 0.1, 0.1), (0.6, 0.2, 0.2), (0.85, 0.1, 0.05), (0.85, 0.1, 0.05)],
        [(0.95, 0.02, 0.03), (0.9, 0.05, 0.05), (0.7, 0.15, 0.15), (0.9, 0.05, 0.05), (0.95, 0.02, 0.03)],
    ],
    dtype=object,
)


def normalize_bpa_from_triplet(triplet):
    t, i, f = triplet
    s = t + i + f
    if s <= 0:
        return {"G": 0.0, "B": 0.0, "Θ": 1.0}
    return {"G": t / s, "B": f / s, "Θ": i / s}


def dst_combine_two(m1, m2):
    inter = {
        ("G", "G"): "G",
        ("G", "Θ"): "G",
        ("Θ", "G"): "G",
        ("B", "B"): "B",
        ("B", "Θ"): "B",
        ("Θ", "B"): "B",
        ("Θ", "Θ"): "Θ",
    }
    k = m1["G"] * m2["B"] + m1["B"] * m2["G"]
    m = {"G": 0.0, "B": 0.0, "Θ": 0.0}
    for (a, b), s in inter.items():
        m[s] += m1[a] * m2[b]
    if k >= 1.0 - 1e-12:
        return m1
    scale = 1.0 / (1.0 - k)
    for key in m:
        m[key] *= scale
    return m


def dst_fuse_triplets(expert_triplets):
    bp = normalize_bpa_from_triplet(expert_triplets[0])
    for triplet in expert_triplets[1:]:
        bp = dst_combine_two(bp, normalize_bpa_from_triplet(triplet))
    return bp["G"], bp["Θ"], bp["B"]


def inflate_to_ivns(triplet, delta):
    t, i, f = triplet

    def clip_pair(x):
        return max(0.0, x - delta), min(1.0, x + delta)

    return clip_pair(t), clip_pair(i), clip_pair(f)


def normalize_ivns_matrix(ivns_mat):
    n_alt, n_crit = ivns_mat.shape
    out = np.empty_like(ivns_mat, dtype=object)
    for j in range(n_crit):
        t_h = np.array([ivns_mat[i, j][0][1] for i in range(n_alt)])
        t_l = np.array([ivns_mat[i, j][0][0] for i in range(n_alt)])
        i_h = np.array([ivns_mat[i, j][1][1] for i in range(n_alt)])
        i_l = np.array([ivns_mat[i, j][1][0] for i in range(n_alt)])
        f_h = np.array([ivns_mat[i, j][2][1] for i in range(n_alt)])
        f_l = np.array([ivns_mat[i, j][2][0] for i in range(n_alt)])
        t_min, t_max = t_l.min(), t_h.max()
        i_min, i_max = i_l.min(), i_h.max()
        f_min, f_max = f_l.min(), f_h.max()
        t_rng = (t_max - t_min) if t_max > t_min else 1.0
        i_rng = (i_max - i_min) if i_max > i_min else 1.0
        f_rng = (f_max - f_min) if f_max > f_min else 1.0
        for i in range(n_alt):
            (tlo, thi), (ilo, ihi), (flo, fhi) = ivns_mat[i, j]
            out[i, j] = (
                ((tlo - t_min) / t_rng, (thi - t_min) / t_rng),
                ((i_max - ilo) / i_rng, (i_max - ihi) / i_rng),
                ((f_max - flo) / f_rng, (f_max - fhi) / f_rng),
            )
    return out


def topsis_rc_from_ivns_benefit(ivns_norm, weights):
    n_alt, n_crit = ivns_norm.shape
    w = np.asarray(weights, float) / np.sum(weights)
    mid = np.zeros((n_alt, n_crit, 3))
    for i in range(n_alt):
        for j in range(n_crit):
            (tlo, thi), (ilo, ihi), (flo, fhi) = ivns_norm[i, j]
            mid[i, j] = [0.5 * (tlo + thi), 0.5 * (ilo + ihi), 0.5 * (flo + fhi)]
    t_plus, t_minus = mid[:, :, 0].max(axis=0), mid[:, :, 0].min(axis=0)
    i_plus, i_minus = mid[:, :, 1].max(axis=0), mid[:, :, 1].min(axis=0)
    f_plus, f_minus = mid[:, :, 2].max(axis=0), mid[:, :, 2].min(axis=0)
    d_plus = np.zeros(n_alt)
    d_minus = np.zeros(n_alt)
    for i in range(n_alt):
        p = np.sum(w * ((mid[i, :, 0] - t_plus) ** 2 + (mid[i, :, 1] - i_plus) ** 2 + (mid[i, :, 2] - f_plus) ** 2))
        m = np.sum(w * ((mid[i, :, 0] - t_minus) ** 2 + (mid[i, :, 1] - i_minus) ** 2 + (mid[i, :, 2] - f_minus) ** 2))
        d_plus[i] = np.sqrt(p)
        d_minus[i] = np.sqrt(m)
    return d_minus / (d_minus + d_plus + 1e-12)


def _minmax_norm_cols(x):
    mn, mx = x.min(axis=0), x.max(axis=0)
    rng = np.where(mx > mn, mx - mn, 1.0)
    return (x - mn) / rng


def _classical_topsis_rc(x, weights):
    w = np.asarray(weights, float) / np.sum(weights)
    v = x * w
    p_plus, p_minus = v.max(axis=0), v.min(axis=0)
    d_plus = np.sqrt(((v - p_plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((v - p_minus) ** 2).sum(axis=1))
    return d_minus / (d_minus + d_plus + 1e-12)


def crisp_topsis_from_svns(ndm_mat, weights):
    s = np.array([[tri[0] - tri[1] - tri[2] for tri in row] for row in ndm_mat])
    return _classical_topsis_rc(_minmax_norm_cols(s), weights)


def fuzzy_topsis_from_svns(ndm_mat, weights):
    mu = np.array([[0.5 * (max(0, t - i - f) + min(1, t + i + f)) for (t, i, f) in row] for row in ndm_mat])
    return _classical_topsis_rc(_minmax_norm_cols(mu), weights)


def ifs_topsis(ndm_mat, weights):
    m = len(ndm_mat)
    n = len(ndm_mat[0]) if m > 0 else 0
    mu, nu, pi = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            mu[i, j], nu[i, j], pi[i, j] = ndm_mat[i, j]
    mu_n, nu_n, pi_n = _minmax_norm_cols(mu), _minmax_norm_cols(nu), _minmax_norm_cols(pi)
    mu_p, mu_m = mu_n.max(axis=0), mu_n.min(axis=0)
    nu_p, nu_m = nu_n.min(axis=0), nu_n.max(axis=0)
    pi_p, pi_m = pi_n.min(axis=0), pi_n.max(axis=0)
    w = weights / weights.sum()
    d_plus, d_minus = np.zeros(m), np.zeros(m)
    for i in range(m):
        d_plus[i] = np.sqrt(np.sum(w * ((mu_n[i] - mu_p) ** 2 + (nu_n[i] - nu_p) ** 2 + (pi_n[i] - pi_p) ** 2)))
        d_minus[i] = np.sqrt(np.sum(w * ((mu_n[i] - mu_m) ** 2 + (nu_n[i] - nu_m) ** 2 + (pi_n[i] - pi_m) ** 2)))
    return d_minus / (d_minus + d_plus + 1e-12)


def neutrosophic_topsis_no_dst(ndm_mat, weights, delta=0.05):
    m = len(ndm_mat)
    n = len(ndm_mat[0]) if m > 0 else 0
    ivns = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            ivns[i, j] = inflate_to_ivns(ndm_mat[i, j], delta)
    return topsis_rc_from_ivns_benefit(normalize_ivns_matrix(ivns), weights)


def mean_aggregate_experts(experts):
    m = len(experts[0])
    n = len(experts[0][0]) if m > 0 else 0
    avg = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            ts = [e[i, j][0] for e in experts]
            ins = [e[i, j][1] for e in experts]
            fs = [e[i, j][2] for e in experts]
            avg[i, j] = float(np.mean(ts)), float(np.mean(ins)), float(np.mean(fs))
    return avg


def run_pipeline_expertfusion_to_rank(experts, delta=0.05, weights=None):
    m = len(experts[0])
    n = len(experts[0][0]) if m > 0 else 0
    if weights is None:
        weights = np.ones(n) / n
    fused = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            fused[i, j] = dst_fuse_triplets([e[i, j] for e in experts])
    ivns = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            ivns[i, j] = inflate_to_ivns(fused[i, j], delta)
    rc = topsis_rc_from_ivns_benefit(normalize_ivns_matrix(ivns), weights)
    return {"rc": rc}


def make_experts_from_baseline(base_ndm, p=3, sigma=0.05):
    rng = np.random.default_rng(123)
    experts = []
    for _ in range(p):
        m = len(base_ndm)
        n = len(base_ndm[0]) if m > 0 else 0
        arr = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                t, ind, f = base_ndm[i, j]
                arr[i, j] = (
                    float(np.clip(t + rng.normal(0, sigma), 0, 1)),
                    float(np.clip(ind + rng.normal(0, sigma), 0, 1)),
                    float(np.clip(f + rng.normal(0, sigma), 0, 1)),
                )
        experts.append(arr)
    return experts


def main():
    print("Comparison A: Single Expert")
    df_a = pd.DataFrame(
        {
            "Crisp": crisp_topsis_from_svns(ndm, weights_equal),
            "Fuzzy": fuzzy_topsis_from_svns(ndm, weights_equal),
            "IFS": ifs_topsis(ndm, weights_equal),
            "Neutro (No DST)": neutrosophic_topsis_no_dst(ndm, weights_equal),
            "Proposed": run_pipeline_expertfusion_to_rank([ndm], 0.05, weights_equal)["rc"],
        },
        index=alternatives,
    )
    print(df_a.round(4))

    print("\nComparison B: Multi-Expert (N=3)")
    experts = make_experts_from_baseline(ndm, 3, 0.05)
    avg_ndm = mean_aggregate_experts(experts)
    df_b = pd.DataFrame(
        {
            "Crisp": crisp_topsis_from_svns(avg_ndm, weights_equal),
            "Fuzzy": fuzzy_topsis_from_svns(avg_ndm, weights_equal),
            "IFS": ifs_topsis(avg_ndm, weights_equal),
            "Neutro (No DST)": neutrosophic_topsis_no_dst(avg_ndm, weights_equal),
            "Proposed": run_pipeline_expertfusion_to_rank(experts, 0.05, weights_equal)["rc"],
        },
        index=alternatives,
    )
    print(df_b.round(4))

    print("\nRunning Monte Carlo Simulation (N=1000)...")
    num_sim = 1000
    results = np.zeros((num_sim, 3))
    for s in range(num_sim):
        mc_exp = make_experts_from_baseline(ndm, 3, 0.05)
        results[s] = run_pipeline_expertfusion_to_rank(mc_exp, 0.05, weights_equal)["rc"]

    bins = np.linspace(0, 1, 11)
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(10)]
    binned_df = pd.DataFrame(
        {alt: pd.cut(results[:, i], bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels) for i, alt in enumerate(alternatives)}
    )
    print("\nBinned RC frequencies (N=1000)")
    print(binned_df.fillna(0).astype(int))

    fig, ax = plt.subplots(figsize=(12, 7))
    binned_df.plot(kind="bar", width=0.85, ax=ax, color=["#440154", "#21918C", "#FDE725"], edgecolor="black")
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                fontfamily="serif",
            )
    ax.set_xlabel("Relative Closeness (RC) Score Range", fontfamily="serif", fontweight="bold")
    ax.set_ylabel("Frequency (N=1000 trials)", fontfamily="serif", fontweight="bold")
    ax.legend(title="Alternatives", prop={"family": "serif"})
    plt.xticks(rotation=45, fontfamily="serif")
    plt.yticks(fontfamily="serif")
    sns.despine()
    plt.tight_layout()
    save_figure(fig, "MC_Score_Grouped_Bar")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
