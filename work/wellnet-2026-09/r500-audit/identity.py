"""
THE IDENTITY.

Job 2C tuned two supposedly different truths -- "the excess is organised by
r/R500" and "the excess is organised by physical r" -- and got bit-identical
output.  That is not a coding accident.

    log10(r / R500_i) = log10(r) - log10(R500_i)

and log10(R500_i) is CONSTANT WITHIN CLUSTER i, so it lies in the span of the
cluster indicator variables.  Any model that already allows each cluster its own
level therefore cannot tell the two apart: the design matrices
[cluster indicators | log r] and [cluster indicators | log(r/R500)] span the same
column space, exactly.

This is the `variable-lists-collapse` pattern -- take the rank before building
the search.  Here the rank is taken on the real X-COP design.

The only information R500 carries is therefore the BETWEEN-cluster offsets, and
this module measures how much of the residual variance those offsets could
possibly explain.
"""
from __future__ import annotations
import json

import numpy as np

import ingest as I
import nullsim as N

KPC = I.KPC


def main():
    cl = I.load_all(verbose=False)
    pts = I.xcop_points(cl)
    r = np.array([p["r"] for p in pts])
    gb = np.array([p["gb"] for p in pts])
    go = np.array([p["go"] for p in pts])
    R5 = np.array([p["R500_hse"] for p in pts])
    nm = np.array([p["name"] for p in pts])
    y = I.rar_residual(gb, go)
    names = sorted(set(nm))
    Dm = np.column_stack([(nm == c).astype(float) for c in names])  # indicators
    lr = np.log10(r / KPC)
    lt = np.log10(r / R5)

    out = {}
    A = np.column_stack([Dm, lr])
    B = np.column_stack([Dm, lt])
    C = np.column_stack([Dm, lr, lt])
    ra, rb, rc = (np.linalg.matrix_rank(M, tol=1e-9) for M in (A, B, C))
    out["rank_indicators_plus_logr"] = int(ra)
    out["rank_indicators_plus_log_r_over_R500"] = int(rb)
    out["rank_indicators_plus_BOTH"] = int(rc)
    out["columns_A"] = int(A.shape[1])
    out["columns_C"] = int(C.shape[1])
    print(f"rank[ indicators | log r ]            = {ra}  of {A.shape[1]} columns")
    print(f"rank[ indicators | log(r/R500) ]      = {rb}  of {B.shape[1]} columns")
    print(f"rank[ indicators | log r | log(r/R500)] = {rc}  of {C.shape[1]} columns")
    print(f"-> adding log(r/R500) to a model that already has log r and per-cluster")
    print(f"   levels adds {rc - ra} independent direction(s).")

    # residual of log(r/R500) after projecting out [indicators, log r]
    coef, *_ = np.linalg.lstsq(A, lt, rcond=None)
    resid = lt - A @ coef
    out["max_abs_residual_of_log_r_over_R500_on_A"] = float(np.max(np.abs(resid)))
    out["rms_residual"] = float(np.std(resid))
    print(f"   residual of log(r/R500) on that span: max |e| = "
          f"{np.max(np.abs(resid)):.3e}, rms = {np.std(resid):.3e}")

    # within-cluster demeaned versions are identical
    def demean(v):
        w = v.copy()
        for c in names:
            m = nm == c
            w[m] = w[m] - w[m].mean()
        return w
    d = np.max(np.abs(demean(lr) - demean(lt)))
    out["max_abs_diff_of_within_cluster_demeaned"] = float(d)
    print(f"   within-cluster demeaned log r vs log(r/R500): max diff = {d:.3e}")

    # ------------------------------------------------ variance decomposition
    tot = float(np.var(y, ddof=0))
    mu = np.array([y[nm == c].mean() for c in names])
    nn = np.array([(nm == c).sum() for c in names], float)
    between = float(np.sum(nn * (mu - y.mean()) ** 2) / len(y))
    within = tot - between
    out["variance"] = dict(total=tot, between_cluster=between, within_cluster=within,
                           between_fraction=between / tot)
    print(f"\nvariance of the RAR residual: total {tot:.5f}, "
          f"between-cluster {between:.5f} ({100*between/tot:.1f}%), "
          f"within-cluster {within:.5f} ({100*within/tot:.1f}%)")
    print("   R500 can only ever act on the between-cluster part.")

    # how much of the BETWEEN part can R500 explain?  regress the per-cluster mean
    # residual on ln R500
    lR = np.log10(np.array([R5[nm == c][0] for c in names]))
    rr = N.pear(lR, mu)
    out["between_cluster_regression"] = dict(
        n_clusters=len(names), pearson_meanY_vs_log10R500=rr,
        r_squared=rr ** 2,
        spearman=N.spear(lR, mu),
        slope_dex_per_dex=float(np.polyfit(lR, mu, 1)[0]),
        share_of_total_variance_explainable=float(rr ** 2 * between / tot))
    print(f"   corr(per-cluster mean residual, log10 R500) = {rr:+.4f} over "
          f"{len(names)} clusters (r^2 = {rr**2:.3f})")
    print(f"   -> R500 could at most explain {100*rr**2*between/tot:.2f}% of the "
          f"total residual variance")

    # ------------------------------------- leverage of the normalisation
    sdlnR = float(np.std(np.log(np.array([R5[nm == c][0] for c in names])), ddof=1))
    sdlnr = float(np.mean([np.std(np.log(r[nm == c]), ddof=1) for c in names]))
    out["leverage"] = dict(sd_ln_R500=sdlnR, sd_ln_r_within=sdlnr,
                           ratio=sdlnR / sdlnr,
                           R500_min_kpc=float(min(R5) / KPC),
                           R500_max_kpc=float(max(R5) / KPC),
                           R500_max_over_min=float(max(R5) / min(R5)))
    print(f"\nleverage: R500 spans {min(R5)/KPC:.0f}-{max(R5)/KPC:.0f} kpc "
          f"(a factor {max(R5)/min(R5):.2f}); sd(ln R500) = {sdlnR:.4f} against "
          f"sd(ln r) = {sdlnr:.4f} inside one cluster, ratio {sdlnR/sdlnr:.3f}")

    # ------------------------------------- what a fixed-effects model actually says
    # y ~ per-cluster level + slope*log r   vs   + slope*log(r/R500):  identical.
    # The only non-degenerate question is whether the per-cluster LEVELS are
    # predicted by R500, which is the between-cluster regression above.
    for lab, X in (("indicators + log r", A), ("indicators + log(r/R500)", B)):
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(np.sum((y - X @ c) ** 2))
        out.setdefault("fixed_effects", {})[lab] = dict(
            rss=rss, slope=float(c[-1]),
            r_squared=1 - rss / float(np.sum((y - y.mean()) ** 2)))
        print(f"   fixed-effects fit, {lab:<26} RSS = {rss:.6f}, "
              f"slope = {c[-1]:+.4f}")

    json.dump(out, open("identity_results.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote identity_results.json")


if __name__ == "__main__":
    main()
