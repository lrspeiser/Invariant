"""discriminator.py -- a flexible, invariant, CALIBRATED discriminator.

Its job (protocol item 3) is to approximate the best available test and to
LOCATE information, not to be the result.  It is a gradient-boosted
classifier on the invariant per-object features of invariants.py, trained at
the OBJECT level (a galaxy or a cluster is one row; the universe is the
label), and read out at the CORPUS level as the sum of per-object log-odds --
the corpus log-likelihood ratio a naive combination of independent objects
would give.

Sizing.  Discriminants are fitted ONLY on calibration sets and scored ONLY on
disjoint audit sets.  The null distribution of the corpus-level AUC is
measured by A-vs-A: the same pipeline run with the two 'classes' being two
random halves of ONE universe, repeated over splits and universes.  The z of
a real pair is (AUC - 0.5) / sd_null, capped at Z_CAP as BF's stats were,
and a permutation p-value on the audit scores is reported beside it.

Truth columns (``T_``), set and scene identifiers never enter a fit.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

import invariants as IV

Z_CAP = 8.5
GBDT = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
            min_samples_leaf=40, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=0)


# ======================================================================
# feature assembly
# ======================================================================
def feature_table(stack, tag, grp, drop=(), keep=None, randomise_alignment=None,
                  match=None):
    """Feature matrix for one arm and object class.

    drop / keep      channel names (invariants.galaxy_channel / cluster_channel)
    randomise_alignment   a Generator -> the observed axes are replaced by random
                     ones before the phase features are formed
    match            {feature: value} subtracted from 'level' features (the
                     mean-profile-matched ablation)
    Returns X (n, p), names, sets, scenes.
    """
    d = stack[tag][grp]
    base = {k: v for k, v in d.items() if not k.startswith("T_") and k not in ("set", "scene")}
    if grp == "clu":
        ph = IV.cluster_phase_features(base, rng=randomise_alignment)
        chan = IV.cluster_channel
    else:
        ph = IV.galaxy_phase_features(base, rng=randomise_alignment)
        chan = IV.galaxy_channel
    feats = dict(base)
    feats.update(ph)
    # raw axes and raw quadrupole vectors never enter a fit: only their
    # projections relative to observed axes do
    for k in ("pa_bar", "ax_ext", "psi_obs", "m3_re", "m3_im", "m3_c11", "m3_c12", "m3_c22"):
        feats.pop(k, None)
    for k in [k for k in feats if k.startswith(("z_re_", "z_im_", "z_c"))]:
        feats.pop(k)
    names = [k for k in feats if (keep is None or chan(k) in keep) and chan(k) not in drop]
    X = np.stack([np.asarray(feats[k], float) for k in names], 1) if names else np.zeros((len(d["set"]), 0))
    if match:
        for j, k in enumerate(names):
            if k in match:
                X[:, j] = X[:, j] - match[k]
    return X, names, np.asarray(d["set"]), np.asarray(d["scene"])


LEVEL_PREFIXES = ("res_", "ly_", "lv_", "vz_", "vr_", "rz_", "rr_", "dz_", "wres_", "wl_", "mono_",
                  "wpred_", "tres_", "yres_", "hres_", "t_lA", "y_lA", "h_lA", "t_slope", "y_slope",
                  "h_slope", "dres_", "lsig_", "d_lA", "d_slope", "thE", "sl_", "kb_in", "ep_",
                  "q2_", "n_img", "ldelay")


def level_means(stack, tags, grp):
    """Per-feature means over the calibration rows of the given arms, for the
    mean-profile-matched ablation."""
    out = {}
    for tag in tags:
        X, names, sets, _ = feature_table(stack, tag, grp)
        m = np.nanmean(X, 0)
        for k, v in zip(names, m):
            if k.startswith(LEVEL_PREFIXES):
                out.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


# ======================================================================
# splits
# ======================================================================
def split_sets(sets, frac_cal=0.5, seed=0):
    """Deterministic split of set ids into calibration and audit halves."""
    u = np.unique(sets)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(u))
    ncal = int(round(frac_cal * len(u)))
    return set(u[perm[:ncal]].tolist()), set(u[perm[ncal:]].tolist())


def rows_in(sets, which):
    return np.isin(sets, list(which))


# ======================================================================
# fitting and scoring
# ======================================================================
MAX_ROWS_PER_CLASS = 60_000


def fit(X1, X0, seed=0):
    """Fit CDM (1) against the class (0).  Each class is capped at
    MAX_ROWS_PER_CLASS rows by a seeded uniform sub-sample, so a fit on the
    full pool (135k class rows) costs the same as one on a small pool; the
    cap is far above the point where the GBDT's audit AUC stops improving."""
    rng = np.random.default_rng(seed)
    if len(X1) > MAX_ROWS_PER_CLASS:
        X1 = X1[rng.choice(len(X1), MAX_ROWS_PER_CLASS, replace=False)]
    if len(X0) > MAX_ROWS_PER_CLASS:
        X0 = X0[rng.choice(len(X0), MAX_ROWS_PER_CLASS, replace=False)]
    X = np.vstack([X1, X0])
    y = np.concatenate([np.ones(len(X1)), np.zeros(len(X0))])
    ok = np.any(np.isfinite(X), 1)
    clf = HistGradientBoostingClassifier(**dict(GBDT, random_state=seed))
    clf.fit(X[ok], y[ok])
    return clf


def logodds(clf, X):
    ok = np.any(np.isfinite(X), 1)
    out = np.zeros(len(X))
    if ok.sum():
        p = np.clip(clf.predict_proba(X[ok])[:, 1], 1e-4, 1 - 1e-4)
        out[ok] = np.log(p / (1 - p))
    return out


def corpus_scores(llr, sets, n_sub=None, rng=None):
    """Per-corpus sum of object log-odds; optional sub-sampling of objects per
    corpus for the sample-size scan."""
    out = {}
    for s in np.unique(sets):
        v = llr[sets == s]
        if n_sub is not None and len(v) > n_sub:
            v = rng.choice(v, size=n_sub, replace=False)
        out[int(s)] = float(np.sum(v))
    return out


def auc(s0, s1):
    a = np.concatenate([s0, s1])
    r = rankdata(a)
    n0, n1 = len(s0), len(s1)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def perm_p(s0, s1, n_perm=3000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(auc(s0, s1) - 0.5)
    a = np.concatenate([s0, s1])
    r = rankdata(a)
    n, n1, n0 = len(a), len(s1), len(s0)
    sel = np.argsort(rng.random((n_perm, n)), axis=1)[:, :n1]
    null = (r[sel].sum(1) - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return float((1.0 + np.sum(np.abs(null - 0.5) >= obs - 1e-12)) / (n_perm + 1.0))


class Test:
    """One trained discriminator: CDM arm(s) vs class arm(s), galaxies and
    clusters fitted separately, combined at the corpus level."""

    def __init__(self, stack, pos_tags, neg_tags, cal, opts=None, seed=0, groups=("gal", "clu")):
        self.opts = opts or {}
        self.groups = groups
        self.clf = {}
        self.names = {}
        for grp in groups:
            X1, names, s1, _ = _concat(stack, pos_tags, grp, self.opts)
            X0, _, s0, _ = _concat(stack, neg_tags, grp, self.opts)
            m1, m0 = rows_in(s1, cal), rows_in(s0, cal)
            if X1.shape[1] == 0:
                continue
            self.clf[grp] = fit(X1[m1], X0[m0], seed=seed)
            self.names[grp] = names

    def score(self, stack, tag, sets_use, n_sub=None, rng=None, per_object=False):
        total = {}
        obj = {}
        for grp in self.groups:
            if grp not in self.clf:
                continue
            X, names, sets, _ = feature_table(stack, tag, grp, **self.opts)
            m = rows_in(sets, sets_use)
            llr = logodds(self.clf[grp], X[m])
            if per_object:
                obj[grp] = (llr, sets[m])
            cs = corpus_scores(llr, sets[m], n_sub=None if n_sub is None else n_sub[grp], rng=rng)
            for s, v in cs.items():
                total[s] = total.get(s, 0.0) + v
        return (total, obj) if per_object else total

    def importance(self, stack, pos_tag, neg_tag, aud, n_rep=3, seed=0):
        """Permutation importance by CHANNEL at the corpus level: the drop in
        audit AUC when a channel's columns are shuffled across rows."""
        out = {}
        rng = np.random.default_rng(seed)
        base = self.auc_pair(stack, pos_tag, neg_tag, aud)
        for grp in self.groups:
            if grp not in self.clf:
                continue
            chan = IV.galaxy_channel if grp == "gal" else IV.cluster_channel
            channels = sorted(set(chan(k) for k in self.names[grp]))
            for c in channels:
                cols = [j for j, k in enumerate(self.names[grp]) if chan(k) == c]
                vals = []
                for rep in range(n_rep):
                    vals.append(self.auc_pair(stack, pos_tag, neg_tag, aud, shuffle=(grp, cols, rng)))
                out[f"{grp}:{c}"] = dict(auc_shuffled=float(np.mean(vals)), drop=float(base - np.mean(vals)))
        out["_base_auc"] = base
        return out

    def auc_pair(self, stack, pos_tag, neg_tag, aud, shuffle=None):
        sp = self._score_shuffled(stack, pos_tag, aud, shuffle)
        sn = self._score_shuffled(stack, neg_tag, aud, shuffle)
        return auc(np.array(list(sn.values())), np.array(list(sp.values())))

    def _score_shuffled(self, stack, tag, sets_use, shuffle):
        total = {}
        for grp in self.groups:
            if grp not in self.clf:
                continue
            X, names, sets, _ = feature_table(stack, tag, grp, **self.opts)
            m = rows_in(sets, sets_use)
            Xm = X[m].copy()
            if shuffle is not None and shuffle[0] == grp:
                _, cols, rng = shuffle
                perm = rng.permutation(len(Xm))
                for j in cols:
                    Xm[:, j] = Xm[perm, j]
            llr = logodds(self.clf[grp], Xm)
            for s, v in corpus_scores(llr, sets[m]).items():
                total[s] = total.get(s, 0.0) + v
        return total


def _concat(stack, tags, grp, opts):
    Xs, S, C = [], [], []
    names = None
    for t in tags:
        X, names, sets, scenes = feature_table(stack, t, grp, **opts)
        Xs.append(X)
        S.append(sets)
        C.append(scenes)
    return np.vstack(Xs), names, np.concatenate(S), np.concatenate(C)


# ======================================================================
# calibrated separation
# ======================================================================
def null_auc_sd(stack, tags, cal, aud, opts=None, n_rep=12, seed=0, groups=("gal", "clu")):
    """A-vs-A: within ONE arm, random halves of the calibration sets are
    labelled 1/0, the discriminator is fitted, and audit sets (also split in
    random halves) are scored.  Returns the sd of the null AUC and the draws."""
    rng = np.random.default_rng(seed)
    draws = []
    for tag in tags:
        for rep in range(n_rep):
            cal_l = list(cal)
            aud_l = list(aud)
            rng.shuffle(cal_l)
            rng.shuffle(aud_l)
            h = len(cal_l) // 2
            ha = len(aud_l) // 2
            c1, c0 = set(cal_l[:h]), set(cal_l[h:])
            a1, a0 = set(aud_l[:ha]), set(aud_l[ha:])
            # fit on cal half 1 (label 1) vs cal half 0 (label 0) of the SAME arm
            clfs = {}
            for grp in groups:
                X, names, sets, _ = feature_table(stack, tag, grp, **(opts or {}))
                if X.shape[1] == 0:
                    continue
                clfs[grp] = (fit(X[rows_in(sets, c1)], X[rows_in(sets, c0)], seed=rep), X, sets)
            s1, s0 = {}, {}
            for grp, (clf, X, sets) in clfs.items():
                for target, use in ((s1, a1), (s0, a0)):
                    m = rows_in(sets, use)
                    for s, v in corpus_scores(logodds(clf, X[m]), sets[m]).items():
                        target[s] = target.get(s, 0.0) + v
            draws.append(auc(np.array(list(s0.values())), np.array(list(s1.values()))))
    draws = np.array(draws)
    return float(np.std(draws)), draws


def separation(scores_pos, scores_neg, sd_null, n_perm=3000, seed=0):
    sp = np.array(list(scores_pos.values()))
    sn = np.array(list(scores_neg.values()))
    a = auc(sn, sp)
    z = float(np.clip(abs(a - 0.5) / max(sd_null, 1e-9), 0.0, Z_CAP))
    return dict(auc=a, z=z, z_capped=bool(abs(a - 0.5) / max(sd_null, 1e-9) > Z_CAP),
                p_perm=perm_p(sn, sp, n_perm=n_perm, seed=seed), n_pos=len(sp), n_neg=len(sn),
                sd_null=sd_null)


def rate_at(scores_pos, scores_neg_cal, scores_neg_aud, alpha=0.05):
    """One-sided detection rate: critical value from the class's CALIBRATION
    corpora, rate measured on CDM corpora and on the class's AUDIT corpora."""
    cal = np.array(list(scores_neg_cal.values()))
    crit = float(np.quantile(cal, 1 - alpha))
    sp = np.array(list(scores_pos.values()))
    sa = np.array(list(scores_neg_aud.values()))
    from universes.stats import rate_with_ci
    return dict(crit=crit, power=rate_with_ci(int(np.sum(sp >= crit)), len(sp)),
                realised_fpr=rate_with_ci(int(np.sum(sa >= crit)), len(sa)))
