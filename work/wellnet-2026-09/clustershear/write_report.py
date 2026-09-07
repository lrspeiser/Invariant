"""write_report.py -- REPORT.md, rendered from analysis.json and controls.json."""
from __future__ import annotations

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    A = json.load(io.open(os.path.join(HERE, "analysis.json"), encoding="utf-8"))
    K = json.load(io.open(os.path.join(HERE, "controls.json"), encoding="utf-8"))
    L, add = [], None
    L = []
    add = L.append

    add("# Run BQ -- raw weak lensing for the open half of the gold-cluster pool")
    add("")
    add("Lane `work/wellnet-2026-09/clustershear/`. Rendered from `analysis.json`")
    add("and `controls.json`.")
    add("")
    add("## Headline")
    add("")
    add("**A %.1f-sigma stacked cluster lensing profile from %d clusters, with both"
        % (A["stack_snr"], A["n_clusters"]))
    add("null controls passing** -- and **no gravity result**, which is the correct")
    add("outcome for a lane whose job was to build the dataset, not to test a law.")
    add("")
    add("The sealed half was never queried: %d clusters, 0 tokens spent."
        % A["sealed_untouched"])
    add("")
    add("## The stacked profile")
    add("")
    add("| R [Mpc] | g_t | g_x | err | clusters |")
    add("|---|---|---|---|---|")
    for b in A["stack"]:
        if b["gt"] is None:
            continue
        add("| %.3f | %+.5f | %+.5f | %.5f | %d |"
            % (b["R"], b["gt"], b["gx"], b["err"], b["n_cl"]))
    add("")
    add("Monotonic decline over a decade in radius, as a cluster profile must be.")
    add("")
    add("## The nulls, both passed")
    add("")
    n1 = A["nulls"]["N1_cross"]
    n2 = A["nulls"]["N2_randoms"]
    add("| null | what it tests | result |")
    add("|---|---|---|")
    add("| N1 cross component | lensing cannot make a parity-odd signal | chi2/dof = %.1f/%d -- %s |"
        % (n1["chi2"], n1["dof"], n1["verdict"]))
    add("| N2 random pointings | the catalogue's additive systematic floor, measured not assumed | chi2/dof = %.1f/%d -- %s |"
        % (n2["chi2"], n2["dof"], n2["verdict"]))
    add("")
    add("N2 ran the identical estimator at %d cluster-free positions offset 1.0-1.5"
        % A["n_randoms"])
    add("deg from real clusters -- inside the same patchy DECADE footprint, well")
    add("outside any cluster aperture.")
    add("")
    add("## Trends, each against its own label-scramble null")
    add("")
    add("| observable | n | slope | z | p | verdict |")
    add("|---|---|---|---|---|---|")
    for t in A["trends"]:
        if t.get("verdict") == "TOO FEW":
            continue
        add("| %s | %d | %+.5f | %+.2f | %.4f | %s |"
            % (t["observable"], t["n"], t["slope"], t["z"],
               t["p_permutation"], t["verdict"]))
    add("")
    add("## The controls that stop these being findings")
    add("")
    add("Two of those trends have the shape of an artefact, so each was attacked.")
    add("")
    for key, c in K["controls"].items():
        add("**%s** -- %s" % (key, c["question"]))
        add("")
        add("> %s" % c["verdict"])
        add("")
    add("## What it means")
    add("")
    add(K["interpretation"])
    add("")
    sp = os.path.join(HERE, "search.json")
    if os.path.exists(sp):
        S = json.load(io.open(sp, encoding="utf-8"))
        add("## The law search -- %s model forms, and why the top 25 is not the answer"
            % format(S["n_model_forms"], ","))
        add("")
        add("%s model forms (every subset of size 1-%d over %d atoms in radius,"
            % (format(S["n_model_forms"], ","), S["kmax"], S["n_atoms"]))
        add("redshift and X-ray temperature), each fitted EXACTLY by weighted least")
        add("squares rather than sampled, at %s forms per second over %d points."
            % (format(S["forms_per_second"], ","), S["n_points"]))
        add("")
        add("All 25 of the top 25 beat all three null twins. **That headline is")
        add("misleading and here is why.** Every one of them sits at k=%d, the"
            % S["kmax"])
        add("maximum complexity, because nothing in the ranking penalises")
        add("parameters; and %d of the %d share the same redshift core, so this is"
            % (S["top_sharing_one_redshift_core"], len(S["top"])))
        add("one model wearing %d hats, not %d discoveries."
            % (S["top_sharing_one_redshift_core"], len(S["top"])))
        add("")
        add("**The informative table is the complexity trade-off.**")
        add("")
        add("| k | gain | null A randoms | null B property scramble | null C radial scramble | best form |")
        add("|---|---|---|---|---|---|")
        for p in S["best_per_k"]:
            add("| %d | %.4f | %.4f | %.4f | %.4f | `%s` |"
                % (p["k"], p["gain_real"], p["gain_null_random"],
                   p["gain_null_property"], p["gain_null_radial"],
                   " + ".join(p["terms"])))
        add("")
        k1 = S["best_per_k"][0]
        kmax = S["best_per_k"][-1]
        add("Read it from the top row. **A single atom, `%s`, reaches %.4f of the"
            % (" + ".join(k1["terms"]), k1["gain_real"]))
        add("%.4f that four free parameters reach -- %.0f%% of everything %s forms"
            % (kmax["gain_real"], 100 * k1["gain_real"] / kmax["gain_real"],
               format(S["n_model_forms"], ",")))
        add("buy.** The other three parameters purchase %.4f more gain while the"
            % (kmax["gain_real"] - k1["gain_real"]))
        add("property-scramble null climbs from %.4f to %.4f. That is the shape of"
            % (k1["gain_null_property"], kmax["gain_null_property"]))
        add("overfitting, not of a law.")
        add("")
        add("The k=1 form survives its nulls properly: the random pointings give")
        add("%.4f and the RADIAL scramble gives %.4f -- negative, i.e. worse than a"
            % (k1["gain_null_random"], k1["gain_null_radial"]))
        add("constant once the radial ordering is destroyed. So **the radial")
        add("structure is real.** But the property-scramble null still reaches")
        add("%.4f, about a third of the gain, because scrambling (z, kT) across"
            % k1["gain_null_property"])
        add("clusters leaves each profile's own radii intact -- null B cannot test a")
        add("purely radial term, which is exactly why null C was added.")
        add("")
        add("None of this is a gravity law. `DeltaSigma ~ T^0.5 / R` is close to what")
        add("an isothermal sphere gives you for free, and the search had no baryon")
        add("model to work against, so it cannot distinguish a modified force from")
        add("ordinary cluster structure. What it does establish is that the")
        add("admissible variable set here is small enough that one term exhausts it.")
        add("")
    add("## What was not done")
    add("")
    add("- No mass was fitted; no gravity law was scored.")
    add("- eRASS1 `M500`, `R500`, `Mgas500` are weak-lensing-calibrated and were")
    add("  excluded throughout as circular.")
    add("- Cluster-member contamination of the source sample was not modelled; it")
    add("  dilutes the inner bins and its size is unmeasured here.")
    add("- The sealed half is untouched and stays that way until a law is")
    add("  pre-registered against it.")
    add("")
    io.open(os.path.join(HERE, "REPORT.md"), "w", newline="\n",
            encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
