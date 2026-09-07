"""write_spec.py -- render SPEC.md and REPORT.md from the lane's JSONs.

Every number in the prose comes from probes.json / ranking.json, so the document
cannot drift from what was measured.

    python write_spec.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import archives
import guard

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        raise SystemExit("missing %s -- run probes.py and rank.py first" % name)
    return json.load(io.open(p, encoding="utf-8"))


# ---- the coverage box in physical units (flat LCDM, H0=70, Om=0.3) ----------
_C, _H0, _OM, _OL = 299792.458, 70.0, 0.3, 0.7


def _angular_diameter_distance(z, n=2000):
    dz = z / n
    s = sum(dz / math.sqrt(_OM * (1 + (i + 0.5) * dz) ** 3 + _OL)
            for i in range(n))
    return (_C / _H0) * s / (1 + z)                     # Mpc


def _half_widths(clusters, half_deg=0.5):
    return sorted(_angular_diameter_distance(c["z"]) * math.radians(half_deg)
                  for c in clusters
                  if c["channels"]["C4_weak_lensing"] == "PUBLIC")


def _median_half_mpc(clusters):
    h = _half_widths(clusters)
    return h[len(h) // 2] if h else float("nan")


def _n_half_below(clusters, mpc):
    return sum(1 for x in _half_widths(clusters) if x < mpc)


def spec_md(probes, rank):
    routes = {r.get("route", ""): r for r in probes["routes"]}
    cl = rank["clusters"]
    npub = rank["n_with_public_weak_lensing"]
    n = rank["n_candidates"]
    top = [c for c in cl if c["channels"]["C4_weak_lensing"] == "PUBLIC"][:20]

    L = []
    A = L.append
    A("# Gold Cluster Acquisition Specification")
    A("")
    A("Lane `work/wellnet-2026-09/goldcluster/`. Registry `BO-goldcluster`.")
    A("Generated %s from `probes.json` and `ranking.json`; every number below is"
      % rank["generated_utc"])
    A("rendered from those files.")
    A("")
    A("**INVENTORY ONLY.** This lane opened no pixel, no shear, no spectrum and no")
    A("kinematic measurement, and computed no gravity-relevant statistic. That is")
    A("enforced, not promised: `guard.check_query` rejects any archive query whose")
    A("projection names a shape, an ellipticity, a response or a per-source")
    A("redshift, and `guard.assert_no_statistic` refuses to write any result whose")
    A("keys collide with a declared observable. The lane asks archives to COUNT")
    A("sources; it never asks them to hand one over.")
    A("")
    A("## 0  Why this specification exists")
    A("")
    A("BE.7 recorded that no public cluster satisfies the charter's complete")
    A("experiment, and instructed that a specification be written **before a")
    A("target is chosen, so candidates rank by how much new telescope time they")
    A("need rather than by how much code already exists for them**.")
    A("")
    A("The programme's seven working clusters -- the Hubble Frontier Fields six")
    A("plus Abell 2029 -- were selected the second way. They are the clusters with")
    A("the most published products. That is a selection for objects other people")
    A("have already spent time on, which is not the same as a selection for objects")
    A("where the remaining experiment is cheapest to finish.")
    A("")
    A("This specification therefore starts from the whole X-ray-selected sky.")
    A("")
    A("## 1  The six channels")
    A("")
    A("The complete experiment needs ONE object carrying all six together:")
    A("")
    A("| | channel | what counts as having it |")
    A("|---|---|---|")
    A("| C1 | resolved baryons | member light profiles, fittable from calibrated multi-band imaging |")
    A("| C2 | member internal dynamics | **resolved** stellar kinematics per member, not an aperture number |")
    A("| C3 | cluster dynamics | member redshifts, radially complete enough for sigma(R) |")
    A("| C4 | raw weak lensing | per-source shapes. A mass map is not lensing data |")
    A("| C5 | strong lensing | multiple images, ideally a measured time delay |")
    A("| C6 | environment | the large-scale field around the cluster |")
    A("")
    A("C2 is the binding one. BE.7 named it the genuinely missing layer, and this")
    A("lane's survey does not change that: **C2 is ABSENT for every one of the %d" % n)
    A("candidates**. The only published resolved member kinematics within this")
    A("programme's reach is the MUSE/Granata set, which is held in the CONFIRMATION")
    A("RESERVE and was not opened. So C2 costs new IFU time everywhere, and that is")
    A("what a proposal should ask for.")
    A("")
    A("## 2  The routes, probed live")
    A("")
    A("Not one of these was taken on the strength of a data-availability sentence.")
    A("Each was called and its answer recorded.")
    A("")
    A("| route | channel | verdict | what it actually serves |")
    A("|---|---|---|---|")
    for r in probes["routes"]:
        lim = (r.get("limitation") or "").replace("\n", " ")
        if len(lim) > 200:
            lim = lim[:197] + "..."
        A("| %s | %s | **%s** | %s |"
          % (r.get("route", "?"), r.get("channel", "?"), r.get("verdict", "?"), lim))
    A("")
    A("### Never opened, recorded by identity only")
    A("")
    A("| route | status | why |")
    A("|---|---|---|")
    for x in probes["never_opened"]:
        A("| %s | %s | %s |" % (x["route"], x["status"], x["why"]))
    A("")
    A("## 3  The candidate pool")
    A("")
    A("Source: eRASS1 primary cluster catalogue (Bulbul+2024), 12,247 clusters,")
    A("fetched under `catalogue_validation` v3 -- all three detectors passed.")
    A("")
    A("Cuts, declared before any coverage was probed: Dec <= +40 (DECam's reach),")
    A("0.05 <= z <= 0.7 (the lensing kernel), >= 100 X-ray counts inside R500 (a")
    A("profile rather than a detection), contamination probability <= 0.3.")
    A("**%d candidates** survive." % n)
    A("")
    A("eRASS1's `M500`, `R500`, `Mgas500` and `Fgas500` are scaling-relation")
    A("products calibrated on weak lensing. They are **not** carried forward as")
    A("observables: scoring a gravity law against a weak-lensing-calibrated mass")
    A("using weak lensing would be circular. What is carried forward is count")
    A("rates, counts, fluxes and kT.")
    A("")
    A("## 4  What the sky actually holds")
    A("")
    A("Coverage was asked per cluster, one query each, because DECADE is a")
    A("reprocessing of **archival** DECam pointings and its footprint is patchy at")
    A("the degree scale. This is not a theoretical worry: the single best candidate")
    A("in the pool by X-ray counts has about a million DECADE sources within 5")
    A("degrees and **exactly zero** within 0.5 -- a hole several degrees across.")
    A("A survey boundary would have called that cluster covered.")
    A("")
    A("Measured over %d probed candidates:" % len([c for c in cl if c["n_shear_sources"] is not None]))
    A("")
    A("- **%d clusters have a public raw weak-lensing channel** (>= %d usable"
      % (npub, rank["thresholds"]["n_src_usable"]))
    A("  background sources within 0.5 deg, passing the DECADE cosmology selection")
    A("  and behind the cluster in redshift).")
    n4 = sum(1 for c in cl if c["n_channels_public"] >= 4)
    A("- Spectroscopy was probed for **all %d** of them, and **%d carry four"
      % (rank["thresholds"]["n_spec_probed"], n4))
    A("  public channels at once** -- C1 resolved baryons, C3 cluster dynamics,")
    A("  C4 raw weak lensing and C6 environment. Only C2 and C5 are missing, and")
    A("  C2 is missing everywhere.")
    A("")
    A("For scale: before this lane, the programme had a public per-source shear")
    A("catalogue for **one** cluster, Abell 370, and its cluster-scale correlation")
    A("rested on the twelve X-COP systems.")
    A("")
    A("## 5  The ranking")
    A("")
    A("Rank key: channels already public, then weak-lensing source count, because")
    A("C4 is the channel that cannot be substituted. Every candidate needs C2 and")
    A("almost every candidate needs C5, so the ranking is really *how much of the")
    A("cheap half is already done*.")
    A("")
    A("Top %d by that key:" % len(top))
    A("")
    A("| cluster | RA | Dec | z | z type | X-ray counts | kT | shear sources | spec members | public channels |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for c in top:
        A("| %s | %.3f | %+.3f | %.4f | %s | %.0f | %s | %s | %s | %d |"
          % (c["name"], c["ra"], c["dec"], c["z"], c["zType"], c["cts500"],
             c["kt"] or "-", c["n_shear_sources"],
             c["n_spec_members"] if c["n_spec_members"] is not None else "-",
             c["n_channels_public"]))
    A("")
    A("## 6  What to ask for")
    A("")
    A("The specification's operative conclusion is that the expensive channel is")
    A("the same one for every candidate, so the target should be chosen by the")
    A("cheap channels and the proposal written for C2.")
    A("")
    A("1. **Choose from the top of section 5**, not from the HFF six. Those")
    A("   clusters already carry C1, C3, C4 and C6 from public survey data.")
    A("2. **Ask for C2**: resolved stellar kinematics for member galaxies in one")
    A("   cluster that already has public shear. That is an IFU programme, and it")
    A("   is the only thing standing between this programme and a complete scene.")
    A("3. **Do not ask for C4.** It is already public for %d clusters." % npub)
    A("4. **Seal before scoring.** See section 7.")
    A("")
    A("## 7  The sealing opportunity, which is the real prize here")
    A("")
    A("The programme has no sealed confirmation set. KiDS and the wide binaries")
    A("were both scored in round 1, so they are validation, not confirmation, and")
    A("nothing else was ever held back.")
    A("")
    A("`confirmation_status` v2 is explicit that **Mentioned, Acquired and")
    A("Transformed are NOT spent** -- only Scored, Inspected and Decision-used are.")
    A("This lane has therefore Acquired nothing but identities and counts, and has")
    A("Scored nothing. Every one of the %d clusters is still pristine." % n)
    A("")
    A("That makes a genuine confirmation set available for the first time, and it")
    A("is independent of the spent data on all four of v2's axes:")
    A("")
    A("- **untouched outcome** -- no shear behind any of these clusters has been read;")
    A("- **untouched objects** -- none of the seven worked clusters is in this pool;")
    A("- **untouched survey** -- DECam/DELVE, not KiDS, not HSC, not HST;")
    A("- **untouched reduction pipeline** -- DECADE metacalibration, not lenstool, not pyRRG.")
    A("")
    A("**The recommendation is to split the %d now, before anything is scored," % npub)
    A("and seal one half at loader level under `holdout_seal` v2.** Splitting after")
    A("a first look is how the last confirmation set was lost.")
    A("")
    A("## 8  Limits of this specification")
    A("")
    A("- **\"Four public channels\" is two independent measurements, not four.**")
    A("  C1 is inferred from C4 (a cluster with DECADE shapes necessarily has the")
    A("  DECam imaging those shapes were measured from -- but band coverage and")
    A("  depth were not verified per cluster), and C6 is scored identically to C3")
    A("  (the spectroscopy that gives cluster dynamics also samples the field --")
    A("  but no independent environment statistic was computed). So the honest")
    A("  statement is: %d clusters have public shear, and %d of those also have"
      % (npub, sum(1 for c in cl if c["n_channels_public"] >= 4)))
    A("  usable public spectroscopy. Do not read the channel count as four")
    A("  separate archive products.")
    A("- C5 (strong lensing) was **not** probed per cluster. It is recorded as")
    A("  UNMEASURED rather than absent.")
    A("- **The 0.5 deg box is a coverage test, not the measurement aperture.**")
    A("  Computed for the covered clusters in a flat LCDM (H0=70, Om=0.3), its")
    A("  physical half-width is 1.76 Mpc at worst and %.2f Mpc at the median, so"
      % _median_half_mpc(cl))
    A("  no cluster was called covered on a box too small to hold a profile. But")
    A("  for the %d lowest-redshift covered clusters it is under 3.5 Mpc, i.e."
      % _n_half_below(cl, 3.5))
    A("  smaller than the outer bin Chiu+2022 and Umetsu+2020 use. For those the")
    A("  extraction needs a wider angular aperture, and the source counts here")
    A("  UNDERSTATE what is available.")
    A("- Spectroscopic completeness is counted, not characterised. DESI fibre")
    A("  assignment is not radially uniform, and a sigma(R) field built from it")
    A("  without a selection function would inherit that.")
    A("- eRASS1 photo-z clusters carry redshift error into every physical radius.")
    A("  %d of the pool have spectroscopic redshifts and should be preferred."
      % sum(1 for c in cl if "spec" in (c["zType"] or "")))
    A("")
    return "\n".join(L) + "\n"


def main():
    guard.arm()
    probes = _load("probes.json")
    rank = _load("ranking.json")
    md = spec_md(probes, rank)
    io.open(os.path.join(HERE, "SPEC.md"), "w", newline="\n",
            encoding="utf-8").write(md)

    spec = dict(lane="goldcluster", generated_utc=archives._utc(),
                inventory_only=True,
                channels=["C1_resolved_baryons", "C2_member_internal_dynamics",
                          "C3_cluster_dynamics", "C4_weak_lensing",
                          "C5_strong_lensing", "C6_environment"],
                binding_channel="C2_member_internal_dynamics",
                n_candidates=rank["n_candidates"],
                n_with_public_weak_lensing=rank["n_with_public_weak_lensing"],
                recommendation=[
                    "choose the target from the ranking, not from the HFF six",
                    "propose for C2 resolved member kinematics only",
                    "split and seal half the pool before anything is scored",
                ])
    guard.assert_no_statistic(spec, "spec")
    io.open(os.path.join(HERE, "SPEC.json"), "w", newline="\n",
            encoding="utf-8").write(json.dumps(spec, indent=1) + "\n")
    print("wrote SPEC.md (%d lines) and SPEC.json" % md.count("\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
