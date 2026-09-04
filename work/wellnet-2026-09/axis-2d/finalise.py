"""Merge the four stages into one machine-readable result, with source hashes.

Nothing is recomputed here.  This only collects `axis_power.json`,
`selection.json`, `shear2d.json` and `amplitudes.json` into
`axis_2d_results.json`, records the SHA-256 of every script that produced them,
and prints the handful of numbers the report opens with.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ("axis_power.py", "selection.py", "shear2d.py", "amplitudes.py",
           "crosscheck.py", "report.py", "report_template.py", "finalise.py")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def merge_power():
    """Combine the parallel geometry chunks into one axis_power.json.

    The full grid is run as three processes because a single one takes hours
    on this grid and an earlier single-process run died silently after eight of
    fifteen rows.  Each writes its own file and checkpoints after every row.
    """
    import glob
    parts = sorted(glob.glob(os.path.join(HERE, "axis_power_?.json")))
    if not parts:
        return None
    rows, cg, cfg = [], {}, None
    for f in parts:
        d = json.load(open(f, encoding="utf-8"))
        rows.extend(d.get("rows", []))
        cg = cg or d.get("coarse_grain", {})
        if cfg is None:
            cfg = dict(d["config"])
        else:
            cfg["qs"] = sorted(set(cfg["qs"]) | set(d["config"]["qs"]))
    rows.sort(key=lambda r: (r["axis_ratio"], r["noise"]))
    out = dict(config=cfg, rows=rows, coarse_grain=cg,
               merged_from=[os.path.basename(f) for f in parts])
    with open(os.path.join(HERE, "axis_power.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def surface_summary(ap):
    """The three power surfaces, collapsed to the table the report opens with."""
    if not ap:
        return None
    amps = ap["config"]["amps"]
    top = max(amps)
    out = {}
    for noise in ap["config"]["noise"]:
        rows = [r for r in ap["rows"] if abs(r["noise"] - noise) < 1e-12]
        tab = []
        for r in sorted(rows, key=lambda x: x["axis_ratio"]):
            e = dict(axis_ratio=r["axis_ratio"])
            for p in ("source", "external", "network"):
                cells = [c for c in r["power"].values()
                         if c["provenance"] == p]
                e[p] = {str(a): float(np.mean(
                    [c["power"] for c in cells
                     if abs(c["amp"] - a) < 1e-12])) for a in amps}
                kn = [c["power_axis_known"] for c in cells
                      if abs(c["amp"] - top) < 1e-12
                      and c["power_axis_known"] is not None]
                e[p + "_axis_known_top"] = float(np.mean(kn)) if kn else None
                e[p + "_audit_fpr"] = r["audit_fpr"][p]
            tab.append(e)
        out[str(noise)] = tab
    return out


def manifest():
    """Provenance for everything this lane pulled off the network."""
    import glob
    cache = os.path.join(HERE, "cache")
    npz = sorted(glob.glob(os.path.join(cache, "shear_*.npz")))
    pos = os.path.join(cache, "selection_positions.json")
    n_src = 0
    for f in npz:
        try:
            n_src += int(np.load(f)["n"])
        except Exception:                                       # noqa: BLE001
            pass
    ex = {}
    for tag, fn in (("selection", "selection_raw_example"),
                    ("shear", "shear_raw_example")):
        q = os.path.join(cache, fn + ".query.txt")
        c = os.path.join(cache, fn + ".csv")
        if os.path.exists(q):
            ex[tag] = dict(query=open(q, encoding="utf-8").read(),
                           raw_bytes=(os.path.getsize(c)
                                      if os.path.exists(c) else None),
                           raw_sha256=(sha(c) if os.path.exists(c) else None))
    m = dict(
        source="NOIRLab Astro Data Lab TAP, table delve_dr3.decade_shear "
               "(DECADE metacalibration shape catalogue, DELVE DR3). "
               "Unauthenticated; no credentials are sent.",
        endpoint="https://datalab.noirlab.edu/tap/sync",
        retrieved_utc=dt.datetime.now(dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        x_ray_catalogue="Bahar+2022, VizieR J/A+A/661/A7 tables 1 and 2, "
                        "542 eFEDS systems, already on disk in ../lead01/ "
                        "with its own manifests",
        selection_pass=dict(
            file="cache/selection_positions.json",
            note="POSITIONS AND REDSHIFTS ONLY -- this pass never requests a "
                 "shape column, so the sample cannot have been selected on "
                 "the quantity that is scored",
            n_systems=(len(json.load(open(pos, encoding="utf-8")))
                       if os.path.exists(pos) else 0),
            sha256=sha(pos) if os.path.exists(pos) else None,
            bytes=os.path.getsize(pos) if os.path.exists(pos) else None),
        shear_pass=dict(
            files=f"cache/shear_<id>.npz ({len(npz)} systems)",
            n_background_sources=n_src,
            columns=["ra", "dec", "mcal_g_1_noshear", "mcal_g_2_noshear",
                     "mcal_w_noshear", "dnf_z", "mcal_g_1_1p", "mcal_g_1_1m",
                     "mcal_g_2_2p", "mcal_g_2_2m"],
            note="metacalibration response recovered per radial bin from the "
                 "1p/1m/2p/2m sheared copies; the SELECTION response is not "
                 "applied, which biases the shear AMPLITUDE by a few per cent "
                 "but cancels in every ratio reported here"),
        raw_examples_kept=ex,
        NOT_USED=dict(
            zenodo_hsc_y3_gama09h=[
                "10.5281/zenodo.15482851 and its two siblings, titled "
                "'HSC Y3 Shape Catalog -- GAMA09H Full Field'.  These describe "
                "exactly this field with e1/e2/RA/Dec and are SYNTHETIC: the "
                "creator list includes an LLM simulation assistant and the "
                "correction notice calls one of them 'an algorithmically "
                "scaled or pipeline-derived artifact'.  Not downloaded, not "
                "used, not consulted."],
            sealed_holdouts=["KiDS", "wide binaries -- never loaded"]))
    with open(os.path.join(cache, "decade_cache.manifest.json"), "w") as f:
        json.dump(m, f, indent=1)
    return m


def main():
    print("=" * 78)
    print("FINALISE -- axis_2d_results.json")
    print("=" * 78)
    man = manifest()
    print(f"\n   manifest: {man['shear_pass']['n_background_sources']:,} "
          f"background sources cached over "
          f"{man['shear_pass']['files']}")
    ap = merge_power() or load("axis_power.json")
    sel = load("selection.json")
    s2 = load("shear2d.json")
    am = load("amplitudes.json")
    out = dict(
        generated_utc=dt.datetime.now(dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        lane="work/wellnet-2026-09/axis-2d",
        source_sha256={s: sha(os.path.join(HERE, s)) for s in SCRIPTS
                       if os.path.exists(os.path.join(HERE, s))},
        job1_power_surfaces=ap,
        job1_summary=surface_summary(ap),
        job2_shear2d=s2,
        job3_selection=(dict(
            selection_function=sel["selection_function"],
            parent=sel["parent"],
            mask_systematic_rho=sel["mask_systematic_rho"],
            neighbour_shear_median=sel["neighbour_shear_median"],
            neighbour_shear_p90=sel["neighbour_shear_p90"],
            n_dev=len(sel["dev"]), n_ctrl=len(sel["ctrl"]),
            dev=sel["dev"], ctrl=sel["ctrl"],
            sha256=sha(os.path.join(HERE, "selection.json")))
            if sel else None),
        job4_amplitudes=am,
        crosscheck=load("crosscheck.json"),
        data_manifest=man)
    with open(os.path.join(HERE, "axis_2d_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n   written: axis_2d_results.json")
    for k, v in out["source_sha256"].items():
        print(f"      {k:16s} {v[:16]}")
    if ap:
        print(f"\n   power surfaces: {len(ap['rows'])} (geometry, noise) rows")
    if sel:
        print(f"   selection: DEV {len(sel['dev'])}, CTRL {len(sel['ctrl'])}")
    if s2:
        for tag, d in s2["samples"].items():
            if not d.get("n"):
                continue
            ch = d.get("channels", {}).get("a2s_pred", {})
            f_ = ch.get("fit") or {}
            print(f"   {tag}: n = {d['n']}, alpha = "
                  f"{f_.get('alpha', float('nan')):+.4f} "
                  f"+- {f_.get('e_alpha', float('nan')):.4f}, "
                  f"p = {ch.get('p_two_sided')}")


if __name__ == "__main__":
    main()
