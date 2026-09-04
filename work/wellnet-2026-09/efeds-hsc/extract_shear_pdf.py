"""Exact recovery of the published stacked shear profile from a vector PDF.

Chiu et al. 2022 (A&A 661, A11) publish the stacked eFEDS tangential reduced
shear profile as a figure only -- it is in no table, in no VizieR catalogue and
in no repository (see acquire/access_probes.json and REPORT.md Section 1).

The figure is a matplotlib-produced vector PDF, so this is NOT a digitisation:
the marker centres, the error-bar endpoints and the axis tick positions are
stored as exact PostScript-style coordinates in the content stream.  Recovering
them and inverting the (linear in log) axis transform reproduces the plotted
numbers to the precision of the PDF coordinate format, ~1e-6 pt, which is
~5e-9 dex -- six orders of magnitude below the measurement error.

The axis calibration is FITTED from the four labelled major ticks on each axis
and the fit residual is printed as a gate: a log-axis must give a residual at
round-off, and anything larger means the axis is not what it looks like.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import zlib
import datetime as dt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "acquire", "chiu_src", "materials", "g.pdf")
OUT = os.path.join(HERE, "efeds_stacked_shear.tsv")


def streams(path):
    d = open(path, "rb").read()
    out = []
    for m in re.finditer(rb"stream\r?\n", d):
        s, e = m.end(), d.find(b"endstream", m.end())
        try:
            out.append(zlib.decompress(d[s:e]))
        except zlib.error:
            pass
    return d, b"\n".join(out).decode("latin1")


NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def major_ticks(txt, axis):
    """Return [(pixel, label_value)] for the labelled major ticks.

    Major ticks are drawn at width 0.8 as a 3.5 pt stub off the axis, and the
    very next text-showing operator carries the label.
    """
    if axis == "x":
        pat = re.compile(NUM + r" 48\.6 m\s*\n" + NUM + r" 45\.1 l")
    else:
        pat = re.compile(r"48\.6 " + NUM + r" m\s*\n45\.1 " + NUM + r" l")
    got = []
    for m in pat.finditer(txt):
        pix = float(m.group(1))
        tail = txt[m.end():m.end() + 400]
        lab = re.search(r"\(([-+]?[\d.]+)\) Tj", tail)
        if lab:
            got.append((pix, float(lab.group(1))))
    return got


def calibrate(ticks, name):
    pix = np.array([t[0] for t in ticks])
    val = np.log10(np.array([t[1] for t in ticks]))
    A = np.vstack([pix, np.ones_like(pix)]).T
    coef, *_ = np.linalg.lstsq(A, val, rcond=None)
    resid = val - A @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    print(f"   {name}: {len(ticks)} labelled ticks "
          f"{[t[1] for t in ticks]}, log-linear fit residual "
          f"{rms:.3e} dex")
    if rms > 1e-6:
        raise SystemExit(f"{name} axis is not log-linear (rms {rms:.3e}); "
                         "the calibration assumption is wrong")
    return coef                                   # log10(value) = a*pix + b


def markers(txt, name):
    """Cumulative-transform marker positions for an XObject glyph."""
    blocks = re.findall(r"q\s*\n((?:1 0 0 1 [^\n]*Do\s*\n)+)Q", txt)
    out = []
    for b in blocks:
        rows = re.findall(r"1 0 0 1 " + NUM + r" " + NUM + r" cm /(\w+) Do", b)
        if not rows or rows[0][2] != name:
            continue
        x = y = 0.0
        for dx, dy, _ in rows:
            x += float(dx)
            y += float(dy)
            out.append((x, y))
    return out


def bars(txt):
    """Vertical error-bar segments drawn at line width 1.5."""
    segs = re.findall(NUM + r" " + NUM + r" m\s*\n" + NUM + r" " + NUM
                      + r" l\s*\n\s*\nS", txt)
    out = []
    for x1, y1, x2, y2 in segs:
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        # strictly inside the axes rectangle, so the two frame verticals at
        # x = 48.6 and x = 317.52 are excluded
        if abs(x1 - x2) < 1e-9 and y2 > y1 and 48.6 < x1 < 317.52:
            out.append((x1, y1, y2))
    return out


def curves(txt):
    """The two 10-point model polylines (dashed blue, dotted red)."""
    out = []
    for m in re.finditer(r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\n?\s*RG\s*"
                         r"/DeviceRGB cs\s*\n\s*\n"
                         r"((?:[\d.]+ [\d.]+ [ml]\s*\n)+)", txt):
        pts = re.findall(NUM + r" " + NUM + r" [ml]", m.group(4))
        if len(pts) >= 8:
            rgb = tuple(round(float(m.group(i)), 3) for i in (1, 2, 3))
            out.append((str(rgb), [(float(a), float(b)) for a, b in pts]))
    return out


def main():
    print("=" * 78)
    print("EXTRACT -- Chiu+2022 stacked eFEDS tangential reduced shear profile")
    print("=" * 78)
    raw, txt = streams(PDF)
    print(f"\n   {os.path.relpath(PDF, HERE)}  "
          f"{len(raw)} bytes  sha256 {hashlib.sha256(raw).hexdigest()[:16]}...")

    print("\n   axis calibration (gate: log-linear residual < 1e-6 dex)")
    cx = calibrate(major_ticks(txt, "x"), "x  R [h^-1 Mpc]")
    cy = calibrate(major_ticks(txt, "y"), "y  g_+")

    def X(p):
        return 10.0 ** (cx[0] * p + cx[1])

    def Y(p):
        return 10.0 ** (cy[0] * p + cy[1])

    filled = markers(txt, "M0")           # used in the fiducial lensing fit
    open_c = markers(txt, "M1")           # R < 0.5 h^-1 Mpc, excluded there
    allseg = bars(txt)
    mx = [x for x, _ in filled] + [x for x, _ in open_c]
    # keep only bars whose x coincides with a data marker; this drops the
    # error-bar sample drawn inside the legend box
    seg = [s for s in allseg if min(abs(s[0] - m) for m in mx) < 1e-3]
    print(f"\n   markers: {len(filled)} filled, {len(open_c)} open; "
          f"{len(seg)} data error bars ({len(allseg) - len(seg)} rejected as "
          f"legend/frame)")
    if len(filled) + len(open_c) != 10 or len(seg) != 10:
        raise SystemExit("expected 10 radial bins and 10 error bars "
                         f"(got {len(filled) + len(open_c)} / {len(seg)})")

    pts = [(x, y, 0) for x, y in open_c] + [(x, y, 1) for x, y in filled]
    pts.sort()
    seg.sort()
    rows = []
    for (px, py, used), (bx, lo, hi) in zip(pts, seg):
        if abs(px - bx) > 1e-3:
            raise SystemExit(f"marker/bar x mismatch {px} vs {bx}")
        g, glo, ghi = Y(py), Y(lo), Y(hi)
        rows.append(dict(R_hinvMpc=X(px), g_plus=g,
                         err_lo=g - glo, err_hi=ghi - g, used_fiducial=used))

    # gate: the bar must be symmetric in LINEAR g (it is a +-1 sigma bar drawn
    # on a log axis), which also proves the marker is the central value
    asym = np.array([abs(r["err_hi"] - r["err_lo"]) / r["err_hi"]
                     for r in rows])
    print(f"   gate: |err_hi - err_lo| / err_hi  max {asym.max():.2e}  "
          f"=> bars are symmetric in linear g, marker is the central value")
    if asym.max() > 5e-3:
        raise SystemExit("error bars are not symmetric in linear g; the "
                         "marker may not be the plotted central value")

    # radial binning gate: Chiu Sect. 4.4 declares ten logarithmic bins
    # between 0.2 and 3.5 h^-1 Mpc
    R = np.array([r["R_hinvMpc"] for r in rows])
    dlog = np.diff(np.log10(R))
    expect = np.log10(3.5 / 0.2) / 10.0
    print(f"   gate: bin spacing {dlog.mean():.6f} +- {dlog.std():.2e} dex, "
          f"paper declares {expect:.6f} "
          f"({100 * abs(dlog.mean() - expect) / expect:.2f}% off)")
    if dlog.std() > 1e-4 or abs(dlog.mean() - expect) / expect > 0.02:
        raise SystemExit("radial binning does not match the paper's stated "
                         "ten log bins over 0.2-3.5 h^-1 Mpc")

    mods = curves(txt)
    print(f"   model curves recovered: {len(mods)} "
          f"({', '.join(m[0] for m in mods)})")
    labels = ["bestfit_miscentered", "bestfit_no_miscentering"]
    for lab, (rgb, pt) in zip(labels, mods):
        p = os.path.join(HERE, f"chiu2022_model_{lab}.tsv")
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"# Chiu+2022 Fig. 8 model curve, RGB {rgb}\n")
            f.write("R_hinvMpc\tg_plus\n")
            for a, b in pt:
                f.write(f"{X(a):.6f}\t{Y(b):.8f}\n")
        print(f"      wrote chiu2022_model_{lab}.tsv ({len(pt)} points)")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Chiu et al. 2022 A&A 661 A11, Fig. 8: stacked tangential\n"
                "# reduced shear profile of the eFEDS cluster sample, recovered\n"
                "# from the vector content stream of arXiv:2107.05652 e-print\n"
                "# file materials/g.pdf.  Errors are +-1 sigma, symmetric in\n"
                "# linear g.  used_fiducial = 0 for the three R < 0.5 h^-1 Mpc\n"
                "# bins the paper excludes from its fiducial lensing model.\n")
        f.write("R_hinvMpc\tg_plus\terr_lo\terr_hi\tused_fiducial\n")
        for r in rows:
            f.write(f"{r['R_hinvMpc']:.6f}\t{r['g_plus']:.8f}\t"
                    f"{r['err_lo']:.8f}\t{r['err_hi']:.8f}\t"
                    f"{r['used_fiducial']}\n")

    blob = open(OUT, "rb").read()
    meta = {
        "file": os.path.basename(OUT),
        "source_url": "https://arxiv.org/e-print/2107.05652",
        "source_file_within_eprint": "materials/g.pdf",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "paper": "Chiu I.-N. et al. 2022, A&A 661, A11 "
                 "(2022A&A...661A..11C), Fig. 8",
        "retrieved_utc": dt.datetime.now(dt.timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "row_count": len(rows),
        "column_count": 5,
        "columns": [
            {"name": "R_hinvMpc", "unit": "h^-1 Mpc",
             "note": "clustercentric physical radius, fiducial cosmology "
                     "Om=0.3 h=0.7"},
            {"name": "g_plus", "unit": "dimensionless",
             "note": "stacked tangential REDUCED shear g_+ = gamma/(1-kappa), "
                     "lensing-weighted over the 313 eFEDS systems with HSC "
                     "S19A coverage"},
            {"name": "err_lo", "unit": "dimensionless"},
            {"name": "err_hi", "unit": "dimensionless"},
            {"name": "used_fiducial", "unit": "flag",
             "note": "1 if the bin is used in the paper's fiducial lensing "
                     "model (R > 0.5 h^-1 Mpc)"},
        ],
        "extraction": "exact vector-coordinate recovery from the PDF content "
                      "stream; NOT pixel digitisation.  Axis transform fitted "
                      "on four labelled major ticks per axis, log-linear "
                      "residual < 1e-6 dex.  Gates: error bars symmetric in "
                      "linear g to <5e-3; bin spacing matches the paper's "
                      "declared ten log bins over 0.2-3.5 h^-1 Mpc.",
        "axis_calibration": {
            "x_log10R_per_pt": cx[0], "x_intercept": cx[1],
            "y_log10g_per_pt": cy[0], "y_intercept": cy[1]},
        "model_curves_also_present_in_figure": [m[0] for m in mods],
        "caveat": "This is ONE stacked profile over the whole sample.  It is "
                  "not per-cluster shear and carries no within-sample "
                  "potential-depth leverage on its own.",
    }
    with open(OUT + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    print(f"\n   wrote {os.path.basename(OUT)}  ({len(rows)} bins)")
    print(f"   {'R [h-1 Mpc]':>12s} {'g_+':>10s} {'+-1sig':>10s} {'S/N':>6s} "
          f"{'used':>5s}")
    for r in rows:
        s = r["g_plus"] / (0.5 * (r["err_lo"] + r["err_hi"]))
        print(f"   {r['R_hinvMpc']:12.4f} {r['g_plus']:10.6f} "
              f"{0.5 * (r['err_lo'] + r['err_hi']):10.6f} {s:6.2f} "
              f"{r['used_fiducial']:5d}")
    tot = np.sqrt(sum((r["g_plus"] / (0.5 * (r["err_lo"] + r["err_hi"]))) ** 2
                      for r in rows))
    print(f"\n   total S/N of the stacked profile (diagonal only): {tot:.1f}")


if __name__ == "__main__":
    main()
