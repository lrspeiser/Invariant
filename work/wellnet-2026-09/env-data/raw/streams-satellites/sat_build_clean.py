"""JOB2 satellites lane: cleaned TSVs + a host-disk-orientation cross-reference.

Nothing here invents a measurement. Two operations only:
  1. fixed-width MRT  ->  TSV  (faithful, column-for-column)
  2. join of already-acquired tables on host identifier / name
The raw upstream files are untouched.
"""
import csv
import os
import re
import sys

HERE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites"
sys.path.insert(0, HERE)
from _manifest import write_manifest  # noqa: E402


# ---------------------------------------------------------------- MRT -> TSV
def read_mrt(path):
    """Parse an AAS MRT: returns (specs, rows) where specs = [(lo, hi, unit, label, expl)]."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    bstart = next(i for i, l in enumerate(lines)
                  if l.strip().lower().startswith("byte-by-byte description"))
    rules = [i for i, l in enumerate(lines) if re.fullmatch(r'-{20,}', l.strip())]
    after = [i for i in rules if i > bstart]
    # layout: <rule> "Bytes Format Units Label Explanations" <rule> <byte specs> <rule> ...
    assert len(after) >= 3, "unexpected MRT rule layout in %s (%d rules)" % (path, len(after))
    specs = []
    for l in lines[after[1] + 1:after[2]]:
        m = re.match(r'\s*(\d+)\s*-\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)$', l)
        if m:
            specs.append((int(m.group(1)), int(m.group(2)), m.group(4),
                          m.group(5), m.group(6).strip()))
            continue
        m = re.match(r'\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)$', l)
        if m:
            p = int(m.group(1))
            specs.append((p, p, m.group(3), m.group(4), m.group(5).strip()))
    data_start = after[-1] + 1
    rows = []
    for l in lines[data_start:]:
        if not l.strip():
            continue
        rows.append([l[lo - 1:hi].strip() for lo, hi, _, _, _ in specs])
    return specs, rows


def mrt_to_tsv(src, out, note, mm, url):
    specs, rows = read_mrt(os.path.join(HERE, src))
    dest = os.path.join(HERE, out)
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(s[3] for s in specs) + "\n")
        fh.write("\t".join("" if s[2] == "---" else s[2] for s in specs) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    write_manifest(dest, url,
                   query="fixed-width AAS MRT %s parsed on its published byte ranges" % src,
                   columns=[{"name": s[3], "unit": "" if s[2] == "---" else s[2],
                             "description": s[4][:200]} for s in specs],
                   row_count=len(rows), note=note, measurement_or_model=mm,
                   source_file_within_archive=src,
                   extraction={"method": "byte-range parse of the MRT Byte-by-byte block",
                               "n_columns": len(specs), "n_rows": len(rows),
                               "derived_from_local_file": src},
                   extra={"acquisition_job": "JOB2 streams-satellites"})
    print("MRT->TSV %-46s rows=%d cols=%d" % (out, len(rows), len(specs)))
    return len(rows)


# ------------------------------------------------- host orientation crossref
def norm(s):
    return re.sub(r'[^A-Z0-9]', '', (s or '').upper())


def name_keys(s):
    """All normalised spellings of a galaxy name, incl. NGC/IC/UGC zero-padding."""
    k = norm(s)
    out = [k]
    m = re.fullmatch(r'(NGC|IC|UGC|ESO|PGC)0*(\d+)([A-Z]*)', k)
    if m:
        pre, num, suf = m.group(1), int(m.group(2)), m.group(3)
        for w in (3, 4, 5):
            out.append('%s%0*d%s' % (pre, w, num, suf))
        out.append('%s%d%s' % (pre, num, suf))
    return out


def lookup(d, s):
    for k in name_keys(s):
        if k in d:
            return d[k]
    return None


def load_vizier_tsv(name):
    lines = open(os.path.join(HERE, name), encoding="utf-8", errors="replace").read().splitlines()
    h = next(i for i, l in enumerate(lines) if l and not l.startswith("#") and l.strip())
    cols = lines[h].split("\t")
    rows = [l.split("\t") for l in lines[h + 3:] if l.strip() and not l.startswith("#")]
    return cols, rows


def build_crossref():
    # --- UNGC: name -> (b/a, i, W50, vAmp, Bmag, Kmag, Dist)
    ucols, urows = load_vizier_tsv("sat_ungc_karachentsev2013_catalog.tsv")
    ui = {c: j for j, c in enumerate(ucols)}
    ungc = {}
    for r in urows:
        for k in ("Name", "SimbadName", "NEDname"):
            if k in ui and len(r) > ui[k]:
                for nk in name_keys(r[ui[k]]):
                    ungc.setdefault(nk, r)

    # --- HyperLEDA for SAGA hosts: PGC -> (logR25, PA)
    hcols, hrows = load_vizier_tsv("sat_hyperleda_saga_hosts_PA.tsv")
    hi = {c: j for j, c in enumerate(hcols)}
    leda = {r[hi["PGC"]].strip(): r for r in hrows if len(r) > hi["PGC"]}

    # --- SAGA DR3 hosts (cleaned TSV produced above)
    srows = list(csv.reader(open(os.path.join(HERE, "sat_saga_dr3_tableC1_hosts.tsv"),
                                 encoding="utf-8"), delimiter="\t"))
    shdr = srows[0]
    si = {c: j for j, c in enumerate(shdr)}
    saga = srows[2:]

    # --- HyperLEDA for ELVES hosts: name -> (logR25, PA)
    hecols, herows = load_vizier_tsv("sat_hyperleda_elves_hosts_PA.tsv")
    hei = {c: j for j, c in enumerate(hecols)}
    ledaname = {}
    for r in herows:
        if len(r) > hei["elves_host"]:
            for nk in name_keys(r[hei["elves_host"]]):
                ledaname.setdefault(nk, r)

    # --- ELVES hosts
    erows = [l.split("\t") for l in
             open(os.path.join(HERE, "sat_elves_table1_hosts.tsv"),
                  encoding="utf-8").read().splitlines()[2:]]
    ALIAS = {"M31": "MESSIER031", "M81": "MESSIER081", "NGC5457": "MESSIER101",
             "M104": "NGC4594", "CENA": "NGC5128", "MW": "MILKYWAY"}

    out = os.path.join(HERE, "sat_host_orientation_crossref.tsv")
    cols = ["survey", "host_name", "host_id", "pgc", "n_satellites_in_survey",
            "ba_survey", "PA_survey_deg", "sersic_survey",
            "logR25_hyperleda", "ba_hyperleda", "PA_hyperleda_deg",
            "ba_ungc", "inclination_ungc_deg", "W50_ungc_kms", "vAmp_ungc_kms",
            "Kmag_ungc", "Bmag_ungc", "dist_Mpc",
            "has_orientation", "has_inplane_rotation"]
    units = ["", "", "", "", "", "", "deg", "", "", "", "deg", "", "deg",
             "km/s", "km/s", "mag", "mag", "Mpc", "", ""]
    n_sa_or = n_sa_rot = n_el_or = n_el_rot = 0
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(cols) + "\n")
        fh.write("\t".join(units) + "\n")
        for r in saga:
            pgc = r[si["PGC"]].strip()
            l = leda.get(pgc)
            logr = l[hi["logR25"]].strip() if l else ""
            ba_l = "%.3f" % (10 ** -float(logr)) if logr else ""
            pa_l = l[hi["PA"]].strip() if l else ""
            if pa_l == "--":
                pa_l = ""
            u = lookup(ungc, r[si["Name"]])
            ba_s = r[si["ba"]].strip()
            pa_s = r[si["PA"]].strip()
            inc_u = u[ui["i"]].strip() if u else ""
            w50 = u[ui["W50"]].strip() if u else ""
            vamp = u[ui["vAmp"]].strip() if u else ""
            orient = "YES" if (ba_s and pa_s) else ("YES" if (ba_l and pa_l) else "NO")
            rot = "YES" if (w50 or vamp) else "NO"
            n_sa_or += orient == "YES"
            n_sa_rot += rot == "YES"
            fh.write("\t".join(["SAGA-DR3", r[si["Name"]].strip(), r[si["HOSTID"]].strip(),
                                pgc, r[si["nsat-GSPc"]].strip(), ba_s, pa_s,
                                r[si["Sersic"]].strip(), logr, ba_l, pa_l,
                                u[ui["b/a"]].strip() if u else "", inc_u, w50, vamp,
                                u[ui["Kmag"]].strip() if u else "",
                                u[ui["Bmag"]].strip() if u else "",
                                r[si["Dist"]].strip(), orient, rot]) + "\n")
        for r in erows:
            name = r[0].strip()
            u = lookup(ungc, ALIAS.get(name, name)) or lookup(ungc, name)
            le = lookup(ledaname, ALIAS.get(name, name)) or lookup(ledaname, name)
            ba_u = u[ui["b/a"]].strip() if u else ""
            inc_u = u[ui["i"]].strip() if u else ""
            w50 = u[ui["W50"]].strip() if u else ""
            vamp = u[ui["vAmp"]].strip() if u else ""
            logr_e = le[hei["logR25"]].strip() if le else ""
            ba_le = "%.3f" % (10 ** -float(logr_e)) if logr_e else ""
            pa_le = le[hei["PA"]].strip() if le else ""
            if pa_le == "--":      # HyperLEDA has no PA measurement for this galaxy
                pa_le = ""
            orient = ("YES" if ((ba_u or inc_u or ba_le) and pa_le)
                      else ("PARTIAL(no PA)" if (ba_u or inc_u) else "NO"))
            rot = "YES" if (w50 or vamp) else "NO"
            n_el_or += orient == "YES"
            n_el_rot += rot == "YES"
            fh.write("\t".join(["ELVES", name, name,
                                le[hei["PGC"]].strip() if le else "",
                                "", "", "", "", logr_e, ba_le, pa_le,
                                ba_u, inc_u, w50, vamp,
                                u[ui["Kmag"]].strip() if u else "",
                                u[ui["Bmag"]].strip() if u else "",
                                r[1].strip(), orient, rot]) + "\n")

    n = len(saga) + len(erows)
    write_manifest(out, "(derived: join of files already acquired in this lane)",
                   query="LEFT JOIN of sat_saga_dr3_tableC1_hosts.tsv (on PGC) with "
                         "sat_hyperleda_saga_hosts_PA.tsv, and of sat_elves_table1_hosts.tsv "
                         "(on galaxy name, with Messier aliases) with "
                         "sat_ungc_karachentsev2013_catalog.tsv. No values recomputed except "
                         "ba_hyperleda = 10**(-logR25), the catalogue's own definition.",
                   columns=[{"name": c, "unit": u} for c, u in zip(cols, units)],
                   row_count=n,
                   note="DERIVED JOIN TABLE. Per-host summary of what disk-orientation and "
                        "in-plane-rotation information exists, for the two surveys that pair "
                        "many satellites with one host. SAGA hosts with full orientation "
                        "(b/a AND PA): %d of %d. ELVES hosts with b/a AND PA "
                        "(PA from HyperLEDA): %d of %d. Hosts with an in-plane HI rotation "
                        "measure: SAGA %d, ELVES %d."
                        % (n_sa_or, len(saga), n_el_or, len(erows), n_sa_rot, n_el_rot),
                   measurement_or_model="MEASUREMENT (all joined columns are observed "
                                        "photometric shape or HI line width); the file "
                                        "itself is a DERIVED JOIN, not an upstream product.",
                   extra={"acquisition_job": "JOB2 streams-satellites",
                          "saga_hosts_with_ba_and_PA": n_sa_or,
                          "saga_hosts_total": len(saga),
                          "elves_hosts_with_full_orientation_ba_and_PA": n_el_or,
                          "elves_hosts_total": len(erows),
                          "saga_hosts_with_HI_rotation": n_sa_rot,
                          "elves_hosts_with_HI_rotation": n_el_rot})
    print("CROSSREF rows=%d | SAGA orientation %d/%d, HI-rot %d | ELVES orientation %d/%d, "
          "HI-rot %d" % (n, n_sa_or, len(saga), n_sa_rot, n_el_or, len(erows), n_el_rot))


MEAS_SAGA_HOST = (
    "MIXED. MEASUREMENT: RAdeg, DEdeg, HRV, Dist, DistMod, KsMag, rmag, gr, sb, ba, PA, "
    "Sersic, log(MHI), nz-saga, nz-total, nsat-* counts, sep-MW, sep-massive. "
    "MODEL: log(Mhalo) is a halo mass from the Lim et al. (2017) group catalogue -- it "
    "PRESUPPOSES DARK MATTER and must not be treated as an observation. "
    "DERIVED-WITH-ASSUMPTIONS: log(M*) and log(sfr) use a stellar-population M/L and an "
    "IMF (no dark matter involved).")
MEAS_SAGA_SAT = (
    "MIXED. MEASUREMENT: RAdeg, DEdeg, Rhost (projected separation), rmag, gr, sb, ba, PA, "
    "Sersic, z, DVhost (line-of-sight velocity offset from the host), all emission-line "
    "fluxes and equivalent widths, NUVmag, log(MHI). "
    "DERIVED-WITH-ASSUMPTIONS: log(M*), log(sfr)*, quenched flag (stellar-population "
    "modelling; no dark matter). NO dark-matter-dependent column is present in this table.")

if __name__ == "__main__":
    mrt_to_tsv("sat_saga_dr3_tableC1_hosts.mrt", "sat_saga_dr3_tableC1_hosts.tsv",
               "SAGA DR3 (Mao+2024, ApJ 976,117) Table C1 host catalogue, MRT converted to "
               "TSV. Carries host axis ratio 'ba' and position angle 'PA' from DESI Legacy "
               "Imaging -- i.e. the HOST DISK ORIENTATION needed to place each satellite "
               "relative to the host disk plane.",
               MEAS_SAGA_HOST, "https://sagasurvey.org/data/saga-dr3-tableC1.txt")
    mrt_to_tsv("sat_saga_dr3_tableC3_satellites.mrt", "sat_saga_dr3_tableC3_satellites.tsv",
               "SAGA DR3 Table C3 confirmed-satellite catalogue, MRT converted to TSV. "
               "Join to the host table on HOSTID. Carries projected position (RAdeg, DEdeg, "
               "Rhost) and line-of-sight velocity offset DVhost for every satellite.",
               MEAS_SAGA_SAT, "https://sagasurvey.org/data/saga-dr3-tableC3.txt")
    mrt_to_tsv("sat_saga_dr3_tableC4_candidates.mrt", "sat_saga_dr3_tableC4_candidates.tsv",
               "SAGA DR3 Table C4 satellite candidates without reliable redshifts.",
               MEAS_SAGA_SAT, "https://sagasurvey.org/data/saga-dr3-tableC4.txt")
    build_crossref()
