"""JOB2 satellites lane: transcribe AASTeX deluxetable bodies from arXiv source to TSV.

Raw arXiv tarballs are kept unmodified; the extracted .tex live under _arxiv_extract/.
Row counts are asserted against the sample sizes stated in each paper.
"""
import os
import re
import sys

HERE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites"
sys.path.insert(0, HERE)
from _manifest import write_manifest  # noqa: E402

ACCENTS = {r'\"{o}': 'o', r'\"o': 'o', r"\'{e}": 'e', r"\'e": 'e', r'\"{u}': 'u',
           r'\~{n}': 'n', r'\c{c}': 'c', r'\^{o}': 'o'}


def detex(s):
    for k, v in ACCENTS.items():
        s = s.replace(k, v)
    s = re.sub(r'\\cite[a-z]*\*?\{[^}]*\}', '', s)
    s = re.sub(r'\\(colhead|tablenotemark|nodata|phantom)\b', ' ', s)
    s = s.replace(r'\pm', '+/-').replace('$', '').replace('~', ' ')
    s = re.sub(r'\\[a-zA-Z]+', '', s)          # residual macros
    s = s.replace('{', '').replace('}', '')
    s = s.replace('\\', '')
    return ' '.join(s.split()).strip()


def split_pm(cell):
    """Return (value, minus_err, plus_err) if the cell matches a known error form."""
    c = cell.replace(' ', '')
    m = re.fullmatch(r'(-?[\d.]+)_-([\d.]+)\^\+([\d.]+)', c)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.fullmatch(r'(-?[\d.]+)\+/-([\d.]+)', c)
    if m:
        return m.group(1), m.group(2), m.group(2)
    return None


def parse_deluxetable(path):
    """Return (colnames, unitrow, rows). Asserts exactly one data environment."""
    txt = open(path, encoding='utf-8', errors='replace').read()
    n_env = len(re.findall(r'\\begin\{(?:deluxetable\*?|longtable|table\*?)\}', txt))
    n_sd = len(re.findall(r'\\startdata', txt))
    assert n_sd == 1, "%s: expected 1 \\startdata, found %d (SPLIT-TABLE RISK)" % (path, n_sd)

    # brace-balanced extraction of \tablehead{...}
    hdr_txt = ''
    i = txt.find(r'\tablehead')
    if i >= 0:
        j = txt.find('{', i)
        depth, k = 0, j
        while k < len(txt):
            if txt[k] == '{':
                depth += 1
            elif txt[k] == '}':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        hdr_txt = txt[j + 1:k]
    hdr_lines = [l for l in hdr_txt.split(r'\\') if l.strip()]
    cols = [detex(c) for c in hdr_lines[0].split('&')] if hdr_lines else []
    units = [detex(c) for c in hdr_lines[1].split('&')] if len(hdr_lines) > 1 else [''] * len(cols)
    units += [''] * (len(cols) - len(units))

    body = re.search(r'\\startdata(.*?)\\enddata', txt, re.S).group(1)
    rows = []
    for raw in body.split('\\\\'):
        if not raw.strip() or raw.strip().startswith('%'):
            continue
        cells = [detex(c) for c in raw.split('&')]
        if not any(cells):
            continue
        rows.append(cells)
    return cols, units, rows, n_env


def emit(tex_path, out_name, expect_rows, note, mm, source_url, archive_member):
    cols, units, rows, n_env = parse_deluxetable(tex_path)
    ncell = max(len(r) for r in rows)
    if len(cols) < ncell:
        cols += ['col%d' % i for i in range(len(cols), ncell)]
        units += [''] * (ncell - len(units))

    # widen with split value/err columns where the form is recognised
    out_cols, out_units = [], []
    for c, u in zip(cols[:ncell], units[:ncell]):
        out_cols.append(c)
        out_units.append(u)
    extra_specs = []
    for j in range(ncell):
        hits = sum(1 for r in rows if j < len(r) and split_pm(r[j]))
        if hits >= 0.5 * len(rows):
            extra_specs.append(j)
    for j in extra_specs:
        base = cols[j] if j < len(cols) else 'col%d' % j
        u = units[j] if j < len(units) else ''
        out_cols += [base + '_value', base + '_err_minus', base + '_err_plus']
        out_units += [u, u, u]

    dest = os.path.join(HERE, out_name)
    with open(dest, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\t'.join(out_cols) + '\n')
        fh.write('\t'.join(out_units) + '\n')
        for r in rows:
            r = r + [''] * (ncell - len(r))
            line = list(r[:ncell])
            for j in extra_specs:
                s = split_pm(r[j])
                line += list(s) if s else ['', '', '']
            fh.write('\t'.join(line) + '\n')

    status = "OK " if len(rows) == expect_rows else "MISMATCH"
    print("%s %-46s rows=%-5d expected=%-5d ncols=%-3d envs=%d"
          % (status, out_name, len(rows), expect_rows, len(out_cols), n_env))
    write_manifest(
        dest, source_url,
        query="GET %s ; tar member %s ; AASTeX deluxetable body transcribed to TSV"
              % (source_url, archive_member),
        columns=[{"name": c, "unit": u} for c, u in zip(out_cols, out_units)],
        row_count=len(rows), note=note,
        source_file_within_archive=archive_member,
        measurement_or_model=mm,
        extraction={"method": "regex parse of \\startdata..\\enddata in a single "
                              "deluxetable environment",
                    "n_table_environments_in_file": n_env,
                    "n_startdata_in_file": 1,
                    "expected_row_count_from_paper": expect_rows,
                    "observed_row_count": len(rows),
                    "row_count_matches_paper": len(rows) == expect_rows,
                    "note": "value/err columns added where cells matched "
                            "'x+/-e' or 'x_-a^+b'; original cell text retained too"},
        extra={"acquisition_job": "JOB2 streams-satellites"},
    )
    return len(rows)


P = os.path.join(HERE, '_arxiv_extract', 'pace2022')
E = os.path.join(HERE, '_arxiv_extract', 'elves2022')
PU = "https://arxiv.org/e-print/2205.05699"
EU = "https://arxiv.org/e-print/2203.00014"

MEAS = ("MEASUREMENT -- observed sky positions, structural parameters, distances, "
        "magnitudes, line-of-sight velocities/dispersions and Gaia proper motions. "
        "No dark-matter halo is assumed to produce these numbers.")
MODEL = ("MODEL -- NOT AN OBSERVATION. Orbital pericentre/apocentre/eccentricity are "
         "obtained by integrating each dwarf's measured 6-D phase-space vector in an "
         "ASSUMED Milky Way (+LMC) dark-matter halo potential. These columns presuppose "
         "dark matter and must never be used as data. Retained for cross-checking only.")

if __name__ == "__main__":
    emit(os.path.join(P, 'table_overview.tex'), 'sat_pace2022_table1_dsph_properties.tsv',
         54, "Pace, Erkal & Li 2022 (ApJ 940,136) Table 1: dSph properties -- RA, Dec, "
             "r_h, ellipticity, PA, distance, M_V, v_los, sigma_los. ROW COUNT 54, not the "
             "52 quoted in the abstract: the abstract's 52 is the proper-motion sample "
             "(Table 2). Table 1 additionally lists Cet III and Vir I, which have no PM "
             "measurement. Verified by set difference Table1-Table2 = {Cet III, Vir I}.",
         MEAS, PU, 'table_overview.tex')
    emit(os.path.join(P, 'table_results.tex'), 'sat_pace2022_table2_systemic_pm.tsv',
         52, "Pace, Erkal & Li 2022 Table 2: Gaia EDR3 systemic proper motions "
             "(mu_alpha*, mu_delta) for 52 MW dSphs, faint + gold member samples",
         MEAS, PU, 'table_results.tex')
    emit(os.path.join(P, 'table_orbit.tex'), 'sat_pace2022_table3_orbits_MODEL.tsv',
         46, "Pace, Erkal & Li 2022 Table 3: dSph ORBITAL properties (r_peri, r_apo, ecc) "
             "integrated in an assumed MW+LMC halo potential. ROW COUNT 46, not 52: six "
             "dwarfs with no systemic line-of-sight velocity (Boo IV, Cen I, Cet II, "
             "Hor II, Pic I, Pic II) cannot be orbit-integrated and are absent. NB Table 3 "
             "abbreviates Pisces II as 'Pis II' where Tables 1-2 use 'Psc II' -- same object.",
         MODEL, PU, 'table_orbit.tex')
    emit(os.path.join(E, 'host_table.tex'), 'sat_elves_table1_hosts.tsv',
         31, "Carlsten et al. 2022 (ApJ 933,47) Table 1: the ELVES host galaxies -- "
             "distance, v_rec, M_Ks, M_V, B-V, log M*, survey coverage radius. "
             "ROW COUNT 31 host entries (paper: '31 such hosts', of which '30 surveyed'; "
             "NGC3621 has r_cover=0 i.e. unsurveyed, and the Milky Way is listed as a "
             "reference host with dist=0). "
             "NOTE: contains NO host inclination, position angle or axis ratio.",
         MEAS, EU, 'host_table.tex')
    emit(os.path.join(E, 'overview_table.tex'), 'sat_elves_overview_table.tsv',
         31, "Carlsten et al. 2022 ELVES overview table (satellite counts per host); "
             "31 rows, matching the 31-entry host list",
         MEAS, EU, 'overview_table.tex')
