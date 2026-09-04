"""JOB2 satellites lane: non-VizieR acquisitions (SAGA DR3 official release, LVDB)."""
import os
import sys

HERE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites"
sys.path.insert(0, HERE)
from _manifest import write_manifest, http_get  # noqa: E402

JOBS = [
    # --- SAGA DR3 (Mao et al. 2024, ApJ 976, 117) official release, MRT format ---
    ("sat_saga_dr3_tableC1_hosts.mrt",
     "https://sagasurvey.org/data/saga-dr3-tableC1.txt",
     "SAGA DR3 Table C1 -- 101 MW-mass host galaxies"),
    ("sat_saga_dr3_tableC2_bkgz.mrt",
     "https://sagasurvey.org/data/saga-dr3-tableC2.txt",
     "SAGA DR3 Table C2 -- background galaxy redshifts"),
    ("sat_saga_dr3_tableC3_satellites.mrt",
     "https://sagasurvey.org/data/saga-dr3-tableC3.txt",
     "SAGA DR3 Table C3 -- 378 confirmed satellites in 101 systems"),
    ("sat_saga_dr3_tableC4_candidates.mrt",
     "https://sagasurvey.org/data/saga-dr3-tableC4.txt",
     "SAGA DR3 Table C4 -- likely satellite candidates lacking reliable redshifts"),
    # --- SAGA DR2 official CSVs (cross-check against VizieR J/ApJ/907/85) ---
    ("sat_saga_dr2_hosts_official.csv",
     "https://sagasurvey.org/data/saga_stage2_hosts.csv",
     "SAGA Stage II (DR2) host galaxies, official CSV"),
    ("sat_saga_dr2_sats_official.csv",
     "https://sagasurvey.org/data/saga_stage2_sats.csv",
     "SAGA Stage II (DR2) satellites, official CSV"),
    # --- Local Volume Database v1.1.1 (Pace 2024/2025, OJAp 8, 142) ---
    ("sat_lvdb_v1.1.1_comb_all.csv",
     "https://github.com/apace7/local_volume_database/releases/download/v1.1.1/comb_all.csv",
     "LVDB v1.1.1 combined catalogue (dwarfs + star clusters), CSV"),
    ("sat_lvdb_v1.1.1_comb_all.ecsv",
     "https://github.com/apace7/local_volume_database/releases/download/v1.1.1/comb_all.ecsv",
     "LVDB v1.1.1 combined catalogue, ECSV (carries per-column units in header)"),
    ("sat_lvdb_v1.1.1_pm_overview.csv",
     "https://github.com/apace7/local_volume_database/releases/download/v1.1.1/pm_overview.csv",
     "LVDB v1.1.1 proper-motion overview: per-system systemic PM with literature provenance"),
]


def count_rows(path):
    """Row count: for CSV, data rows after header; for MRT, rows after the final rule line."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    if path.endswith(".csv") or path.endswith(".ecsv"):
        body = [l for l in lines if l.strip() and not l.startswith("#")]
        return max(len(body) - 1, 0), (body[0].split(",") if body else [])
    # MRT: header block ends with a line of dashes; data follows the LAST rule line
    rule_idx = [i for i, l in enumerate(lines) if l.strip().startswith("---") and len(l.strip()) > 20]
    if rule_idx:
        start = rule_idx[-1] + 1
        return len([l for l in lines[start:] if l.strip()]), []
    return len([l for l in lines if l.strip()]), []


def mrt_columns(path):
    """Parse the Byte-by-byte description of an MRT file -> [{name, unit, explanation}]."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    cols, inbbb = [], False
    for l in lines:
        if l.strip().lower().startswith("byte-by-byte description"):
            inbbb = True
            continue
        if inbbb:
            if l.strip().startswith("---"):
                if cols:
                    break
                continue
            parts = l.split()
            if len(parts) >= 4 and ("-" in parts[0] or parts[0].replace(".", "").isdigit()):
                # Bytes Format Units Label Explanations
                unit = parts[2]
                label = parts[3]
                expl = " ".join(parts[4:])
                cols.append({"name": label, "unit": "" if unit == "---" else unit,
                             "explanation": expl[:160]})
    return cols


def main():
    for name, url, note in JOBS:
        dest = os.path.join(HERE, name)
        try:
            http_get(url, dest)
            n, hdr = count_rows(dest)
            if name.endswith(".mrt"):
                cols = mrt_columns(dest)
            elif name.endswith(".ecsv"):
                cols = [{"name": c, "unit": ""} for c in hdr]
            else:
                cols = [{"name": c, "unit": ""} for c in hdr]
            write_manifest(
                dest, url, query="GET " + url,
                columns=cols or [{"name": "(see file header)", "unit": ""}],
                row_count=n, note=note,
                measurement_or_model=(
                    "MIXED -- observed positions/photometry/redshifts are MEASUREMENTS; "
                    "any halo/virial/dynamical mass column is a MODEL. Per-column labels "
                    "in _SECTION_SATELLITES.md"),
                extra={"acquisition_job": "JOB2 streams-satellites"},
            )
        except Exception as e:
            print("FAIL %-42s %s: %s" % (name, type(e).__name__, e))


if __name__ == "__main__":
    main()
