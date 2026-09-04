"""Acquire the Bolocam 140 GHz Sunyaev-Zel'dovich maps for the six HFF clusters.

Source: BOLOCAM Galaxy Cluster Archive at IRSA/IPAC, DOI 10.26131/IRSA562,
reference Sayers et al. 2013, ApJ 768, 177.

The SZ decrement image is a RAW OBSERVABLE (Compton-y, i.e. the line-of-sight
integral of electron pressure).  The gNFW_fit_map shipped in the same tarball is
a MODEL and is labelled as such in the manifest.
"""
import os
import subprocess
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import write_manifest, LANE, sha256  # noqa: E402

BASE = "https://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/bolocam/"
OUT = os.path.join(LANE, "gas", "bolocam_sz")

# archive filename -> (cluster label, archive RA, archive Dec)
TARGETS = {
    "ABELL_2744.tgz":          ("Abell 2744", "00:14:15.96", "-30:23:31.13"),
    "ABELL_0370.tgz":          ("Abell 370", "02:39:52.80", "-01:34:35.99"),
    "MACS_J0416.1-2403.tgz":   ("MACS J0416.1-2403", "04:16:08.38", "-24:04:20.78"),
    "MACS_J0717.5+3745.tgz":   ("MACS J0717.5+3745", "07:17:31.40", "+37:45:23.99"),
    "MACS_J1149.6+2223.tgz":   ("MACS J1149.5+2223", "11:49:35.40", "+22:24:05.00"),
    "ABELL_S1063.tgz":         ("Abell S1063 = RXC J2248.7-4431", "22:48:43.95", "-44:31:51.31"),
}


def curl(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    p = subprocess.run(["curl", "-sSL", "--max-time", "600", "-o", dest, url],
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("curl failed for %s: %s" % (url, p.stderr[:300]))
    return os.path.getsize(dest)


def main():
    for fn, (cluster, ra, dec) in TARGETS.items():
        url = BASE + fn
        dest = os.path.join(OUT, fn)
        n = curl(url, dest)
        # Enumerate members so the manifest records what is actually inside.
        try:
            with tarfile.open(dest) as tf:
                members = [{"name": m.name, "bytes": m.size}
                           for m in tf.getmembers() if m.isfile()]
        except Exception as e:
            print("TARBALL UNREADABLE", fn, repr(e))
            continue
        names = [m["name"] for m in members]
        write_manifest(
            dest, url,
            note=(cluster + ". Bolocam 140 GHz (2.1 mm) Sunyaev-Zel'dovich observation from the "
                  "BOXSZ sample (Sayers et al. 2013, ApJ 768, 177; IRSA DOI 10.26131/IRSA562). "
                  "Archive position " + ra + " " + dec + ". "
                  "The UNFILTERED and FILTERED SZ decrement images are RAW OBSERVABLES: the SZ "
                  "decrement is proportional to the Compton-y parameter, i.e. the line-of-sight "
                  "integral of the ELECTRON PRESSURE n_e * kT_e. This is an independent thermal "
                  "measurement of the intracluster gas that does not require a deprojection or a "
                  "hydrostatic assumption to interpret. The tarball ALSO contains a gnfw_fit_map, "
                  "which is a generalised-NFW PARAMETRIC FIT and is a MODEL, not an observation; it "
                  "must not be used as data. Also included: noise realisations, the pipeline signal "
                  "transfer function (needed to interpret the filtered map correctly), and bootstrap "
                  "Monte Carlo output from the gNFW fit. "
                  "CAVEAT: Bolocam's PSF is 58 arcsec FWHM, so this constrains the gas on scales of "
                  "roughly 0.1-3.5 R500 and carries no core information. "
                  "CAVEAT: a filtered map has had large-scale modes removed by the reduction "
                  "pipeline; use the transfer function, do not treat the filtered map as sky truth."),
            extraction="Verbatim upstream gzipped tarball, byte-for-byte unmodified. Not unpacked in place.",
            row_count=None, column_count=None, columns=None,
            extra={"cluster": cluster,
                   "product": "sz_pressure_observable",
                   "instrument": "Bolocam at the Caltech Submillimeter Observatory, 140 GHz",
                   "psf_fwhm_arcsec": 58,
                   "is_raw_observable": True,
                   "presupposes_dark_matter": False,
                   "derived_assumes_newtonian_hse": False,
                   "deprojected": False,
                   "core_excised": False,
                   "archive_position_ra_hms": ra,
                   "archive_position_dec_dms": dec,
                   "tar_members": members,
                   "n_tar_members": len(members),
                   "contains_model_product": any("gnfw" in x.lower() for x in names),
                   "model_product_files": [x for x in names if "gnfw" in x.lower()],
                   "irsa_doi": "10.26131/IRSA562"})
        print("%-28s %9d bytes  %2d members  %s" % (fn, n, len(members), cluster))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
