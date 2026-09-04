"""End-to-end validation of the MaNGA -> Tempel environment cross-match.

Takes clusters whose velocity dispersion is known independently from the
literature, finds the MaNGA galaxies projected near them at matching redshift,
and checks that the pipeline assigns them a host with the right sigma_v and,
where applicable, the right X-ray identification.  A cross-match bug -- wrong
coordinate convention, wrong redshift window, an off-by-one in the group join --
would show up here as a nonsense dispersion or an empty host.
"""
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

MASTER = (r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration"
          r"\work\wellnet-2026-09\env-data\clean\manga_env_master.csv")

# name: (RA deg, Dec deg, z, literature sigma_v km/s)
KNOWN = {
    "Coma / A1656":    (194.9531, 27.9807, 0.0231, 1000),
    "A2199":           (247.1593, 39.5515, 0.0303, 770),
    "A2151 Hercules":  (241.1465, 17.7228, 0.0367, 780),
    "A1367":           (176.1900, 19.7000, 0.0216, 780),
    "A2147":           (240.5708, 15.9764, 0.0353, 820),
}


def main():
    d = pd.read_csv(MASTER, low_memory=False)
    c = SkyCoord(d.objra.to_numpy() * u.deg, d.objdec.to_numpy() * u.deg)
    print("%-17s %5s %10s %8s %8s   %s"
          % ("cluster", "N", "sigma_v", "(lit)", "Ngal", "X-ray host assigned"))
    for name, (ra, de, z, lit) in KNOWN.items():
        sep = c.separation(SkyCoord(ra * u.deg, de * u.deg)).deg
        m = (sep < 1.5) & (np.abs(d.z - z) < 0.006)
        n = int(m.sum())
        if n == 0:
            print("%-17s %5d   -- no MaNGA galaxies in the search box --" % (name, n))
            continue
        sv, ng = d.t14_grp_sigma_v[m], d.t14_Ngal[m]
        xr = sorted({x for x in d.t14_mcxc_name[m].astype(str)
                     if x and x not in ("nan", "")})
        print("%-17s %5d %10s %8d %8s   %s"
              % (name, n,
                 "%.0f" % np.nanmedian(sv) if np.isfinite(sv).any() else "n/a",
                 lit,
                 "%.0f" % np.nanmedian(ng) if np.isfinite(ng).any() else "n/a",
                 ", ".join(xr[:3]) if xr else "(none)"))


if __name__ == "__main__":
    main()
