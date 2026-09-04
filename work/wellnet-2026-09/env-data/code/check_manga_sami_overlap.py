"""Is a direct MaNGA/SAMI cross-calibration possible from the data itself?

The report says the two pair tables must not be pooled without a
cross-calibration of their stellar-mass and size scales.  That is only
actionable if some galaxies appear in both surveys.  This measures how many do.
"""
import os

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

LANE = (r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration"
        r"\work\wellnet-2026-09\env-data")

m = pd.read_csv(os.path.join(LANE, "clean", "manga_env_master.csv"), low_memory=False)
s = pd.read_csv(os.path.join(LANE, "raw", "sami", "sami_dr3_master_galaxy_inventory.tsv"),
                sep="\t", low_memory=False)

sra = pd.to_numeric(s.RA_OBJ, errors="coerce").to_numpy()
sde = pd.to_numeric(s.DEC_OBJ, errors="coerce").to_numpy()
ok = np.isfinite(sra) & np.isfinite(sde)
print("SAMI rows with usable coordinates: %d of %d" % (int(ok.sum()), len(s)))
mc = SkyCoord(m.objra.to_numpy() * u.deg, m.objdec.to_numpy() * u.deg)
sc = SkyCoord(sra[ok] * u.deg, sde[ok] * u.deg)

idx, d2d, _ = mc.match_to_catalog_sky(sc)
for tol in (3.0, 10.0, 60.0):
    print("MaNGA galaxies with a SAMI counterpart within %5.1f arcsec: %d"
          % (tol, int((d2d.arcsec < tol).sum())))

print("\nsky footprints")
print("  MaNGA  RA %.1f to %.1f   Dec %.1f to %.1f"
      % (m.objra.min(), m.objra.max(), m.objdec.min(), m.objdec.max()))
print("  SAMI   RA %.1f to %.1f   Dec %.1f to %.1f"
      % (np.nanmin(sc.ra.deg), np.nanmax(sc.ra.deg),
         np.nanmin(sc.dec.deg), np.nanmax(sc.dec.deg)))
print("  MaNGA declination below 0 deg: %d of %d" % (int((m.objdec < 0).sum()), len(m)))
print("  SAMI  declination below 0 deg: %d of %d"
      % (int((sc.dec.deg < 0).sum()), len(sc)))

n3 = int((d2d.arcsec < 3.0).sum())
print("\nVERDICT: %s"
      % ("a direct cross-calibration IS possible on %d shared galaxies" % n3
         if n3 >= 20 else
         "NO direct cross-calibration is possible from these two surveys alone "
         "(%d shared galaxies). Any pooling of the MaNGA and SAMI pair tables "
         "would have to be calibrated through a third catalogue that overlaps "
         "both, not from the surveys themselves." % n3))
