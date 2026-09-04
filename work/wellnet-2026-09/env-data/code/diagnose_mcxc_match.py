"""Why does Coma get no X-ray host flag?

The MCXC flag in build_manga_env.py matches the TEMPEL GROUP CENTRE to an MCXC
cluster within 10 arcmin.  For a very rich friends-of-friends group the
luminosity-weighted centre can sit far from the X-ray peak, so the flag has
false negatives on exactly the richest systems.  This measures the offset.
"""
import os
import sys

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vizier_tsv import read_vizier_tsv, num

LANE = (r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration"
        r"\work\wellnet-2026-09\env-data")
d = pd.read_csv(os.path.join(LANE, "clean", "manga_env_master.csv"), low_memory=False)
m = read_vizier_tsv(os.path.join(LANE, "raw", "groups", "mcxc_piffaretti2011.tsv"))
mc = SkyCoord(m["RAJ2000"].to_numpy(), m["DEJ2000"].to_numpy(),
              unit=(u.hourangle, u.deg))

coma = SkyCoord(194.9531 * u.deg, 27.9807 * u.deg)
i = np.argmin(mc.separation(coma).arcmin)
print("nearest MCXC to the Coma centre: %s (%s), sep %.2f arcmin, z=%s, L500=%s"
      % (m["MCXC"][i], m["OName"][i].strip(), mc[i].separation(coma).arcmin,
         m["z"][i], m["L500"][i]))

g = d[(np.abs(d.z - 0.0231) < 0.006)
      & (SkyCoord(d.objra.to_numpy() * u.deg,
                  d.objdec.to_numpy() * u.deg).separation(coma).deg < 1.5)]
gid = g.t14_GroupID.mode()
print("\nMaNGA galaxies near Coma: %d, dominant Tempel GroupID %s (Ngal %s)"
      % (len(g), gid.iloc[0] if len(gid) else None, g.t14_Ngal.median()))
gc = SkyCoord(g.t14_grp_RA.median() * u.deg, g.t14_grp_DE.median() * u.deg)
print("Tempel group centre: RA %.4f Dec %.4f" % (gc.ra.deg, gc.dec.deg))
print("offset of that centre from the MCXC X-ray position: %.1f arcmin"
      % gc.separation(mc[i]).arcmin)
print("\n=> the 10 arcmin group-centre match cannot reach it. The X-ray flag as "
      "built has FALSE NEGATIVES on the richest FoF systems.")

# A galaxy-centred alternative: galaxy within 2 Mpc of an X-ray peak at matching z
from astropy.cosmology import FlatLambdaCDM
cos = FlatLambdaCDM(H0=70, Om0=0.3)
gal = SkyCoord(d.objra.to_numpy() * u.deg, d.objdec.to_numpy() * u.deg)
idx, d2d, _ = gal.match_to_catalog_sky(mc)
zc = num(m, "z").to_numpy()[idx]
dA = np.asarray(cos.angular_diameter_distance(np.clip(d.z.to_numpy(), 1e-5, None))
                .to(u.Mpc))
rproj = d2d.radian * dA
hit = (np.abs(d.z.to_numpy() - zc) < 0.01) & (rproj < 2.0)
print("\ngalaxy-centred X-ray flag (within 2 Mpc projected of an MCXC peak and "
      "|dz|<0.01): %d galaxies, versus %d from the group-centre flag"
      % (int(hit.sum()), int(d.t14_mcxc_L500_1e44.notna().sum())))
sub = d[hit]
print("   distinct MCXC clusters hit: %d" % len(set(m["MCXC"].to_numpy()[idx][hit])))
print("   Coma among them: %s"
      % ("yes" if any("1259" in x for x in m["MCXC"].to_numpy()[idx][hit]) else "no"))
