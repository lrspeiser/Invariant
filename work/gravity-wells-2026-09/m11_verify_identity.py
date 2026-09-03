"""
AUDIT: were the X-COP cluster identities matched correctly?

Two results now rest on that matching -- the merger-bias null (h12) and the
temperature test (m10). Both paired bench objects to cluster names by the RANK
of each profile's outermost radius. That is only valid if the profiles all
extend to the same multiple of R500, which was assumed and never checked.

And there is a much better route available. The bench's `extent` values look
like R500 in kpc, which if true means the identities can be matched BY VALUE
instead of by rank -- exact, not inferred.
"""
import os
import re
import numpy as np
from astropy.io import fits
from invariant_bench import Bench, KPC

XR = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
      "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/")
SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
BAR = "=" * 78

tex = open(SCR + "xcop_T/XCOP_thermo.tex", encoding="utf-8").read()
i = tex.index("Basic properties of the X-COP sample")
blk = tex[i:i + 4000]
TAB = {}
for line in blk.split("\n"):
    if "&" not in line or line.strip().startswith("%") or "\\hline" in line:
        continue
    cells = [c.strip() for c in line.split("&")]
    if len(cells) < 6:
        continue
    nm = cells[0].strip()
    if not re.match(r"^[A-Za-z]+\d", nm):
        continue
    def val(s):
        m = re.search(r"([\d.]+)", s.replace("$", ""))
        return float(m.group(1)) if m else float("nan")
    TAB[nm] = dict(z=val(cells[1]), M500=val(cells[3]), R500=val(cells[4]))
print(f"parsed {len(TAB)} clusters from Table 1")
for k, v in sorted(TAB.items()):
    print(f"   {k:<10} z={v['z']:.4f}  M500={v['M500']:.2f}e14  R500={v['R500']:.0f} kpc")

b = Bench(verbose=False)
xc = b.d["xcop"]
ext = np.asarray(xc.extent, float) * np.ones(len(xc))
uq = sorted(np.unique(ext) / KPC)
print(f"\nbench extents (kpc): {[f'{v:.0f}' for v in uq]}")

print("\n" + BAR + "\nDoes `extent` equal R500?\n" + BAR)
exact = {}
for v in uq:
    hit = [k for k, t in TAB.items() if abs(t["R500"] - v) < 2.0]
    exact[v] = hit[0] if len(hit) == 1 else None
    print(f"   extent {v:>7.1f}  ->  {hit if hit else 'NO MATCH'}")
nmatch = sum(1 for v in exact.values() if v)
print(f"\n   {nmatch} of {len(uq)} bench extents match an R500 to within 2 kpc")

print("\n" + BAR + "\nWas the rank pairing correct?\n" + BAR)
RM = {}
for d in sorted(TAB):
    f = os.path.join(XR, d, f"{d}_density_L1.fits")
    if os.path.exists(f):
        with fits.open(f) as h:
            RM[d] = float(np.nanmax(np.asarray(h[1].data["R_OUT"], float)))
rank_by_profile = [k for k, _ in sorted(RM.items(), key=lambda t: t[1])]
rank_by_r500 = [k for k, _ in sorted(TAB.items(), key=lambda t: t[1]["R500"])]
print(f"   profile-end order : {rank_by_profile}")
print(f"   R500 order        : {rank_by_r500}")
print(f"\n   IDENTICAL? {rank_by_profile == rank_by_r500}")
if rank_by_profile != rank_by_r500:
    print("\n   THEY DIFFER. The rank pairing used in h12 and m10 assigned the")
    print("   wrong name to some clusters. Positions where they disagree:")
    for j, (a, c) in enumerate(zip(rank_by_profile, rank_by_r500)):
        if a != c:
            print(f"      slot {j}: rank pairing said {a}, correct is {c}")
    print(f"\n   profile end / R500 ratio per cluster (assumed constant, is not):")
    for k in sorted(RM):
        print(f"      {k:<10}{RM[k]:>9.1f} / {TAB[k]['R500']:>6.0f} = "
              f"{RM[k]/TAB[k]['R500']:.3f}")

print("\n" + BAR + "\nCORRECT identities, matched by R500 value\n" + BAR)
print(f"   {'extent kpc':>12}{'cluster':>12}{'M500':>9}{'kT500':>9}")
import math
OM, OL = 0.3, 0.7
FINAL = {}
for v in uq:
    nm = exact[v]
    if nm:
        t = TAB[nm]
        Ez = math.sqrt(OM * (1 + t["z"]) ** 3 + OL)
        T500 = 8.85 * (t["M500"] * 1e14 / 1e15) ** (2 / 3) * Ez ** (2 / 3)
        FINAL[v] = dict(name=nm, **t, T500=T500)
        print(f"   {v:>12.1f}{nm:>12}{t['M500']:>9.2f}{T500:>9.2f}")
    else:
        print(f"   {v:>12.1f}{'UNMATCHED':>12}")
import json
json.dump({f"{k:.4f}": v for k, v in FINAL.items()},
          open(SCR + "xcop_identity.json", "w", encoding="utf-8"), indent=1)
print(f"\n   wrote xcop_identity.json with {len(FINAL)} verified identities")
