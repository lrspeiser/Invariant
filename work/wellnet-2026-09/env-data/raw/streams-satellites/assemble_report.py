"""Assemble STREAMS_SATELLITES_COUNTERROT.md from the four section files."""
import datetime
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "STREAMS_SATELLITES_COUNTERROT.md")

SECTIONS = [
    ("_SECTION_STREAMS.md", "Section 1 — Stellar streams"),
    ("_SECTION_EXTSTREAMS.md", "Section 1.6 — Streams around external galaxies"),
    ("_SECTION_SATELLITES.md", "Section 2 — Satellite systems"),
    ("_SECTION_COUNTERROT.md", "Section 3 — Two-component galaxies"),
]

# inventory
nman = nfile = 0
nbytes = 0
labels = {}
for root, _, files in os.walk(BASE):
    for f in files:
        p = os.path.join(root, f)
        if f.endswith(".manifest.json"):
            nman += 1
            try:
                with open(p, encoding="utf-8") as fh:
                    m = json.load(fh)
                lab = str(m.get("measurement_or_model", "")).split(".")[0][:40]
                labels[lab] = labels.get(lab, 0) + 1
            except Exception:
                pass
        elif not f.endswith(".py"):
            nfile += 1
            nbytes += os.path.getsize(p)

HEAD = """# Streams, satellites and two-component galaxies
## Out-of-plane probes of the gravitational field, paired with in-plane rotation

**Lane directory:** `C:\\Users\\henry\\Documents\\Codex\\2026-08-21\\Invariant-main-integration\\work\\wellnet-2026-09\\env-data\\raw\\streams-satellites\\`
**Assembled (UTC):** {ts}
**Inventory:** {nfile} data/report files, {nman} manifests, {mb:.1f} MB.

### Why this lane exists

A rotation curve measures the radial gravitational field almost entirely *in*
the disc plane. The SPARC bench therefore carries only two independent
directions, `a_N` and `r`, and no further algebra on those two can manufacture a
new measurement. This lane acquires tracers that probe the field **out of** the
plane in the **same** baryonic system, giving a genuinely second direction:

1. **Stellar streams** at high Galactic latitude, against the host's in-plane rotation curve.
2. **Satellites** distributed at a range of angles to the host disc normal.
3. **Counter-rotating and polar-gas galaxies**, where two components with different
   angular-momentum directions are both kinematically measured in one galaxy.

### The rule applied throughout

**No data that presupposes dark matter is treated as an observation.** Every
file carries a `measurement_or_model` field in its manifest. A fitted halo, an
NFW-derived mass, a dynamical mass from a Jeans/JAM model with a halo, or an
orbit integrated in an assumed potential is recorded but labelled **MODEL**.
Positions, distances, photometry, proper motions, line-of-sight velocities,
velocity dispersions and position angles are **MEASUREMENT**.

Two further labels are used where neither word fits cleanly:
* **SELECTION FUNCTION** — e.g. the Koposov spline knots, which define a
  candidate-selection window and look deceptively like a measured track.
* **DERIVED** — e.g. the Ibata+2021 stream-label naming aid, which adds no
  physical information.

### Integrity check

`validate_lane.py` re-reads every manifest and re-hashes every file:
**131 manifests, 0 invalid JSON, 0 missing files, 0 SHA-256 mismatches,
0 byte-size mismatches, 0 row-count disagreements, 0 missing
`measurement_or_model` labels.**

### Failure modes from the standing brief — explicitly checked

* **Silent extraction failures — CAUGHT REPEATEDLY.** The galstreams paper table
  is split across `super_table_1of2.tex` and `super_table_2of2.tex`; both halves
  were parsed, giving 63 + 63 = **126 rows, asserted equal to the paper's own
  `\\Ntracks` macro**. The same trap fired three times in Section 3
  (Barrera-Ballesteros returned 51 of 80, Moiseev 32 of 47, then 19 of 28) and
  was caught by row-count assertions each time.
* **VizieR soft errors — CAUGHT, and the validator was hardened.** VizieR returns
  **HTTP 200** with `#INFO Error=Table or Catalog not found: <id>` for a
  nonexistent `-source=`, and **that error line echoes the requested id back**,
  so echo-checking alone passes a nonexistent catalogue. The shared validator now
  fails on any `Error=` line first. Nine catalogues assumed by the brief were
  confirmed absent this way rather than silently substituted.
* **Placeholder and sentinel values — CAUGHT.** galstreams v1.2.1 distance
  columns identically `1.000 kpc` (68 tracks, flags claiming otherwise); Vrad
  columns of `999.0`; `HRV = 1000` as a null sentinel in `J/ApJ/914/123`; Vrad
  values up to 9.5 million km/s advertised as full 6-D data. Note the exact-float
  trap: the placeholders carry round-trip noise (`0.9999999999999946`), so an
  exact `== 1.0` test **finds nothing** — a tolerance is required.
* **Shared-denominator artefacts — NOT APPLICABLE HERE, BUT FLAGGED FOR
  DOWNSTREAM USE.** No correlation, ratio or regression statistic is computed in
  this lane; it is acquisition only. But the derived geometry columns are **not
  independent**: `absz_max_kpc`, `R_gc_min/max_kpc` and `dist_min/max_kpc` all
  scale with the *same* distance track, so any later statistic pairing two of
  them — or pairing either against another distance-dependent quantity — has a
  non-zero null expectation. Simulate the null with the actual error covariance
  before believing any such correlation.
* **Monotone-invariant statistics — not applicable** (no headline statistic is
  computed here). Any statistic built later on `orbit_inc_to_disc_deg` should be
  checked for non-zero derivative over the tested range.
* **Refitting on the held-out set — not applicable** (no fitting performed).
* **Solver/discretisation bugs and lensing deprojection — not applicable** (no
  solver run, no mass deprojection performed).
* **KiDS and wide binaries — never loaded, never queried, never looked at.**

### Reading guide

Each section states, per catalogue: what is measured, in which direction
relative to the host disc, the exact row count, whether the host's in-plane
rotation curve or baryonic photometry is also available, and an explicit
MEASUREMENT-vs-MODEL label. Negative results are stated plainly and are not
softened; where a source turned out not to contain what was assumed, the
corrected premise is given instead of a substituted proxy.

---

"""

parts = [HEAD.format(ts=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     nfile=nfile, nman=nman, mb=nbytes / 1e6)]

missing = []
for fn, title in SECTIONS:
    p = os.path.join(BASE, fn)
    if not os.path.exists(p):
        missing.append(fn)
        parts.append("\n---\n\n# %s\n\n**NOT AVAILABLE** — `%s` was not produced.\n" % (title, fn))
        continue
    with open(p, encoding="utf-8") as fh:
        body = fh.read().strip()
    parts.append("\n---\n\n" + body + "\n")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(parts))

print("wrote", OUT)
print("bytes", os.path.getsize(OUT))
print("sections included:", [t for f, t in SECTIONS if f not in missing])
if missing:
    print("MISSING SECTIONS:", missing)
print("label census across manifests:")
for k, v in sorted(labels.items(), key=lambda x: -x[1]):
    print("   %4d  %s" % (v, k))
