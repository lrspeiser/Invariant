"""Apply the final lane corrections to REPORT.md.

Folds in: the velocity radial-coverage finding, the third VizieR trap variant,
the Braglia+2009 merged-table trap, the AS1063 and MACS J0717 catalogues found
outside CDS, the re-derived membership counts, and the strong-lensing integrity
re-fetch result.
"""
import io
import os

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(LANE, "REPORT.md")
s = io.open(P, encoding="utf-8").read()
orig = s
EM = "—"


def sub(old, new):
    global s
    if old not in s:
        raise SystemExit("PATCH ANCHOR NOT FOUND:\n" + repr(old[:180]))
    s = s.replace(old, new, 1)


# ---- 1. P7 line in the summary table ------------------------------------
sub("| 7. Member velocities | 7 of 7 | A2029 by far the richest |",
    "| 7. Member velocities | 7 of 7, radial coverage very uneven | "
    "A2029 gives 1215 members to 8.75 Mpc; MACS J1149 gives 151 to 0.65 Mpc |")

# ---- 2. Correct absence #9 and add the access-route finding -------------
sub("8. **Strong lensing for Abell 2029.** It is a relaxed low-z cool-core cluster,\n"
    "   not a lens.",
    "8. **Strong lensing for Abell 2029.** It is a relaxed low-z cool-core cluster,\n"
    "   not a lens.\n"
    "\n"
    "One apparent absence turned out to be an ACCESS-ROUTE problem rather than a real\n"
    "one, and it is worth separating from the list above. The wide-field spectroscopic\n"
    "catalogues for Abell S1063 and MACS J0717 are **not** missing; they are simply not\n"
    "reachable by the CDS route:\n"
    "\n"
    "- **Abell S1063**: the CLASH-VLT public release (Mercurio et al. 2021, A&A 656,\n"
    "  A147) is distributed from the project's own site, not CDS. VizieR\n"
    "  `J/A+A/656/A147` does not exist and silently serves the Cooper+2013 fallback;\n"
    "  cdsarc returns 404. Acquired: 3850 redshifts, replacing the 290-row NED cone.\n"
    "- **MACS J0717**: VizieR `J/ApJS/211/21` (Ebeling, Ma & Barrett 2014) *is* real and\n"
    "  *is* a spectroscopic catalogue. Acquired: 1266 rows for the J0717.5+3745 field,\n"
    "  asserted against the CDS ReadMe entry `table4.dat 70 1266`, replacing the NED\n"
    "  cone with a single homogeneous Keck/DEIMOS + LRIS + GMOS survey. The `+` in the\n"
    "  `MACS=` filter must be percent-encoded as `%2B` or the filter matches nothing.\n"
    "  Incidentally that parent table also holds 65 redshifts in the MACS J0416 field.\n"
    "\n"
    "Three access routes are simply not scriptable and produced false negatives\n"
    "elsewhere in this programme: `cdsarc.cds.unistra.fr/ftp/...` **data** files sit\n"
    "behind an Anubis bot-check returning a 4.4 KB HTML challenge instead of data (the\n"
    "ReadMe works with a browser user-agent; use the unprotected VizieR `asu-*`\n"
    "services for data); IOPscience 302-redirects to `validate.perfdrive.com`, so\n"
    "machine-readable tables cannot be pulled there; and `www.aanda.org` returns HTTP\n"
    "403 to all automated clients, which this lane hit directly.")

# ---- 3. Velocity coverage section ---------------------------------------
anchor = "---\n\n## 4. The binding constraint"
insert = """---

## 3b. Velocity coverage is the second uneven axis

All seven clusters have spectroscopic redshifts, but the downstream use is a
projected dispersion **field** sigma(R), and radial coverage varies by more than
an order of magnitude. N counts galaxies within rest-frame |dv| < 3000 km/s of
the cluster redshift; no dispersion was computed anywhere in this lane.

| Cluster | Best catalogue | N members | Max member R | Resolved sigma(R)? |
|---|---|---|---|---|
| Abell 2029 | Sohn 2019b (MMT/Hectospec, no colour cut) | 1215 | 8.75 Mpc | yes, comfortably |
| Abell S1063 | CLASH-VLT / Mercurio 2021 | 1192 | full VIMOS field | yes |
| MACS J0416 | Caminha 2017 (CLASH-VLT VIMOS) | 982 | 5.49 Mpc | yes |
| MACS J0717 | Ebeling 2014 (Keck/DEIMOS + LRIS + GMOS) | 559 | wide field | yes |
| Abell 2744 | Owers 2011 (AAOmega + literature) | 418 | 4.19 Mpc | yes |
| Abell 370 | PilotWINGS Lagattuta 2022 | 382 | 0.90 Mpc | core only |
| MACS J1149 | Schuldt 2024 | 151 | 0.65 Mpc | **marginal, core only** |

**Membership was re-derived, not inherited.** Neither the AS1063 nor the
MACS J0717 catalogue ships a membership column, so both were cut at rest-frame
|dv| < 3000 km/s about the cluster redshift (z = 0.3480 and 0.5458). That yields
1192 members for AS1063 against the 1234 that Mercurio et al. publish from a
peak-plus-gap selection, and 559 for MACS J0717 against the 537 that Limousin
et al. and Jauzac et al. publish from the same data. Both agree at the 3-4% level
with a genuinely different procedure, which is the expected size of disagreement
and not a signal worth chasing. **The counts above are a sanity check on the
ingest, not a reproduction of the published member lists** — the downstream test
should re-derive membership deliberately.

Three caveats that change how these should be weighted:

- **The MACS J0717 GLASS grism catalogue is unusable for kinematics.** Its
  redshift errors are sigma_z ~ 0.003-0.01 against a cluster velocity signal of
  ~0.005 in z: the error is comparable to the quantity being measured. Retained
  for membership only. The Ebeling catalogue replaces it.
- **CLASH-VLT applied a colour preselection** (R <~ 24 with colour cuts), so
  radial completeness for MACS J0416 and AS1063 is not uniform even though N and
  reach are good.
- MUSE catalogues are deep but confined to footprints under about 0.5 Mpc and
  give no radial leverage on their own.

**MACS J1149 is now the only cluster whose velocity field cannot support a
resolved sigma(R).** That is a change from the earlier reading, in which AS1063
and MACS J0717 also looked thin; both were artefacts of the CDS access route
rather than real data gaps.

"""
sub(anchor, insert + anchor)

# ---- 4. Third VizieR variant, merged-table trap, ID-with-space trap -----
sub("  returned HTTP 200 carrying an *unrelated* catalogue (`I/16`) plus\n"
    "  `CatalogsExamined=10213`. Nothing was substituted in any case.",
    "  returned HTTP 200 carrying an *unrelated* catalogue (`I/16`) plus\n"
    "  `CatalogsExamined=10213`. A **third** variant appeared in the velocity lane:\n"
    "  for a nonexistent `-source=`, VizieR returned HTTP 200 echoing\n"
    "  `#Name: J/MNRAS/430/1125` (Cooper et al. 2013, an RMS near-infrared YSO\n"
    "  survey) -- a completely unrelated real catalogue served silently in place of\n"
    "  the request, and URL-encoding the `+` does not help. **The only detector that\n"
    "  works across all three variants is to check that the response echoes back the\n"
    "  exact identifier requested.** Twelve identifiers were rejected this way in the\n"
    "  velocity lane and three more in the strong-lensing lane, five of them supplied\n"
    "  by the task brief itself. Nothing was substituted in any case.\n"
    "- *A merged-table trap, new to this programme*: VizieR fuses Braglia et al.\n"
    "  2009's two separate published tables -- A2744 (395 rows) and A2537 (809 rows) --\n"
    "  into a single 1204-row table distinguished only by an `A` column. Ingested\n"
    "  unfiltered it injects galaxies roughly 530 Mpc away into the A2744 velocity\n"
    "  field. The delivered file is filtered to `A=2744`, exactly 395 rows.\n"
    "- *A whitespace-parsing trap that silently loses a row*: the CLASH-VLT AS1063\n"
    "  catalogue has exactly one line with a SPACE INSIDE ITS OBJECT ID\n"
    "  (`CLASHVLTJ2249 9.98-442802.3`, which should read `CLASHVLTJ224959.98-442802.3`),\n"
    "  so that row splits into 8 fields while the other 3849 split into 7, shifting\n"
    "  every column. The failure is silent and quantifiable: the quality-flag counts\n"
    "  reported to this lane by a whitespace parse summed to 3849, one short of the\n"
    "  file's 3850 data lines. Repairing the identifier and asserting the field count\n"
    "  per row recovers it -- flag 3 goes from 3004 to 3005 and the total reconciles.\n"
    "- *Row counts asserted against the authoritative source, not the paper text*:\n"
    "  every VizieR table's row count was checked against the CDS ReadMe `Records`\n"
    "  column (ReadMes preserved in `velocities/raw/`). All matched exactly.")

# ---- 5. Process honesty + open items ------------------------------------
sub("- Anything about the *quality* of these data as constraints. This lane acquired\n"
    "  and characterised; it fitted nothing and scored nothing.",
    "- Anything about the *quality* of these data as constraints. This lane acquired\n"
    "  and characterised; it fitted nothing and scored nothing.\n"
    "\n"
    "Two process items worth recording rather than hiding. A background VizieR fetch\n"
    "in the strong-lensing lane **exited non-zero**, crashing after writing all 22\n"
    "catalogue files but before emitting its summary, with an empty log that swallowed\n"
    "the traceback; that lane's first report described the run as clean, which was\n"
    "wrong. The products were reconstructed by parsing the downloaded files directly,\n"
    "and an independent re-fetch has since confirmed that all nine VizieR catalogues\n"
    "the products depend on return **byte-identical data blocks** -- so no truncation\n"
    "occurred, and that now rests on direct comparison rather than inference.\n"
    "Separately, two mid-build bugs were caught and fixed: an early builder wrote\n"
    "strong-lensing products before applying redshift propagation, so its manifests\n"
    "briefly described counts the TSVs did not contain; and cz-sourced velocity tables\n"
    "(Owers, Sohn 2017, Boschin) had their redshifts converted while their errors were\n"
    "left in km/s. Both chains were rebuilt and re-verified.")

# ---- 6. Tiering, corrected ----------------------------------------------
sub("- **Tier 1, resolved baryonic model plus inner-region lensing plus kinematics** " + EM + "\n"
    "  A2744, MACS J1149, AS1063, MACS J0416. All four have seven-band Sérsic member\n"
    "  fits, measured member velocity dispersions, strong-lensing image lists with\n"
    "  large spectroscopic fractions, gas (SZ for MACS J0416, X-ray profiles for the\n"
    "  others), ICL fractions, and spectroscopic velocity catalogues. MACS J1149\n"
    "  additionally carries the SN Refsdal time delays.",
    "- **Tier 1, resolved baryonic model plus inner-region lensing plus kinematics** " + EM + "\n"
    "  A2744, MACS J0416, AS1063, MACS J1149. All four have seven-band Sérsic member\n"
    "  fits, measured *internal* member velocity dispersions from MUSE, strong-lensing\n"
    "  image lists with large spectroscopic fractions, gas (SZ for MACS J0416, X-ray\n"
    "  profiles for the others) and ICL fractions. **They are not equal on the\n"
    "  cluster-kinematics axis**: A2744, MACS J0416 and AS1063 all support a resolved\n"
    "  sigma(R) with 400-1200 members, whereas MACS J1149 has only 151 members inside\n"
    "  0.65 Mpc. MACS J1149 earns its place regardless, because it is the only cluster\n"
    "  in the entire sample carrying measured time delays.")

sub("- **MACS J0717** is the weakest target on two axes at once: no member Sérsic fits\n"
    "  and only 247 spectroscopic redshifts (GLASS). Include it only if the test\n"
    "  tolerates both.",
    "- **MACS J0717** has good cluster kinematics after all -- 559 members from the\n"
    "  Ebeling 2014 Keck survey, not the 17 usable redshifts its GLASS grism catalogue\n"
    "  offers -- but it has **no member Sérsic fits at all**, and it is a quadruple\n"
    "  merger. Include it where resolved member structure is not required.")

if s == orig:
    raise SystemExit("no changes applied")
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("patched REPORT.md: %d -> %d bytes" % (len(orig), len(s)))
