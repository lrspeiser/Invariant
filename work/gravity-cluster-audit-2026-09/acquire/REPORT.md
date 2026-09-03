# Acquisition report

Ten datasets on disk with 14 machine-readable manifests (source URL, retrieval
timestamp, SHA-256, byte size, row count, column names with units). Raw upstream
responses retained unmodified alongside cleaned files.

| File | Rows | Cols |
|---|---:|---:|
| `dms6_table6_sigma_z.tsv` | 30 | 9 |
| `dms7_hR_hz.tsv` | 30 | 5 |
| `dms6_table1_galaxy_properties.tsv` | 30 | 22 |
| `dms6_table5_orientation.tsv` | 30 | 22 |
| `mulroy2019_sample.tsv` | 41 | 13 |
| `mulroy2019_observables.tsv` | 41 | 28 |
| `ettori2019_xcop_hydrostatic_masses.tsv` | 13 | 19 |
| `herbonnet2020_wl_masses.tsv` | 100 | 14 |
| `xxl365gc_clusters.tsv` | 302 | 19 |
| `umetsu2020_xxl_hsc_wl.tsv` | 136 | 28 |

## Corrections to the brief

**DiskMass VI vs VII were conflated in the request.** VI = A&A 557 A130 =
arXiv:1307.8130; VII = A&A 557 A131 = arXiv:1308.0336. The requested quantities
split across both. Both acquired. Neither is in VizieR.

**The X-COP weak-lensing premise was wrong.** Ettori 2019 *does* compare six
clusters against weak lensing but publishes **only aggregate ratios** — medians
1.16 (R500) and 1.17 (R200), error-weighted means 1.18 +- 0.12 and 1.14 +- 0.12.
It never names the six clusters and never tabulates their WL masses; the
comparison exists only as a figure. The WL masses came from *"Herbonnet et al.
2018, in prep."* — a private communication. Verified exhaustively: none of the
11 table files in the arXiv source contains the string "lens".

Acquired instead: the X-COP **hydrostatic** masses (13 clusters, note 13 not 12
— HydraA is present), plus **Herbonnet+2020** (MNRAS 497, 4684,
arXiv:1912.04414, 100 clusters), the published version of those measurements.
**The join is deliberately NOT performed** — naming differs across sources
(HydraA, RXC1825, ZW1215) so naive matching under-matches; the published masses
are titled "(updated)" and may differ from the 2018 draft Ettori used; and
Ettori compared at radii defined by the other method. Anyone performing the join
must validate that it reproduces 1.16 / 1.17.

## Caveats that bear on the vertical test (A_dyn)

1. **`Sigma_dyn` is not tabulated anywhere.** DiskMass VII has six tables and
   none carries it. It is defined by equation and shown only as radial profiles
   in atlas figures: `Sigma_dyn = sigma_z^2 / (pi G k h_z)`. No proxy was
   substituted. Every input is present so it is reconstructible — but that is a
   derivation carrying a free parameter `k`, and it should be an explicit
   decision rather than a silent one.
2. **`sigma_z` is not a resolved profile.** It is the exponential *fit*: central
   `sigma_z_0` in km/s plus scale length `h_sigma_z` in **arcsec**.
3. **`h_z` is inferred, not measured** — derived from `h_R` via the
   Bershady+2010b relation, so the two columns are correlated by construction.
   This matters for any `Sigma_dyn` error budget.
4. `h_R` appears in two units: kpc (VII) and arcsec (VI Table 1). Not reconciled.
5. `Chap2_tab1.tex` carries the authors' own note "CHANGED FROM A&A version" —
   the delivered file is the arXiv version.

## Caveats on the cluster samples

**LoCuSS (Mulroy 2019):** every requested column present — `M_WL`, `kT_X_ce`
(core-excised), `M_gas`, `L_K_tot`, `L_K_BCG`, `z`. But **all observables are
measured within r_500 set by the WL mass**, so they are not aperture-independent
of `M_WL`. `M_post` is the paper's own scaling-relation posterior, not an
independent mass. Empty cells are genuine: `Y_SZA` missing for 11, `lambda` for
8, `Y_X` for 2, `L_K` for 1.

**XXL:** `T300kpc` is a fixed 300 kpc aperture and is **NOT core-excised**, so
it is not comparable to Mulroy's `kT_X_ce`. Naive pooling of LoCuSS and XXL is
therefore blocked. `Mgas500kpc` is likewise a fixed aperture, not within r500.
`r500MT` is derived from an M-T relation, not measured. The Umetsu HSC
weak-lensing subset reaches `M500` = 5e12 Msun, below the ~2e13 group-regime
target.

## Two acquisition traps worth recording

**VizieR returns HTTP 200 with a generic page for a nonexistent `-source=`**,
not an error. A status-code check will silently ingest a non-answer. The
reliable existence check is the CDS FTP directory listing at
`https://cdsarc.cds.unistra.fr/ftp/<CAT>/`.

**LaTeX table extraction failed silently on a two-`table*` layout.** The
Herbonnet table is typeset as two consecutive `table*` environments; bounding on
the first `\end{tabularx}` returned 59 of 100 rows with no error. Caught only by
a row-count assertion, and fixed by bounding on the table's own `\label`. Apply
a row-count assertion to every LaTeX extraction — this failure mode is silent.
