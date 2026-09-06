# Published halo return-field benchmark

Completed as part of the parallel relay investigation. Read the comparative
findings in [the relay report](../mond-atlas-relay-001/README.md).

The source receipts bind the original papers, author parameter file and SPARC
fit archive. Raw files stay private. The public run002 directory contains all
525 selected fit parameter rows, author Milky Way parameters, 11,046 scaled
conditional targets, 504 pilot vectors, all six trained formula fits, 18 score
summaries and the reviewed comparison plot. No raw observational responses were
scored. Run001 preserves the failed strict fit-quality parser attempt.

Five analytic/reference checks precede target calculations. Independent review
under the relay package confirms all table values and reproduces vector forces
using galpy. Reported profile errors are mathematical approximation errors,
not errors against observed galaxy speeds or independent held-out galaxies.

Reproduce into a fresh directory with
`python scripts/run_mond_atlas_halo_return.py <fresh-output-directory>`.
The source files must be present at the private paths in source-receipts.json
and match their hashes. Download URLs are recorded there; no credentials are
needed. Constants use kpc, solar masses and km/s; density conversions retain
the published log10(Msun/pc^3) inputs alongside Msun/kpc^3 values.
