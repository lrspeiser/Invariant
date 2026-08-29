# Items 71–72 independent-confirmation and novelty results

## Decisions

- Item 71: `ITEM71_INTERNAL_REPRODUCIBILITY_PASSED_EXTERNAL_INDEPENDENT_CONFIRMATION_NOT_OBTAINED`
- Item 72: `ITEM72_POTENTIALLY_NEW_SYNTHESIS_OF_KNOWN_MOTIFS_HISTORICAL_NOVELTY_NOT_ESTABLISHED`

These decisions close the current 72-item execution audit. They do not close the scientific
search or claim a solution to the missing-gravity problem.

## Item 71: independent confirmation

All frozen internal replays for Items 59–70 reproduced exactly. This establishes that the
recorded computations and receipts are internally reproducible.

It does not establish independent confirmation. The four previously sealed Item 59
clusters are from the same X-COP release and analysis family as the eight development
clusters. No external-survey cluster was tested, and no independently written
implementation replayed the prediction. A repeat execution of our own code is not counted
as investigator, instrument, or reduction independence.

The only claim still eligible for future confirmation is narrow:

> Conditional on one outer pressure anchor, spherical hydrostatic balance, and the frozen
> nuisance model, the radial cross-scale-boundary ansatz predicted held-out X-COP SZ
> pressure and X-ray temperature profile shapes better than the frozen comparators.

The following are not eligible claims: a universal alternative to GR, elimination of dark
matter, direct lensing prediction, galaxy-to-cluster universality of `beta=1.5`, or
historical novelty.

No new observational targets or sealed SPARC confirmation rows were opened at Item 71.

## Item 72: prior art and formula classification

The scoped search did not locate the exact expression

```text
q = (gbar/a0) / [(gbar/a0) + 0.1]
g = gbar + beta [gbar K_in(q) + a0 K_sym(q)]
```

That absence does not prove global novelty. Closely overlapping ideas were already
published:

- the acceleration scale and local baryonic response in the [galaxy radial acceleration relation](https://arxiv.org/abs/1609.05917);
- an action-derived modified-potential construction in [QUMOND](https://arxiv.org/abs/0911.5464);
- density-dependent gravitational permittivity applied to both galaxies and clusters in [refracted gravity](https://arxiv.org/abs/1603.04943);
- a [covariant scalar-tensor completion of refracted gravity](https://arxiv.org/abs/2109.11217);
- integral and reciprocal kernels in [nonlocal modified Poisson gravity](https://arxiv.org/abs/1111.4702);
- unified matter, lensing, Solar-System, causality, and cosmology structure in [TeVeS](https://arxiv.org/abs/astro-ph/0403694);
- a 2026 [source-side nonlocal response kernel tested on SPARC](https://pubmed.ncbi.nlm.nih.gov/42072602/) with explicit Solar-System and relativistic-completion caveats.

The exact candidate is nevertheless not merely an algebraic rewrite of a purely local RAR
function. The frozen witness constructed two radial profiles with identical local radius
and identical local `gbar`, but different baryonic occupancy elsewhere. A local RAR formula
returned exactly the same value for both profiles; the candidate differed by 6.67% because
its kernels inspect other radii.

That witness establishes behavioral non-equivalence only to formulas depending exclusively
on the local radius and local acceleration. It cannot establish non-equivalence to the
large universe of nonlocal, permittivity, auxiliary-field, or action-level theories.

The final classification is therefore:

- known exact formula: **not found in the scoped search**;
- algebraic rewrite of a purely local RAR/MOND interpolator: **no**;
- combination of known mechanism families: **yes**;
- potentially new synthesis: **yes**;
- historical novelty or new fundamental physics: **not established**.

Novelty does not repair the failed galaxy transfer, Solar-System limit, or missing lensing
and formal structure. Lack of proven novelty also does not erase the narrow empirical
X-COP lead.

## Reproduction

```powershell
python -m sigma_theory_compiler.gravity_items71_72_final_gates replay
python -m pytest tests/test_gravity_items71_72_final_gates.py -q
```

Paid model calls: zero. GPU use: none.
