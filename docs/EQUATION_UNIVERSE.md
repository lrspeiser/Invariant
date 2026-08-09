# Sigma Equation Universe

The equation universe is a provenance-aware prior-art and project-history system. It is designed
to answer four different questions without conflating them:

1. Is a candidate algebraically equivalent to a curated known equation?
2. Is it the same mathematical structure after a type-compatible symbol rename?
3. Was this exact reduced formula already generated and screened by Sigma?
4. If none of those checks match, what was searched and what remains unknown?

The answer to question 4 is always **not found in this finite corpus**, never “novel.” Scientific
originality, patent novelty, and copyright are different questions and require broader review.
Changing notation, rearranging an equality, changing units, integrating by parts, or making a field
redefinition does not create a new theory and must not be used to conceal a source.

## Current production corpus

The audited production database is `runs/equation-universe/equations-v1.sqlite`. Its initial
curated layer contains:

- 7 fully attributed sources;
- 18 known equations, tensor statements, and formal-domain statements, including the
  preferred-foliation acceleration and its gradient-square integration-by-parts prior art;
- 42 typed variables;
- 3 machine-verified derivations;
- exact semantic and typed structural hashes;
- a compact registration of the completed Sigma Generator v2 formula space.

The registered Generator v2 space covers **1,088,651,720 processed formulas**, including
**17,540,440 sampled-static survivors**. It stays compact by retaining the audited generator
manifest, basis, and fixed-width binary survivor ledger. A lookup decomposes a candidate into its
basis terms, reconstructs its deterministic ordinal and candidate ID, and binary-searches the
appropriate survivor block. It does not need one SQLite row per generated formula.

This generator history is screening provenance, not evidence that a formula is physically valid.

## Data model

The SQLite schema has these layers:

| Table | Purpose |
|---|---|
| `sources` | URL, authors, year, source type, license, and allowed ingestion mode |
| `equations` | Original representation, normalized relation, hashes, dimensions, assumptions, and validity domain |
| `variables` | Stable meaning, physical dimensions, field kind, and tensor rank for every symbol |
| `derivations` / `derivation_inputs` | Directed derivation graph with assumptions and machine proof status |
| `equivalence_edges` | Proven semantic equality or typed alpha-structural equivalence |
| `formula_spaces` | Compact, random-access registrations of enormous generated formula histories |
| `import_runs` | Input digest, counts, rejected records, and reproducible ingestion history |

Every curated scalar equation is parsed with an allowlisted AST. Arbitrary Python execution is not
allowed. The canonicalizer moves both sides to a residual, clears rational denominators, normalizes
sign and polynomial content, and hashes the result. Structural hashes permit symbol permutations
only within the same physical-dimension, field-kind, and tensor-rank group.

## Classification semantics

Classifications are ordered from strongest exact overlap to weakest result:

- `known_semantic_equivalent`: the canonical mathematical residual matches a stored equation;
- `known_structural_analogue`: the residual matches after a type-compatible variable rename;
- `known_project_history_exact`: the formula belongs to a registered generated space and was in
  its processed range;
- `not_found_in_corpus`: no match was found in the material actually registered.

Every result includes `novelty_claim_allowed: false`. Nearest expressions use structural feature
similarity only; they are leads for review, not equivalence proofs.

## Source policy

`configs/equation_universe/source_policy.json` is the machine-readable source policy.

- Wikidata structured data may be fully imported under CC0.
- DLMF is metadata-only by default because its notice limits copying and prohibits bulk
  redistribution. Store links and independently encoded mathematical facts, not a bulk mirror.
- Wolfram Functions is metadata-only unless a compatible reuse grant is documented.
- arXiv papers are handled per paper. The default is citation metadata plus independently encoded
  mathematical facts, never copied prose.
- INSPIRE is used for citation and literature metadata unless a record-specific license allows more.
- Sigma-authored structured records may be ingested fully with their internal provenance.

A metadata-only source record is rejected unless `independently_encoded` is true. A blocked source
cannot contribute equations. Source attribution and license metadata cannot be stripped from a
derived record.

## Build, register, audit, and query

From the project root:

```powershell
$env:PYTHONPATH='src'
python -m sigma_theory_compiler equation-universe-build `
  --seed configs\equation_universe\gravity_seed_v1.json `
  --database runs\equation-universe\equations-v1.sqlite `
  --report runs\equation-universe\build-report.json `
  --replace

python -m sigma_theory_compiler equation-universe-register-history `
  --database runs\equation-universe\equations-v1.sqlite `
  --manifest runs\generator-v2\billion-survivor-export.json `
  --basis runs\generator-v2\basis-library.json `
  --survivor-dir runs\generator-v2\survivors `
  --name "Sigma Generator v2 billion-action history"

python -m sigma_theory_compiler equation-universe-audit `
  --database runs\equation-universe\equations-v1.sqlite `
  --output runs\equation-universe\audit-report.json
```

Import another reviewed shard:

```powershell
python -m sigma_theory_compiler equation-universe-import `
  --database runs\equation-universe\equations-v1.sqlite `
  --input configs\equation_universe\reviewed-shard.json
```

Classify a structured record:

```powershell
python -m sigma_theory_compiler equation-universe-classify `
  --database runs\equation-universe\equations-v1.sqlite `
  --record candidate-equation.json `
  --output candidate-prior-art.json
```

A reduced Sigma candidate may add `formula_space_expression` to request an exact Generator v2
history lookup in addition to canonical equation matching.

## Campaign integration

`configs/campaign_v1.json` points every campaign worker at the production equation universe. The
symbolic report and final candidate dossier include an `equation_prior_art` block. The dossier task
also records the soft provenance gate `equation_prior_art_screen`:

- `pass` means the database was queried successfully, regardless of whether it matched;
- `unresolved` means the database was missing, corrupt, or the query could not be parsed;
- it is never a hard physics gate and never awards truth probability;
- a rejected physics candidate still receives this screen because dossier tasks survive terminal
  gate rejection.

## Growing this into a large literature graph

New material should enter as reviewed, content-addressed shards rather than an untraceable scrape.
The intended ingestion pipeline is:

1. discover citation metadata and stable identifiers;
2. apply the provider/source license policy;
3. independently encode the equation, variables, assumptions, and validity domain;
4. run safe parsing and dimensional audit;
5. canonicalize and deduplicate;
6. verify any claimed derivation;
7. quarantine rejected or ambiguous records for human review;
8. import the accepted shard and retain the import digest.

High-value next domains are Newtonian mechanics and gravity, post-Newtonian limits, GR exact
solutions and perturbation equations, scalar/vector/tensor field theories, Einstein-Aether and
Horava gravity, DHOST/degenerate scalar-tensor controls, mathematical identities used in
variation, and the Solar-System observables used by the reference suite.

LLMs may help extract a proposed structured record, suggest symbol mappings, and locate possible
sources. They may not certify equivalence, derivation correctness, provenance, licensing, or
novelty. Those remain deterministic checks plus review.

## Formula modification policy

Known equations can be used as parents for generator proposals, but every modification must be a
new child node with explicit lineage:

- parent equation IDs;
- exact transformation and assumptions;
- changed fields, operators, or couplings;
- dimensional and symmetry checks;
- proof of equivalence if the change is claimed to preserve the theory;
- a new falsification plan if the change is claimed to alter the theory.

This makes modification scientifically useful: the system can say exactly what changed and why.
It is not a mechanism for cosmetically rewriting an existing formula and calling it original.
