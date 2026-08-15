# Goal 25/26 evidence notebook: failure, construction, bounded success

> This generated report is intentionally narrower than a theorem announcement.
> It shows what failed, what was built next, what now passes, and what remains open.

## How to read the evidence

Eight checked JSON receipts were parsed, content-seal replayed, and checked against exact terminal results, measured counts, and claim boundaries. The same sealed report object generated this Markdown file and the notebook twin.

## Goal 25 lane: coordinate-free coefficient construction

### Failure boundary

The directional K0 formula passed six exact controls, but its own receipt left the expanded polynomial packet, D4, and H7 open.

That was a useful failure: it isolated representation, rather than the six tested directions, as the next missing object. There were zero mismatches in the 3,025-entry e1 reference comparison, but no full D4 or H7 conclusion followed.

### Construction

The next receipt serialized K0 as a 55×55 unit-sphere polynomial packet with **847 nonzero entries** and **2732 normal-form terms**. All 3,025 sphere-identity entries reduced to zero. This authorized 15 K55 order-one packets while still recording that zero had yet been registered.

### Bounded success

The K55 order-one gate then registered **15 of 15 packets**, containing 17704 normal-form terms. It reduced 45,375 differentiated matrix identities with **zero nonzero remainders**.

The TC2 order-one gate independently registered **15 of 15 packets**. All 15 are exact zero packets for the sealed fixed-coefficient jet basis, and the product-rule replay has zero nonzero remainders.

This is an order-one registration success. Orders two through four, recurrence rows, full D4, global H7, PDE closure, and lifespan remain open.

## Goal 26 lane: coupled matter and gravity constraints

### First bounded success

At the flat reference, the 85-state Schur construction produced **1 full symmetrizer** on **1 bounded nonzero Maxwell-potential domain**, with zero symmetry-residual entries. This is not candidate-jet uniformity or a global result.

### Typed failure

Differentiating the modified-harmonic gauge source initially stopped on **5 missing jet families**, totaling **780 primitive slots**. The readiness receipt divided them into 48 resumable chunks and claimed zero completed differentiated maps.

### Construction

The materializer registered all 780 slots: 580 formal external-jet atoms and 200 physical-metric third-derivative operator slots. A **17-template** indexed tensor program now constructs all four divergence components. The external atoms are formal inputs, not certified values.

### Bounded success

At the registered flat constant formulation reference, the scalar-expansion gate lowered all **4 gravity-constraint rows** into the 85-state ordering. It found **112 exact Q(√2) coefficients**—28 per row—and bound the common row packet to **12 candidate manifests**.

The nonlinear/general row expansion remains blocked on 1,010 exact scalar values before a common domain: 580 external jets, 280 lower formulation jets, and 150 physical metric jets, including sourced acceleration data.

## Receipt ledger

| Evidence | Terminal result | File SHA-256 |
|---|---|---|
| `directional_k0` | `pass_exact_coordinate_free_K0_directional_lift_formula` | `0f209e5880c971f7aab7ec9014026648a641be4cd7211e68d040ba85cfe823e6` |
| `k0_polynomial` | `pass_exact_coordinate_free_K0_polynomial_packet` | `9b580ee9dc3017627ea062e59147ad2a9b55442630c4db6b546eff48902a2de1` |
| `k55_order_one` | `pass_exact_15_coordinate_free_K55_Taylor_order_one_packets_registered` | `c3b2e7be8bec446ea12f3713f954995cc15d9fd1d91f1ac47da6b5a2f7dfec0b` |
| `tc2_order_one` | `pass_exact_15_coordinate_free_TC2_Taylor_order_one_packets_registered` | `47ed0d25cf97af3027b2eb31effbb7812f54aef7f87b53b68bb0467bea0d22c7` |
| `bounded_b_symmetrizer` | `PASS_EXACT_FLAT_SPHERE_FULL_SYMMETRIZER_BOUNDED_B` | `5a3d63be1ed6e382f936137c99d0241f31b9b372de145309d8e94719a5feea79` |
| `gauge_readiness` | `PASS_RESUMABLE_READINESS_CONTRACT_FIVE_PRIMITIVE_JET_BLOCKERS` | `f331e0b3d4ef5684fa0527f329203025f89fbbd61ce6760c8495643525625501` |
| `gauge_materializer` | `PASS_EXACT_INDEXED_GAUGE_MAP_WITH_FORMAL_EXTERNAL_JET_PACKETS` | `f9fb0398b08ec3a86ce46f5c970a7b04a9de421755f9602164b92b33b5db73d9` |
| `flat_scalar_expansion` | `BOUNDED_PASS_FLAT_SCALAR_ROWS_TYPED_BLOCK_GENERAL_EXTERNAL_JETS` | `81b0ca2f193e1f7073ab801097becb726757b175312e8c3601c5990b47f2e362` |

## Claim boundary

The receipts support the exact directional, polynomial, order-one, flat-reference, and indexed constructions described above. They do **not** establish full D4, global H7, nonlinear/global closure, sourced constraint propagation, universal all-matter closure, or promotion.

Sealed report content SHA-256: `ad17ecd903fe59dfb3cd606da499d9ba65cc1bb4c78550239d12d907199abed1`.
