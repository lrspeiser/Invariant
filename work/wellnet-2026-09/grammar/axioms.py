"""Stage 0 -- the universe grammar, enumerated over the charter's axiom axes.

The charter's Stage 0 asks for a grammar specifying allowed source types, field
ranks, notions of locality, directions, memory and path operations, matter-light
relationships, and required conservation and symmetry properties -- so that

    "The final search is then over sparse combinations of admissible physical
     building blocks, not arbitrary columns."

and its "Alternate universes" section supplies the axes themselves.

This module enumerates that space and answers three questions the programme has
never asked:

    1. How big is the space of universes the charter defines?
    2. How much of it has this programme actually visited?
    3. How much of it is decidable a priori by the existing compiler gates,
       and how much is simply out of reach of the current machinery?

It opens no data.  Run:  python axioms.py
"""
import io
import itertools
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- the 15 axes
# Transcribed from the charter's "Alternate universes should be generated from
# axiom choices" table.  Order and wording preserved.
AXES = {
    "source": ["rest_mass", "full_stress_energy", "pressure_or_entropy",
               "electromagnetic_energy", "vacuum_state", "information_state",
               "combined"],
    "field_type": ["scalar", "vector", "symmetric_tensor", "antisymmetric",
                   "mixed", "graph_or_network", "emergent_geometry"],
    "locality": ["point_local", "finite_range_nonlocal", "path_dependent",
                 "globally_constrained", "past_light_cone"],
    "superposition": ["linear", "nonlinear", "saturating", "thresholded",
                      "phase_changing", "hysteretic"],
    "directionality": ["isotropic", "source_aligned", "disk_aligned",
                       "angular_momentum", "external_field", "tidal",
                       "well_network"],
    "axis_origin": ["local_source", "external_matter", "independent_universal",
                    "dynamically_evolved"],
    "propagation": ["instantaneous_weak_field", "light_speed", "dispersive",
                    "retarded", "finite_memory"],
    "geometry": ["force_in_fixed_space", "curved_metric", "torsion",
                 "nonmetricity", "extra_dimensions", "discrete_causal_network"],
    "eff_dimension": ["three_d", "scale_dependent", "sheet_or_filament"],
    "matter_light": ["universal", "separate_but_fixed", "environment_dependent",
                     "frequency_dependent", "polarization_dependent"],
    "equivalence": ["exact", "approximate", "composition_dependent",
                    "state_dependent"],
    "conservation": ["exact_reciprocal", "exchange_with_field",
                     "explicit_controlled_violation"],
    "cosmology": ["expanding_geometry", "evolving_clocks", "static",
                  "bounce", "emergent_expansion", "path_generated_redshift"],
    "vacuum": ["passive", "polarizable", "coherent", "directional",
               "phase_changing", "history_dependent"],
    "initial_conditions": ["standard_primordial", "enhanced_baryonic_growth",
                           "topological_seeds", "vacuum_phase_seeds"],
}

# ------------------------------------------------- what this programme visited
# Every candidate family the record shows as actually constructed and scored,
# expressed in the charter's axes.  `None` means the family does not commit.
VISITED = {
    "Newton": dict(source="rest_mass", field_type="scalar",
                   locality="point_local", superposition="linear",
                   directionality="isotropic", geometry="force_in_fixed_space",
                   conservation="exact_reciprocal", matter_light="universal"),
    "AQUAL": dict(source="rest_mass", field_type="scalar",
                  locality="point_local", superposition="nonlinear",
                  directionality="isotropic", geometry="force_in_fixed_space",
                  conservation="exact_reciprocal", matter_light="universal"),
    "QUMOND/RAR": dict(source="rest_mass", field_type="scalar",
                       locality="point_local", superposition="nonlinear",
                       directionality="isotropic",
                       geometry="force_in_fixed_space",
                       conservation="exact_reciprocal", matter_light="universal"),
    "family B (depth-gated)": dict(source="rest_mass", field_type="scalar",
                                   locality="globally_constrained",
                                   superposition="nonlinear",
                                   directionality="isotropic"),
    "family C (well network)": dict(source="rest_mass",
                                    field_type="graph_or_network",
                                    locality="finite_range_nonlocal",
                                    superposition="nonlinear",
                                    directionality="well_network",
                                    axis_origin="external_matter"),
    "family D (pairs)": dict(source="rest_mass", field_type="graph_or_network",
                             locality="finite_range_nonlocal",
                             superposition="nonlinear",
                             directionality="well_network"),
    "family E (tidal tensor)": dict(source="rest_mass",
                                    field_type="symmetric_tensor",
                                    locality="point_local",
                                    superposition="nonlinear",
                                    directionality="tidal",
                                    axis_origin="local_source"),
    "tidal-gated scalar": dict(source="rest_mass", field_type="scalar",
                               locality="point_local", superposition="thresholded",
                               directionality="isotropic"),
    "external-axis tensor (AU)": dict(source="rest_mass",
                                      field_type="symmetric_tensor",
                                      locality="point_local",
                                      directionality="external_field",
                                      axis_origin="external_matter"),
    "nonlocal path kernel (AG)": dict(source="rest_mass", field_type="scalar",
                                      locality="path_dependent",
                                      superposition="nonlinear",
                                      directionality="isotropic"),
    "path redshift (AK)": dict(locality="path_dependent",
                               cosmology="path_generated_redshift"),
}

# ------------------------------------------------ what the machinery can reach
# An axis choice is REACHABLE if the current bench can express it AND score it.
# Recorded from the programme's own findings, with the run that established it.
REACH = {
    "field_type": {
        "scalar": "reachable",
        "symmetric_tensor": "reachable",
        "graph_or_network": "reachable",
        "vector": "UNREACHABLE: no vector-potential sector; Run AU declared it "
                  "outside the compiler's scalar-potential model class",
        "antisymmetric": "UNREACHABLE: same",
        "mixed": "UNREACHABLE: same",
        "emergent_geometry": "UNREACHABLE: no geometry solver",
    },
    "geometry": {
        "force_in_fixed_space": "reachable",
        "curved_metric": "UNREACHABLE: no relativistic solver; the lensing "
                         "closure is imposed, not derived (Run AL)",
        "torsion": "UNREACHABLE", "nonmetricity": "UNREACHABLE",
        "extra_dimensions": "UNREACHABLE",
        "discrete_causal_network": "UNREACHABLE",
    },
    "propagation": {
        "instantaneous_weak_field": "reachable",
        "light_speed": "UNREACHABLE: static solver only",
        "dispersive": "UNREACHABLE", "retarded": "UNREACHABLE",
        "finite_memory": "UNREACHABLE: no time-dependent scoring channel",
    },
    "matter_light": {
        "universal": "reachable",
        "separate_but_fixed": "reachable via the slip parameter (Run AL)",
        "environment_dependent": "PARTIAL: AL could fit it but not derive it",
        "frequency_dependent": "UNREACHABLE: no spectral channel",
        "polarization_dependent": "UNREACHABLE",
    },
    "conservation": {
        "exact_reciprocal": "reachable",
        "exchange_with_field": "UNREACHABLE: no candidate declares a carrier",
        "explicit_controlled_violation": "measurable but never admissible",
    },
    "cosmology": {
        "expanding_geometry": "reachable (Run AP, linear only)",
        "evolving_clocks": "UNREACHABLE",
        "static": "reachable",
        "bounce": "UNREACHABLE", "emergent_expansion": "UNREACHABLE",
        "path_generated_redshift": "PARTIAL: Run AK bounded it, 90 sigma "
                                   "against the energy-drain half",
    },
    "vacuum": {
        "passive": "reachable",
        "polarizable": "UNREACHABLE: no latent-field dynamics",
        "coherent": "UNREACHABLE", "directional": "UNREACHABLE",
        "phase_changing": "UNREACHABLE", "history_dependent": "UNREACHABLE",
    },
    "equivalence": {
        "exact": "reachable", "approximate": "reachable",
        "composition_dependent": "UNREACHABLE: no composition channel",
        "state_dependent": "UNREACHABLE",
    },
}


def main():
    names = list(AXES)
    sizes = [len(AXES[a]) for a in names]
    total = 1
    for s in sizes:
        total *= s

    print("=" * 78)
    print("STAGE 0 -- THE UNIVERSE GRAMMAR")
    print("=" * 78)
    print(f"{'axis':<20} {'options':>8}")
    print("-" * 78)
    for a, s in zip(names, sizes):
        print(f"{a:<20} {s:>8}")
    print("-" * 78)
    print(f"{'PRODUCT':<20} {total:>8,}  distinct universes definable by the charter")
    print()

    # how much has been visited?
    committed = []
    for nm, spec in VISITED.items():
        committed.append(len(spec))
    print(f"families this programme actually constructed and scored: "
          f"{len(VISITED)}")
    print(f"  each commits {min(committed)}-{max(committed)} of {len(names)} axes;"
          f" the rest are left implicit")
    # distinct axis-VALUES touched, per axis
    print()
    print(f"{'axis':<20} {'options':>8} {'touched':>8} {'coverage':>10}")
    print("-" * 78)
    tot_opt = tot_touch = 0
    for a in names:
        touched = {spec[a] for spec in VISITED.values() if spec.get(a)}
        tot_opt += len(AXES[a])
        tot_touch += len(touched)
        print(f"{a:<20} {len(AXES[a]):>8} {len(touched):>8} "
              f"{len(touched)/len(AXES[a]):>9.0%}")
    print("-" * 78)
    print(f"{'TOTAL axis-values':<20} {tot_opt:>8} {tot_touch:>8} "
          f"{tot_touch/tot_opt:>9.0%}")
    print()

    # reachability
    print("REACHABILITY of the axes the record can speak to")
    print("-" * 78)
    reach_n = unreach_n = partial_n = 0
    for a, table in REACH.items():
        ok = [v for v, s in table.items() if s.startswith("reachable")]
        pa = [v for v, s in table.items() if s.startswith("PARTIAL")]
        no = [v for v, s in table.items() if s.startswith("UNREACHABLE")]
        reach_n += len(ok)
        partial_n += len(pa)
        unreach_n += len(no)
        print(f"  {a:<16} reachable {len(ok)}  partial {len(pa)}  "
              f"unreachable {len(no)}")
    scored = reach_n + partial_n + unreach_n
    print("-" * 78)
    print(f"  of {scored} axis-values assessed: {reach_n} reachable, "
          f"{partial_n} partial, {unreach_n} UNREACHABLE "
          f"({unreach_n/scored:.0%})")
    print()
    print("The unreachable set is not a list of bad ideas. It is the list of")
    print("physics the current bench cannot express OR score, which is what")
    print("Stage 0 exists to make visible.")

    doc = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        lane="work/wellnet-2026-09/grammar",
        stage="0 (universe grammar)",
        charter="C:/Users/henry/dev/invariant-gravity-discovery-charter.md",
        axes={a: AXES[a] for a in names},
        axis_sizes=dict(zip(names, sizes)),
        total_universes=total,
        visited=VISITED,
        axis_value_coverage=dict(
            total_options=tot_opt, touched=tot_touch,
            fraction=tot_touch / tot_opt),
        reachability=REACH,
        reach_counts=dict(reachable=reach_n, partial=partial_n,
                          unreachable=unreach_n, assessed=scored),
        opened_observational_data=False)
    p = os.path.join(HERE, "axioms.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
