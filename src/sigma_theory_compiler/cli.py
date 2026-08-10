from __future__ import annotations

import argparse
import json
from pathlib import Path

from .action_health import analyze_action_health
from .action_ir import compile_action_file, load_action_grammar
from .adm_ir import compile_adm_ir, write_adm_ir
from .compiler import TheoryCompiler
from .covariant_export import export_covariant_candidate_files
from .covariant_variation import vary_proca_action_file, vary_scalar_action_file
from .cubic_bssn_domain import (
    run_cubic_bssn_domain_campaign,
    write_cubic_bssn_domain_campaign,
)
from .dhost_pack import compile_reduced_dhost_pack
from .dirac_ir import compile_dirac_ir, write_dirac_ir
from .equation_universe import EquationUniverse, build_equation_universe
from .flrw_background import certify_flrw_background
from .flrw_campaign import run_flrw_background_campaign, write_flrw_background_campaign
from .formal_backend import (
    load_field_contract,
    run_formal_control_suite,
    write_formal_report,
)
from .gpu_screen import crosscheck_dense_gpu_screen, run_dense_gpu_screen
from .grammar_exhaustion import audit_static_grammar_exhaustion, write_grammar_exhaustion
from .hamiltonian_ir import compile_physical_hamiltonian_ir, write_physical_hamiltonian_ir
from .high_throughput import crosscheck_manifest, write_crosscheck
from .knowledge import GateOntology, KnowledgeBuilder, prioritize_registered_formulas
from .legendre_ir import compile_legendre_ir, write_legendre_ir
from .observation_eligibility import (
    audit_galaxy_observable_protocol,
    audit_theory_observation_eligibility,
    write_observation_eligibility,
)
from .physics_language import compile_physics_program
from .principal_ir import compile_physical_principal_ir, write_physical_principal_ir
from .quartic_auxiliary_time_campaign import (
    run_quartic_auxiliary_time_campaign,
    write_quartic_auxiliary_time_campaign,
)
from .quartic_constraint_reconstruction_campaign import (
    run_quartic_constraint_reconstruction_campaign,
    write_quartic_constraint_reconstruction_campaign,
)
from .quartic_coordinate_jet_tube_campaign import (
    run_quartic_coordinate_jet_tube_campaign,
    write_quartic_coordinate_jet_tube_campaign,
)
from .quartic_dirac_hamiltonian_campaign import (
    run_quartic_dirac_hamiltonian_campaign,
    write_quartic_dirac_hamiltonian_campaign,
)
from .quartic_euler_remainder_majorant_campaign import (
    run_quartic_euler_remainder_majorant_campaign,
    write_quartic_euler_remainder_majorant_campaign,
)
from .quartic_first_order_reduction_campaign import (
    run_quartic_first_order_reduction_campaign,
    write_quartic_first_order_reduction_campaign,
)
from .quartic_full_symmetrizer_moser_campaign import (
    run_quartic_full_symmetrizer_moser_campaign,
    write_quartic_full_symmetrizer_moser_campaign,
)
from .quartic_geometric_jet_campaign import (
    run_quartic_geometric_jet_campaign,
    write_quartic_geometric_jet_campaign,
)
from .quartic_homogeneous_frequency_symbol_campaign import (
    run_quartic_homogeneous_frequency_symbol_campaign,
    write_quartic_homogeneous_frequency_symbol_campaign,
)
from .quartic_linear_x_campaign import (
    run_quartic_linear_x_symbol_campaign,
    write_quartic_linear_x_symbol_campaign,
)
from .quartic_linearized_energy_campaign import (
    run_quartic_linearized_energy_campaign,
    write_quartic_linearized_energy_campaign,
)
from .quartic_nonlinear_evolution_campaign import (
    run_quartic_nonlinear_evolution_campaign,
    write_quartic_nonlinear_evolution_campaign,
)
from .quartic_nonquasilinear_pde_campaign import (
    run_quartic_nonquasilinear_pde_campaign,
    write_quartic_nonquasilinear_pde_campaign,
)
from .quartic_quasilinear_moser_campaign import (
    run_quartic_quasilinear_moser_campaign,
    write_quartic_quasilinear_moser_campaign,
)
from .quartic_solved_source_moser_campaign import (
    run_quartic_solved_source_moser_campaign,
    write_quartic_solved_source_moser_campaign,
)
from .quartic_symmetrizer_domain import (
    run_quartic_symmetrizer_domain_campaign,
    write_quartic_symmetrizer_domain_campaign,
)
from .quartic_symmetrizer_symbol_moser_campaign import (
    run_quartic_symmetrizer_symbol_moser_campaign,
    write_quartic_symmetrizer_symbol_moser_campaign,
)
from .registry import write_registry
from .relativity import run_relativity_reference_suite, write_relativity_report
from .scalar_tensor_pack import compile_scalar_tensor_pack
from .stability_ir import compile_stability_ir, write_stability_ir
from .static_dictionary import (
    audit_priority_static_lift,
    compile_static_dictionary_ir,
    write_static_artifact,
)
from .survivors import audit_survivor_export, prioritize_generated_survivors
from .validation import run_validation, write_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sigma-compile",
        description="Enumerate and kill candidates in a bounded Sigma Gravity action grammar.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search", help="Compile every candidate in a JSON grammar")
    search.add_argument("--config", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate", help="Run analytic known-answer checks against the compiler"
    )
    validate.add_argument("--output", type=Path, required=True)
    reference = subparsers.add_parser(
        "reference", help="Run GR, Solar-System, and galaxy-exterior golden controls"
    )
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument(
        "--health",
        type=Path,
        default=Path(
            "runs/formal-controls-v1/action-health/einstein_hilbert_control/action-health.json"
        ),
    )
    reference.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/observational_evidence_policy.json"),
    )
    formal = subparsers.add_parser(
        "formal-controls",
        help="Run covariant field-contract, variation-identity, ADM/Dirac, and mode controls",
    )
    formal.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/covariant_field_contract.json"),
    )
    formal.add_argument("--output", type=Path, required=True)
    observation_audit = subparsers.add_parser(
        "observation-audit",
        help="Audit a frozen action-health chain before Solar references or candidate data access",
    )
    observation_audit.add_argument("--health", type=Path, required=True)
    observation_audit.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/observational_evidence_policy.json"),
    )
    observation_audit.add_argument(
        "--mode",
        choices=("known_answer_reference", "candidate_data"),
        required=True,
    )
    observation_audit.add_argument("--output", type=Path, required=True)
    galaxy_protocol_audit = subparsers.add_parser(
        "galaxy-protocol-audit",
        help="Audit the sealed observable-to-observable galaxy discovery protocol",
    )
    galaxy_protocol_audit.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/galaxy_observable_protocol.json"),
    )
    galaxy_protocol_audit.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/observational_evidence_policy.json"),
    )
    galaxy_protocol_audit.add_argument("--output", type=Path, required=True)
    action_compile = subparsers.add_parser(
        "action-compile",
        help="Compile a bounded covariant action specification into canonical fail-closed IR",
    )
    action_compile.add_argument("--spec", type=Path, required=True)
    action_compile.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    action_compile.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    action_compile.add_argument("--output", type=Path, required=True)
    action_static = subparsers.add_parser(
        "action-static-dictionary",
        help="Derive an action-hash-bound static invariant dictionary and optional generator lift decision",
    )
    action_static.add_argument("--spec", type=Path, required=True)
    action_static.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    action_static.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    action_static.add_argument("--generator-expression")
    action_static.add_argument("--output", type=Path, required=True)
    static_lift_audit = subparsers.add_parser(
        "static-lift-audit",
        help="Classify a prioritized Generator queue against a derived static dictionary",
    )
    static_lift_audit.add_argument("--dictionary", type=Path, required=True)
    static_lift_audit.add_argument("--priority", type=Path, required=True)
    static_lift_audit.add_argument("--output", type=Path, required=True)
    covariant_export = subparsers.add_parser(
        "covariant-export",
        help="Export exactly representable prioritized formulas as origin-bound covariant action specs",
    )
    covariant_export.add_argument("--priority", type=Path, required=True)
    covariant_export.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    covariant_export.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    covariant_export.add_argument("--output", type=Path, required=True)
    grammar_exhaustion = subparsers.add_parser(
        "grammar-exhaustion",
        help="Audit whether every family in the frozen q/z queue is now hard-rejected",
    )
    grammar_exhaustion.add_argument("--priority", type=Path, required=True)
    grammar_exhaustion.add_argument("--formal-controls", type=Path, required=True)
    grammar_exhaustion.add_argument("--q-operator", type=Path, required=True)
    grammar_exhaustion.add_argument("--output", type=Path, required=True)
    action_adm = subparsers.add_parser(
        "action-adm",
        help="Compile a bounded covariant action into a verified termwise 3+1 ADM IR",
    )
    action_adm.add_argument("--spec", type=Path, required=True)
    action_adm.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    action_adm.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    action_adm.add_argument("--output", type=Path, required=True)
    action_legendre = subparsers.add_parser(
        "action-legendre",
        help="Compile a bounded action into an exact local Legendre-Hessian IR",
    )
    action_legendre.add_argument("--spec", type=Path, required=True)
    action_legendre.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    action_legendre.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    action_legendre.add_argument("--output", type=Path, required=True)
    action_dirac = subparsers.add_parser(
        "action-dirac",
        help="Compile a bounded action into canonical and distributed Dirac-closure IR",
    )
    action_dirac.add_argument("--spec", type=Path, required=True)
    action_dirac.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    action_dirac.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    action_dirac.add_argument("--output", type=Path, required=True)
    action_stability = subparsers.add_parser(
        "action-stability",
        help="Compile a bounded action into parameter-domain-bound Hamiltonian and principal-symbol stability IR",
    )
    action_stability.add_argument("--spec", type=Path, required=True)
    action_stability.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    action_stability.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    action_stability.add_argument("--output", type=Path, required=True)
    action_principal = subparsers.add_parser(
        "action-principal",
        help="Compile an action-hash-bound gauge-reduced physical principal-symbol IR",
    )
    action_principal.add_argument("--spec", type=Path, required=True)
    action_principal.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    action_principal.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    action_principal.add_argument("--output", type=Path, required=True)
    action_hamiltonian = subparsers.add_parser(
        "action-hamiltonian",
        help="Compile a constraint/gauge-reduced physical Hamiltonian IR on the action hash chain",
    )
    action_hamiltonian.add_argument("--spec", type=Path, required=True)
    action_hamiltonian.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    action_hamiltonian.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    action_hamiltonian.add_argument("--output", type=Path, required=True)
    action_vary = subparsers.add_parser(
        "action-vary-scalar",
        help="Compile and execute a Cadabra scalar-field variation from bounded action IR",
    )
    action_vary.add_argument("--spec", type=Path, required=True)
    action_vary.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    action_vary.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    action_vary.add_argument("--output", type=Path, required=True)
    proca_vary = subparsers.add_parser(
        "action-vary-proca",
        help="Compile and execute a Cadabra Proca-field variation from bounded action IR",
    )
    proca_vary.add_argument("--spec", type=Path, required=True)
    proca_vary.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    proca_vary.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    proca_vary.add_argument("--output", type=Path, required=True)
    health = subparsers.add_parser(
        "action-health",
        help="Run the fail-closed end-to-end formal gate packet for one bounded action",
    )
    health.add_argument("--spec", type=Path, required=True)
    health.add_argument(
        "--grammar", type=Path, default=Path("configs/covariant_action_grammar.json")
    )
    health.add_argument(
        "--contract", type=Path, default=Path("configs/covariant_field_contract.json")
    )
    health.add_argument("--output", type=Path, required=True)
    crosscheck = subparsers.add_parser(
        "crosscheck-v2", help="Independently cross-check a Rust Generator v2 manifest"
    )
    crosscheck.add_argument("--manifest", type=Path, required=True)
    crosscheck.add_argument("--config", type=Path, required=True)
    crosscheck.add_argument("--output", type=Path, required=True)
    knowledge = subparsers.add_parser(
        "knowledge-build",
        help="Build a provenance-preserving evidence graph from a Sigma repository",
    )
    knowledge.add_argument("--repo", type=Path, required=True)
    knowledge.add_argument("--ontology", type=Path, required=True)
    knowledge.add_argument("--database", type=Path, required=True)
    knowledge.add_argument("--summary", type=Path, required=True)
    prioritize = subparsers.add_parser(
        "formula-prioritize", help="Create a hard-gated Pareto work queue from the formula registry"
    )
    prioritize.add_argument("--database", type=Path, required=True)
    prioritize.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser(
        "survivor-audit", help="Verify every compact Generator v2 survivor record and block hash"
    )
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--survivor-dir", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    generated = subparsers.add_parser(
        "generated-prioritize", help="Build a Pareto family work queue from all exported survivors"
    )
    generated.add_argument("--manifest", type=Path, required=True)
    generated.add_argument("--survivor-dir", type=Path)
    generated.add_argument("--basis", type=Path, required=True)
    generated.add_argument("--database", type=Path, required=True)
    generated.add_argument("--max-fronts", type=int, default=8)
    generated.add_argument("--dense-report", type=Path)
    generated.add_argument("--dense-status-dir", type=Path)
    generated.add_argument("--output", type=Path, required=True)
    dense = subparsers.add_parser(
        "dense-static-gpu",
        help="GPU-screen exported survivors on a denser static-convexity lattice",
    )
    dense.add_argument("--manifest", type=Path, required=True)
    dense.add_argument("--survivor-dir", type=Path, required=True)
    dense.add_argument("--basis", type=Path, required=True)
    dense.add_argument("--config", type=Path, required=True)
    dense.add_argument("--status-dir", type=Path, required=True)
    dense.add_argument("--ambiguity-guard", type=float, default=1e-10)
    dense.add_argument("--output", type=Path, required=True)
    dense_check = subparsers.add_parser(
        "dense-static-crosscheck", help="Independently recompute deterministic GPU decisions on CPU"
    )
    dense_check.add_argument("--dense-report", type=Path, required=True)
    dense_check.add_argument("--basis", type=Path, required=True)
    dense_check.add_argument("--survivor-dir", type=Path, required=True)
    dense_check.add_argument("--status-dir", type=Path, required=True)
    dense_check.add_argument("--sample-limit", type=int, default=1024)
    dense_check.add_argument("--output", type=Path, required=True)
    equation_build = subparsers.add_parser(
        "equation-universe-build",
        help="Build a provenance-aware equation and derivation graph",
    )
    equation_build.add_argument("--seed", type=Path, required=True)
    equation_build.add_argument("--database", type=Path, required=True)
    equation_build.add_argument("--report", type=Path, required=True)
    equation_build.add_argument("--replace", action="store_true")
    equation_import = subparsers.add_parser(
        "equation-universe-import",
        help="Import a licensed structured equation bundle into an existing universe",
    )
    equation_import.add_argument("--database", type=Path, required=True)
    equation_import.add_argument("--input", type=Path, required=True)
    equation_history = subparsers.add_parser(
        "equation-universe-register-history",
        help="Register a compact Generator v2 formula space and survivor ledger",
    )
    equation_history.add_argument("--database", type=Path, required=True)
    equation_history.add_argument("--manifest", type=Path, required=True)
    equation_history.add_argument("--basis", type=Path, required=True)
    equation_history.add_argument("--survivor-dir", type=Path)
    equation_history.add_argument("--name", default="Sigma Generator v2 compact formula history")
    equation_audit = subparsers.add_parser(
        "equation-universe-audit",
        help="Audit equation provenance, dimensions, equivalence edges, and derivation proofs",
    )
    equation_audit.add_argument("--database", type=Path, required=True)
    equation_audit.add_argument("--output", type=Path, required=True)
    equation_classify = subparsers.add_parser(
        "equation-universe-classify",
        help="Classify one structured candidate against semantic and structural prior art",
    )
    equation_classify.add_argument("--database", type=Path, required=True)
    equation_classify.add_argument("--record", type=Path, required=True)
    equation_classify.add_argument("--output", type=Path, required=True)
    physics_compile = subparsers.add_parser(
        "physics-compile",
        help="Compile a typed cross-domain concept graph and report missing proof adapters",
    )
    physics_compile.add_argument("--program", type=Path, required=True)
    physics_compile.add_argument("--output", type=Path, required=True)
    scalar_tensor_compile = subparsers.add_parser(
        "scalar-tensor-compile",
        help="Compile a normalized symbolic Horndeski L2-L4 function family",
    )
    scalar_tensor_compile.add_argument("--spec", type=Path, required=True)
    scalar_tensor_compile.add_argument("--output", type=Path, required=True)
    flrw_background = subparsers.add_parser(
        "flrw-background-certify",
        help="Interval-certify an on-shell FLRW trajectory and perturbative health patch",
    )
    flrw_background.add_argument("--ir", type=Path, required=True)
    flrw_background.add_argument("--config", type=Path, required=True)
    flrw_background.add_argument("--output", type=Path, required=True)
    flrw_campaign = subparsers.add_parser(
        "flrw-background-campaign",
        help="Enumerate a scalar-tensor family and interval-certify every eligible FLRW candidate",
    )
    flrw_campaign.add_argument("--ir", type=Path, required=True)
    flrw_campaign.add_argument("--config", type=Path, required=True)
    flrw_campaign.add_argument("--output", type=Path, required=True)
    cubic_bssn_domain = subparsers.add_parser(
        "cubic-bssn-domain-campaign",
        help="Certify uniform arbitrary-local-jet BSSN principal domains for screened cubic candidates",
    )
    cubic_bssn_domain.add_argument("--ir", type=Path, required=True)
    cubic_bssn_domain.add_argument("--campaign", type=Path, required=True)
    cubic_bssn_domain.add_argument("--config", type=Path, required=True)
    cubic_bssn_domain.add_argument("--output", type=Path, required=True)
    quartic_linear_x = subparsers.add_parser(
        "quartic-linear-x-symbol-campaign",
        help="Bind specialized linear-X quartic mutations to exact 11-by-11 principal symbols",
    )
    quartic_linear_x.add_argument("--ir", type=Path, required=True)
    quartic_linear_x.add_argument("--config", type=Path, required=True)
    quartic_linear_x.add_argument("--output", type=Path, required=True)
    quartic_symmetrizer = subparsers.add_parser(
        "quartic-symmetrizer-domain-campaign",
        help="Certify complete modified-harmonic strong-hyperbolicity boxes for bound linear-X quartic candidates",
    )
    quartic_symmetrizer.add_argument("--ir", type=Path, required=True)
    quartic_symmetrizer.add_argument("--bindings", type=Path, required=True)
    quartic_symmetrizer.add_argument("--config", type=Path, required=True)
    quartic_symmetrizer.add_argument("--output", type=Path, required=True)
    quartic_dirac = subparsers.add_parser(
        "quartic-dirac-hamiltonian-campaign",
        help="Certify local on-shell ADM/Dirac closure and quadratic Hamiltonians for bound quartic candidates",
    )
    quartic_dirac.add_argument("--ir", type=Path, required=True)
    quartic_dirac.add_argument("--bindings", type=Path, required=True)
    quartic_dirac.add_argument("--symmetrizers", type=Path, required=True)
    quartic_dirac.add_argument("--config", type=Path, required=True)
    quartic_dirac.add_argument("--output", type=Path, required=True)
    quartic_energy = subparsers.add_parser(
        "quartic-linearized-energy-campaign",
        help="Certify finite-horizon all-wavenumber physical energies for the quartic FLRW branches",
    )
    quartic_energy.add_argument("--dirac-campaign", type=Path, required=True)
    quartic_energy.add_argument("--config", type=Path, required=True)
    quartic_energy.add_argument("--output", type=Path, required=True)
    quartic_reconstruction = subparsers.add_parser(
        "quartic-constraint-reconstruction-campaign",
        help="Reconstruct and bound linear lapse/shift auxiliaries from quartic physical modes",
    )
    quartic_reconstruction.add_argument("--dirac-campaign", type=Path, required=True)
    quartic_reconstruction.add_argument("--energy-campaign", type=Path, required=True)
    quartic_reconstruction.add_argument("--config", type=Path, required=True)
    quartic_reconstruction.add_argument("--output", type=Path, required=True)
    quartic_auxiliary_time = subparsers.add_parser(
        "quartic-auxiliary-time-campaign",
        help="Bound time derivatives of reconstructed quartic lapse/shift auxiliaries",
    )
    quartic_auxiliary_time.add_argument("--dirac-campaign", type=Path, required=True)
    quartic_auxiliary_time.add_argument("--energy-campaign", type=Path, required=True)
    quartic_auxiliary_time.add_argument(
        "--reconstruction-campaign", type=Path, required=True
    )
    quartic_auxiliary_time.add_argument("--config", type=Path, required=True)
    quartic_auxiliary_time.add_argument("--output", type=Path, required=True)
    quartic_moser = subparsers.add_parser(
        "quartic-quasilinear-moser-campaign",
        help="Bound C4 jet derivatives of the quartic quasilinear companion coefficients",
    )
    quartic_moser.add_argument("--symmetrizer-campaign", type=Path, required=True)
    quartic_moser.add_argument("--auxiliary-time-campaign", type=Path, required=True)
    quartic_moser.add_argument("--config", type=Path, required=True)
    quartic_moser.add_argument("--output", type=Path, required=True)
    quartic_first_order = subparsers.add_parser(
        "quartic-first-order-reduction-campaign",
        help="Construct the exact 55-variable physical-space first-order quartic reduction",
    )
    quartic_first_order.add_argument(
        "--symmetrizer-campaign", type=Path, required=True
    )
    quartic_first_order.add_argument("--moser-campaign", type=Path, required=True)
    quartic_first_order.add_argument("--config", type=Path, required=True)
    quartic_first_order.add_argument("--output", type=Path, required=True)
    quartic_geometric_jet = subparsers.add_parser(
        "quartic-geometric-jet-campaign",
        help="Bind the 55-variable quartic state to exact nonlinear covariant geometry",
    )
    quartic_geometric_jet.add_argument(
        "--first-order-campaign", type=Path, required=True
    )
    quartic_geometric_jet.add_argument("--config", type=Path, required=True)
    quartic_geometric_jet.add_argument("--output", type=Path, required=True)
    quartic_nonlinear_evolution = subparsers.add_parser(
        "quartic-nonlinear-evolution-campaign",
        help="Generate and solve the exact local gauge-fixed nonlinear quartic source",
    )
    quartic_nonlinear_evolution.add_argument(
        "--geometric-campaign", type=Path, required=True
    )
    quartic_nonlinear_evolution.add_argument("--config", type=Path, required=True)
    quartic_nonlinear_evolution.add_argument("--output", type=Path, required=True)
    quartic_nonquasilinear_pde = subparsers.add_parser(
        "quartic-nonquasilinear-pde-campaign",
        help="Lift the quartic companion symmetrizer to the full 55-state nonlinear PDE",
    )
    quartic_nonquasilinear_pde.add_argument(
        "--symmetrizer-campaign", type=Path, required=True
    )
    quartic_nonquasilinear_pde.add_argument(
        "--moser-campaign", type=Path, required=True
    )
    quartic_nonquasilinear_pde.add_argument(
        "--first-order-campaign", type=Path, required=True
    )
    quartic_nonquasilinear_pde.add_argument(
        "--geometric-campaign", type=Path, required=True
    )
    quartic_nonquasilinear_pde.add_argument(
        "--nonlinear-campaign", type=Path, required=True
    )
    quartic_nonquasilinear_pde.add_argument("--config", type=Path, required=True)
    quartic_nonquasilinear_pde.add_argument("--output", type=Path, required=True)
    quartic_coordinate_tube = subparsers.add_parser(
        "quartic-coordinate-jet-tube-campaign",
        help="Place a uniform coordinate 2-jet tube inside the quartic covariant box",
    )
    quartic_coordinate_tube.add_argument(
        "--nonquasilinear-pde-campaign", type=Path, required=True
    )
    quartic_coordinate_tube.add_argument("--config", type=Path, required=True)
    quartic_coordinate_tube.add_argument("--output", type=Path, required=True)
    quartic_euler_remainder = subparsers.add_parser(
        "quartic-euler-remainder-majorant-campaign",
        help="Bound every acceleration-independent quartic Euler remainder term",
    )
    quartic_euler_remainder.add_argument(
        "--nonquasilinear-pde-campaign", type=Path, required=True
    )
    quartic_euler_remainder.add_argument(
        "--coordinate-tube-campaign", type=Path, required=True
    )
    quartic_euler_remainder.add_argument("--config", type=Path, required=True)
    quartic_euler_remainder.add_argument("--output", type=Path, required=True)
    quartic_solved_source = subparsers.add_parser(
        "quartic-solved-source-moser-campaign",
        help="Compose the inverse time block with the Euler remainder through order four",
    )
    quartic_solved_source.add_argument("--moser-campaign", type=Path, required=True)
    quartic_solved_source.add_argument(
        "--nonquasilinear-pde-campaign", type=Path, required=True
    )
    quartic_solved_source.add_argument(
        "--coordinate-tube-campaign", type=Path, required=True
    )
    quartic_solved_source.add_argument(
        "--euler-remainder-campaign", type=Path, required=True
    )
    quartic_solved_source.add_argument("--config", type=Path, required=True)
    quartic_solved_source.add_argument("--output", type=Path, required=True)
    quartic_full_symmetrizer = subparsers.add_parser(
        "quartic-full-symmetrizer-moser-campaign",
        help="Differentiate the complete lifted 55-state Riesz symmetrizer",
    )
    quartic_full_symmetrizer.add_argument(
        "--symmetrizer-campaign", type=Path, required=True
    )
    quartic_full_symmetrizer.add_argument("--moser-campaign", type=Path, required=True)
    quartic_full_symmetrizer.add_argument(
        "--nonquasilinear-pde-campaign", type=Path, required=True
    )
    quartic_full_symmetrizer.add_argument(
        "--coordinate-tube-campaign", type=Path, required=True
    )
    quartic_full_symmetrizer.add_argument(
        "--solved-source-campaign", type=Path, required=True
    )
    quartic_full_symmetrizer.add_argument("--config", type=Path, required=True)
    quartic_full_symmetrizer.add_argument("--output", type=Path, required=True)
    quartic_symmetrizer_symbol = subparsers.add_parser(
        "quartic-symmetrizer-symbol-moser-campaign",
        help="Bound mixed state and direction derivatives of the lifted K55 symbol",
    )
    quartic_symmetrizer_symbol.add_argument(
        "--symmetrizer-campaign", type=Path, required=True
    )
    quartic_symmetrizer_symbol.add_argument("--moser-campaign", type=Path, required=True)
    quartic_symmetrizer_symbol.add_argument(
        "--nonquasilinear-pde-campaign", type=Path, required=True
    )
    quartic_symmetrizer_symbol.add_argument(
        "--coordinate-tube-campaign", type=Path, required=True
    )
    quartic_symmetrizer_symbol.add_argument(
        "--solved-source-campaign", type=Path, required=True
    )
    quartic_symmetrizer_symbol.add_argument(
        "--full-symmetrizer-campaign", type=Path, required=True
    )
    quartic_symmetrizer_symbol.add_argument("--config", type=Path, required=True)
    quartic_symmetrizer_symbol.add_argument("--output", type=Path, required=True)
    quartic_homogeneous_frequency = subparsers.add_parser(
        "quartic-homogeneous-frequency-symbol-campaign",
        help="Convert unit-direction K55 bounds to homogeneous xi-derivative bounds",
    )
    quartic_homogeneous_frequency.add_argument(
        "--symbol-campaign", type=Path, required=True
    )
    quartic_homogeneous_frequency.add_argument("--config", type=Path, required=True)
    quartic_homogeneous_frequency.add_argument("--output", type=Path, required=True)
    dhost_compile = subparsers.add_parser(
        "dhost-pack-compile",
        help="Compile a reduced rank-one quadratic DHOST kinetic family",
    )
    dhost_compile.add_argument("--spec", type=Path, required=True)
    dhost_compile.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "search":
        compiler = TheoryCompiler.from_path(args.config)
        registry = compiler.run(config_path=args.config)
        json_path, markdown_path = write_registry(registry, args.output)
        counts = registry["counts"]
        print(f"enumerated={counts['total']}")
        print(f"rejected_pre_covariant={counts['rejected_pre_covariant']}")
        print(f"requires_covariant_lift={counts['requires_covariant_lift']}")
        print(f"registry={json_path}")
        print(f"summary={markdown_path}")
        return 0
    if args.command == "validate":
        report = run_validation()
        path = write_validation(report, args.output)
        print(f"checks={report['counts']['total']}")
        print(f"passed={report['counts']['passed']}")
        print(f"failed={report['counts']['failed']}")
        print(f"report={path}")
        return 0 if report["counts"]["failed"] == 0 else 1
    if args.command == "reference":
        eligibility = audit_theory_observation_eligibility(
            args.health, args.policy, mode="known_answer_reference"
        )
        report = run_relativity_reference_suite(eligibility)
        json_path, markdown_path = write_relativity_report(report, args.output)
        print(f"golden_checks={report['counts']['golden_total']}")
        print(f"passed={report['counts']['passed']}")
        print(f"failed={report['counts']['failed']}")
        print(f"blocked={report['counts']['blocked']}")
        print(f"formal_eligibility={eligibility['status']}")
        print(f"galaxy_control={report['galaxy_negative_control']['status']}")
        print(f"report={json_path}")
        print(f"summary={markdown_path}")
        return (
            0
            if report["counts"]["failed"] == 0
            and report["counts"]["blocked"] == 0
            else 1
        )
    if args.command == "formal-controls":
        report = run_formal_control_suite(args.contract)
        json_path, markdown_path = write_formal_report(report, args.output)
        print(f"formal_checks={report['counts']['total']}")
        print(f"passed={report['counts']['passed']}")
        print(f"failed={report['counts']['failed']}")
        print(f"cadabra_available={report['backends']['cadabra2']['available']}")
        print(f"report={json_path}")
        print(f"summary={markdown_path}")
        return 0 if report["counts"]["failed"] == 0 else 1
    if args.command == "observation-audit":
        result = audit_theory_observation_eligibility(
            args.health, args.policy, mode=args.mode
        )
        path = write_observation_eligibility(result, args.output)
        print(f"status={result['status']}")
        print(f"mode={result['mode']}")
        print(f"observational_dataset_opened={result['observational_dataset_opened']}")
        print(f"errors={len(result['errors'])}")
        print(f"report={path}")
        return 0 if result["status"] == "eligible" else 1
    if args.command == "galaxy-protocol-audit":
        result = audit_galaxy_observable_protocol(args.protocol, args.policy)
        path = write_observation_eligibility(result, args.output)
        print(f"status={result['status']}")
        print(f"observational_dataset_opened={result['observational_dataset_opened']}")
        print(f"formula_search_authorized={result.get('formula_search_authorized', False)}")
        print(f"errors={len(result['errors'])}")
        print(f"report={path}")
        return 0 if result["status"] == "pass" else 1
    if args.command == "action-compile":
        result = compile_action_file(args.spec, args.grammar, args.contract)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"valid={result['valid']}")
        print(f"content_sha256={result['content_sha256']}")
        print(f"errors={len(result['errors'])}")
        print(f"ir={args.output}")
        return 0 if result["valid"] else 1
    if args.command == "action-static-dictionary":
        action_ir = compile_action_file(args.spec, args.grammar, args.contract)
        result = compile_static_dictionary_ir(action_ir, args.generator_expression)
        path = write_static_artifact(result, args.output)
        print(f"status={result['status']}")
        print(f"input_action_sha256={result.get('input_action_sha256')}")
        print(
            "generator_decision="
            f"{(result.get('generator_expression_classification') or {}).get('decision')}"
        )
        print(f"static_dictionary_ir={path}")
        return 0 if result["status"] == "pass" else 1
    if args.command == "static-lift-audit":
        dictionary_ir = json.loads(args.dictionary.read_text(encoding="utf-8"))
        priority_report = json.loads(args.priority.read_text(encoding="utf-8"))
        result = audit_priority_static_lift(priority_report, dictionary_ir)
        path = write_static_artifact(result, args.output)
        print(f"queue_count={result['queue_count']}")
        print(f"currently_liftable={result['currently_liftable_count']}")
        print(f"q_backend_queue={result['q_backend_queue_count']}")
        print(f"report={path}")
        return 0 if result["status"] == "pass" else 1
    if args.command == "covariant-export":
        report_path = export_covariant_candidate_files(
            args.priority,
            load_action_grammar(args.grammar),
            load_field_contract(args.contract),
            args.output,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"representable={report['representable_count']}")
        print(
            "formal_backend_queue="
            f"{len(report.get('formal_backend_queue', []))}"
        )
        print(f"report={report_path}")
        return 0 if report["status"] == "pass" else 1
    if args.command == "grammar-exhaustion":
        report = audit_static_grammar_exhaustion(
            args.priority, args.formal_controls, args.q_operator
        )
        path = write_grammar_exhaustion(report, args.output)
        print(f"status={report['status']}")
        print(f"queue_count={report['queue_count']}")
        print(f"decision_counts={report['decision_counts']}")
        print(f"report={path}")
        return 0 if report["status"] == "exhausted_no_admissible_family" else 1
    if args.command == "action-adm":
        action_ir = compile_action_file(args.spec, args.grammar, args.contract)
        if action_ir["valid"]:
            formal = run_formal_control_suite(args.contract, Path.cwd())
            control_status = {item["name"]: item["status"] == "pass" for item in formal["checks"]}
        else:
            control_status = None
        result = compile_adm_ir(action_ir, control_status)
        path = write_adm_ir(result, args.output)
        print(f"status={result['status']}")
        print(f"templates_complete={result.get('term_templates_complete', False)}")
        print(f"input_action_sha256={result.get('input_action_sha256')}")
        print(f"adm_ir_sha256={result.get('content_sha256')}")
        print(f"adm_ir={path}")
        return 0 if result["status"] == "pass" else 1
    if args.command == "action-legendre":
        action_ir = compile_action_file(args.spec, args.grammar, args.contract)
        if action_ir["valid"]:
            formal = run_formal_control_suite(args.contract, Path.cwd())
            control_status = {item["name"]: item["status"] == "pass" for item in formal["checks"]}
        else:
            control_status = None
        adm_ir = compile_adm_ir(action_ir, control_status)
        adm_stem = (
            f"{args.output.stem.removesuffix('-legendre-ir')}-adm-ir"
            if args.output.stem.endswith("-legendre-ir")
            else f"{args.output.stem}.adm-ir"
        )
        adm_path = write_adm_ir(adm_ir, args.output.with_name(adm_stem + args.output.suffix))
        result = compile_legendre_ir(action_ir, adm_ir)
        path = write_legendre_ir(result, args.output)
        print(f"status={result['status']}")
        print(f"generic_rank={result.get('generic_hessian_rank')}")
        print(f"generic_nullity={result.get('generic_hessian_nullity')}")
        print(f"input_action_sha256={result.get('input_action_sha256')}")
        print(f"adm_ir={adm_path}")
        print(f"legendre_ir={path}")
        return 0 if result["status"] == "pass" else 1
    if args.command == "action-dirac":
        action_ir = compile_action_file(args.spec, args.grammar, args.contract)
        if action_ir["valid"]:
            formal = run_formal_control_suite(args.contract, Path.cwd())
            control_status = {item["name"]: item["status"] == "pass" for item in formal["checks"]}
        else:
            control_status = None
        adm_ir = compile_adm_ir(action_ir, control_status)
        legendre_ir = compile_legendre_ir(action_ir, adm_ir)
        result = compile_dirac_ir(action_ir, adm_ir, legendre_ir, control_status)
        base_stem = args.output.stem.removesuffix("-dirac-ir")
        adm_path = write_adm_ir(
            adm_ir, args.output.with_name(f"{base_stem}-adm-ir{args.output.suffix}")
        )
        legendre_path = write_legendre_ir(
            legendre_ir,
            args.output.with_name(f"{base_stem}-legendre-ir{args.output.suffix}"),
        )
        path = write_dirac_ir(result, args.output)
        closure = result.get("distributed_constraint_closure", {})
        print(f"status={result['status']}")
        print(f"family={closure.get('family')}")
        print(f"physical_dof={closure.get('constraint_surface_rank', {}).get('physical_dof')}")
        print(f"adm_ir={adm_path}")
        print(f"legendre_ir={legendre_path}")
        print(f"dirac_ir={path}")
        return 0 if result["status"] == "pass" else 1
    if args.command == "action-stability":
        action_ir = compile_action_file(args.spec, args.grammar, args.contract)
        if action_ir["valid"]:
            formal = run_formal_control_suite(args.contract, Path.cwd())
            control_status = {item["name"]: item["status"] == "pass" for item in formal["checks"]}
        else:
            control_status = None
        adm_ir = compile_adm_ir(action_ir, control_status)
        legendre_ir = compile_legendre_ir(action_ir, adm_ir)
        dirac_ir = compile_dirac_ir(action_ir, adm_ir, legendre_ir, control_status)
        result = compile_stability_ir(action_ir, dirac_ir, control_status)
        base_stem = args.output.stem.removesuffix("-stability-ir")
        adm_path = write_adm_ir(
            adm_ir, args.output.with_name(f"{base_stem}-adm-ir{args.output.suffix}")
        )
        legendre_path = write_legendre_ir(
            legendre_ir,
            args.output.with_name(f"{base_stem}-legendre-ir{args.output.suffix}"),
        )
        dirac_path = write_dirac_ir(
            dirac_ir,
            args.output.with_name(f"{base_stem}-dirac-ir{args.output.suffix}"),
        )
        path = write_stability_ir(result, args.output)
        print(f"status={result['status']}")
        print(f"family={result.get('family')}")
        print(f"parameter_conditions={result.get('condition_certificate', {}).get('status')}")
        print(f"physical_hamiltonian={result.get('physical_hamiltonian', {}).get('status')}")
        print(f"principal_symbol={result.get('principal_symbol', {}).get('status')}")
        print(f"adm_ir={adm_path}")
        print(f"legendre_ir={legendre_path}")
        print(f"dirac_ir={dirac_path}")
        print(f"stability_ir={path}")
        return 0 if result["status"] == "pass" else 1
    if args.command == "action-principal":
        action_ir = compile_action_file(args.spec, args.grammar, args.contract)
        if action_ir["valid"]:
            formal = run_formal_control_suite(args.contract, Path.cwd())
            control_status = {item["name"]: item["status"] == "pass" for item in formal["checks"]}
        else:
            control_status = None
        adm_ir = compile_adm_ir(action_ir, control_status)
        legendre_ir = compile_legendre_ir(action_ir, adm_ir)
        dirac_ir = compile_dirac_ir(action_ir, adm_ir, legendre_ir, control_status)
        stability_ir = compile_stability_ir(action_ir, dirac_ir, control_status)
        result = compile_physical_principal_ir(action_ir, dirac_ir, stability_ir)
        base_stem = args.output.stem.removesuffix("-principal-ir")
        adm_path = write_adm_ir(
            adm_ir, args.output.with_name(f"{base_stem}-adm-ir{args.output.suffix}")
        )
        legendre_path = write_legendre_ir(
            legendre_ir,
            args.output.with_name(f"{base_stem}-legendre-ir{args.output.suffix}"),
        )
        dirac_path = write_dirac_ir(
            dirac_ir,
            args.output.with_name(f"{base_stem}-dirac-ir{args.output.suffix}"),
        )
        stability_path = write_stability_ir(
            stability_ir,
            args.output.with_name(f"{base_stem}-stability-ir{args.output.suffix}"),
        )
        path = write_physical_principal_ir(result, args.output)
        reduction = result.get("gauge_reduction_certificate", {})
        print(f"status={result['status']}")
        print(f"family={result.get('family')}")
        print(f"physical_modes={reduction.get('retained_mode_count')}")
        print(f"characteristic_speed_squared={result.get('characteristic_speed_squared')}")
        print(f"adm_ir={adm_path}")
        print(f"legendre_ir={legendre_path}")
        print(f"dirac_ir={dirac_path}")
        print(f"stability_ir={stability_path}")
        print(f"principal_ir={path}")
        return 0 if result["status"] == "pass" else 1
    if args.command == "action-hamiltonian":
        action_ir = compile_action_file(args.spec, args.grammar, args.contract)
        if action_ir["valid"]:
            formal = run_formal_control_suite(args.contract, Path.cwd())
            control_status = {item["name"]: item["status"] == "pass" for item in formal["checks"]}
        else:
            control_status = None
        adm_ir = compile_adm_ir(action_ir, control_status)
        legendre_ir = compile_legendre_ir(action_ir, adm_ir)
        dirac_ir = compile_dirac_ir(action_ir, adm_ir, legendre_ir, control_status)
        stability_ir = compile_stability_ir(action_ir, dirac_ir, control_status)
        principal_ir = compile_physical_principal_ir(action_ir, dirac_ir, stability_ir)
        result = compile_physical_hamiltonian_ir(
            action_ir, dirac_ir, stability_ir, principal_ir
        )
        base_stem = args.output.stem.removesuffix("-hamiltonian-ir")
        adm_path = write_adm_ir(
            adm_ir, args.output.with_name(f"{base_stem}-adm-ir{args.output.suffix}")
        )
        legendre_path = write_legendre_ir(
            legendre_ir,
            args.output.with_name(f"{base_stem}-legendre-ir{args.output.suffix}"),
        )
        dirac_path = write_dirac_ir(
            dirac_ir,
            args.output.with_name(f"{base_stem}-dirac-ir{args.output.suffix}"),
        )
        stability_path = write_stability_ir(
            stability_ir,
            args.output.with_name(f"{base_stem}-stability-ir{args.output.suffix}"),
        )
        principal_path = write_physical_principal_ir(
            principal_ir,
            args.output.with_name(f"{base_stem}-principal-ir{args.output.suffix}"),
        )
        path = write_physical_hamiltonian_ir(result, args.output)
        print(f"status={result['status']}")
        print(f"family={result.get('family')}")
        print(f"physical_modes={result.get('physical_mode_count')}")
        print(f"positivity={result.get('positivity_certificate', {}).get('status')}")
        print(
            "generic_nonlinear_total_energy="
            f"{result.get('generic_nonlinear_total_energy', {}).get('status')}"
        )
        print(f"adm_ir={adm_path}")
        print(f"legendre_ir={legendre_path}")
        print(f"dirac_ir={dirac_path}")
        print(f"stability_ir={stability_path}")
        print(f"principal_ir={principal_path}")
        print(f"hamiltonian_ir={path}")
        return 0 if result["status"] == "pass" else 1
    if args.command == "action-vary-scalar":
        result = vary_scalar_action_file(
            args.spec,
            args.grammar,
            args.contract,
            args.output,
            project_root=Path.cwd(),
        )
        print(f"status={result['status']}")
        print(f"action_ir={result['action_ir']}")
        print(f"script={result['script']}")
        if result.get("result_path"):
            print(f"result={result['result_path']}")
        return 0 if result["status"] == "pass" else 1
    if args.command == "action-vary-proca":
        result = vary_proca_action_file(
            args.spec,
            args.grammar,
            args.contract,
            args.output,
            project_root=Path.cwd(),
        )
        print(f"status={result['status']}")
        print(f"action_ir={result['action_ir']}")
        print(f"script={result['script']}")
        if result.get("result_path"):
            print(f"result={result['result_path']}")
        return 0 if result["status"] == "pass" else 1
    if args.command == "action-health":
        result = analyze_action_health(
            args.spec,
            args.grammar,
            args.contract,
            args.output,
            project_root=Path.cwd(),
        )
        print(f"status={result['status']}")
        print(f"family={result.get('family', 'invalid')}")
        print(f"promotion_allowed={result['promotion_allowed']}")
        print(f"report={result['report_path']}")
        return 0 if result["status"] in {"pass", "control_pass"} else 1
    if args.command == "crosscheck-v2":
        report = crosscheck_manifest(args.manifest, args.config)
        path = write_crosscheck(report, args.output)
        print(f"accounting_pass={report['all_accounting_checks_pass']}")
        print(f"sample_count={report['sample_count']}")
        print(f"cross_language_pass={report['all_cross_language_samples_agree']}")
        print(
            "python_convexity_pass="
            f"{report['all_recorded_survivors_pass_python_sampled_convexity']}"
        )
        print(f"rejection_witnesses_pass={report['all_rejection_witnesses_agree']}")
        print(f"report={path}")
        return (
            0
            if report["all_accounting_checks_pass"]
            and report["all_cross_language_samples_agree"]
            and report["all_recorded_survivors_pass_python_sampled_convexity"]
            and report["all_rejection_witnesses_agree"]
            else 1
        )
    if args.command == "knowledge-build":
        ontology = GateOntology.from_path(args.ontology)
        summary = KnowledgeBuilder(args.repo, ontology).build(args.database, args.summary)
        print(f"database={args.database}")
        print(f"integrity={summary['integrity_check']}")
        print(f"test_functions={summary['counts'].get('test_functions', 0)}")
        print(f"protocols={summary['counts'].get('protocols', 0)}")
        print(f"results={summary['counts'].get('results', 0)}")
        print(f"formulas={summary['counts'].get('formulas', 0)}")
        print(f"summary={args.summary}")
        return 0 if summary["integrity_check"] == "ok" else 1
    if args.command == "formula-prioritize":
        report = prioritize_registered_formulas(args.database, args.output)
        print(f"eligible={report['eligible_count']}")
        print(f"excluded={report['excluded_count']}")
        print(f"pareto_front_one={len(report['front_one'])}")
        print(f"report={args.output}")
        return 0
    if args.command == "survivor-audit":
        report = audit_survivor_export(args.manifest, args.output, args.survivor_dir)
        print(f"blocks={report['block_count']}")
        print(f"records={report['record_count']}")
        print(f"all_checks_pass={report['all_checks_pass']}")
        print(f"report={args.output}")
        return 0 if report["all_checks_pass"] else 1
    if args.command == "generated-prioritize":
        report = prioritize_generated_survivors(
            args.manifest,
            args.basis,
            args.database,
            args.output,
            args.survivor_dir,
            args.max_fronts,
            args.dense_report,
            args.dense_status_dir,
        )
        print(f"survivors={report['survivor_count']}")
        print(f"families={report['family_count']}")
        print(f"pareto_fronts={report['pareto_front_count']}")
        print(f"work_queue={len(report['work_queue'])}")
        print(f"report={args.output}")
        return 0
    if args.command == "dense-static-gpu":
        report = run_dense_gpu_screen(
            args.manifest,
            args.basis,
            args.config,
            args.survivor_dir,
            args.status_dir,
            args.output,
            args.ambiguity_guard,
        )
        print(f"grid_points={report['grid_point_count']}")
        print(f"reject={report['counts']['reject']}")
        print(f"pass={report['counts']['pass']}")
        print(f"ambiguous={report['counts']['ambiguous']}")
        print(f"accounting_pass={report['accounting_pass']}")
        print(f"report={args.output}")
        return 0 if report["accounting_pass"] else 1
    if args.command == "dense-static-crosscheck":
        report = crosscheck_dense_gpu_screen(
            args.dense_report,
            args.basis,
            args.survivor_dir,
            args.status_dir,
            args.output,
            args.sample_limit,
        )
        print(f"samples={report['sample_count']}")
        print(f"cpu_gpu_agree={report['all_cpu_gpu_samples_agree']}")
        print(f"status_hashes_pass={report['all_status_file_hashes_pass']}")
        print(f"report={args.output}")
        return (
            0
            if report["all_cpu_gpu_samples_agree"] and report["all_status_file_hashes_pass"]
            else 1
        )
    if args.command == "equation-universe-build":
        result = build_equation_universe(
            args.seed, args.database, args.report, replace=args.replace
        )
        print(f"database={args.database}")
        print(f"equations={result['audit']['counts']['equations']}")
        print(f"derivations={result['audit']['counts']['derivations']}")
        print(f"verified={result['audit']['derivation_proofs'].get('verified', 0)}")
        print(f"rejected={len(result['import']['rejected'])}")
        print(f"report={args.report}")
        return 0 if result["audit"]["passed"] and not result["import"]["rejected"] else 1
    if args.command == "equation-universe-import":
        result = EquationUniverse(args.database).import_file(args.input)
        print(f"sources={result['sources']}")
        print(f"equations={result['equations']}")
        print(f"derivations={result['derivations']}")
        print(f"rejected={len(result['rejected'])}")
        return 0 if not result["rejected"] else 1
    if args.command == "equation-universe-register-history":
        result = EquationUniverse(args.database).register_generator_history(
            args.manifest,
            args.basis,
            args.survivor_dir,
            name=args.name,
        )
        print(f"space_id={result['space_id']}")
        print(f"protocol={result['protocol_version']}")
        print(f"processed_actions={result['processed_actions']}")
        print(f"survivors={result['survivor_count']}")
        print(f"complete_declared_space={result['complete_declared_space']}")
        return 0
    if args.command == "equation-universe-audit":
        result = EquationUniverse(args.database).audit()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"integrity={result['integrity_check']}")
        print(f"equations={result['counts']['equations']}")
        print(f"unproven_derivations={len(result['unproven_derivations'])}")
        print(f"report={args.output}")
        return 0 if result["passed"] else 1
    if args.command == "equation-universe-classify":
        record = json.loads(args.record.read_text(encoding="utf-8"))
        result = EquationUniverse(args.database).classify(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"classification={result['classification']}")
        print(f"semantic_matches={len(result['semantic_matches'])}")
        print(f"structural_matches={len(result['structural_matches'])}")
        print(f"novelty_claim_allowed={result['novelty_claim_allowed']}")
        print(f"report={args.output}")
        return 0
    if args.command == "physics-compile":
        program = json.loads(args.program.read_text(encoding="utf-8"))
        result = compile_physics_program(program)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"status={result['status']}")
        print(f"concepts={result['concept_count']}")
        print(f"declared_mutations={result['mutation_space']['declared_cardinality']}")
        print(f"missing_adapters={len(result['missing_adapters'])}")
        print(f"report={args.output}")
        return 0 if result["status"] != "reject" else 1
    if args.command == "scalar-tensor-compile":
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        result = compile_scalar_tensor_pack(spec)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"status={result['status']}")
        print(f"declared_mutations={result['mutation_space']['declared_cardinality']}")
        print(
            "l4_parent_derivative_binding="
            f"{result['capability_status']['l4_parent_derivative_binding']}"
        )
        print(f"report={args.output}")
        return 0 if result["status"] != "reject" else 1
    if args.command == "flrw-background-certify":
        ir = json.loads(args.ir.read_text(encoding="utf-8"))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = certify_flrw_background(ir, config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"status={result['status']}")
        print(f"errors={len(result['errors'])}")
        if result["status"] == "pass_interval_certified":
            print(f"steps={result['time']['steps']}")
            print(
                "constraint_max_abs="
                f"{result['uniform_certificate']['constraint_max_abs_enclosure']}"
            )
        print(f"report={args.output}")
        return 0 if result["status"] == "pass_interval_certified" else 1
    if args.command == "flrw-background-campaign":
        ir = json.loads(args.ir.read_text(encoding="utf-8"))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_flrw_background_campaign(ir, config)
        manifest_path, certificate_paths = write_flrw_background_campaign(
            result, args.output
        )
        manifest = result["manifest"]
        print(f"status={manifest['status']}")
        print(f"total={manifest['counts']['total']}")
        print(f"eligible={manifest['counts']['generalized_harmonic_eligible']}")
        print(f"certified={manifest['counts']['interval_certified']}")
        print(
            "cubic_G3_only_flrw_screened="
            f"{manifest['counts']['cubic_G3_only_flrw_screened']}"
        )
        print(
            "cubic_G3_only_flrw_rejected="
            f"{manifest['counts']['cubic_G3_only_flrw_rejected']}"
        )
        print(f"modified_harmonic_unresolved={manifest['counts']['modified_harmonic_unresolved']}")
        print(f"certificates={len(certificate_paths)}")
        print(f"report={manifest_path}")
        return (
            0
            if manifest["status"]
            == "pass_all_generalized_harmonic_candidates_interval_certified"
            else 1
        )
    if args.command == "cubic-bssn-domain-campaign":
        ir = json.loads(args.ir.read_text(encoding="utf-8"))
        flrw_campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        trajectory_certificates = {
            record["candidate_id"]: json.loads(
                (args.campaign.parent / record["certificate"]).read_text(
                    encoding="utf-8"
                )
            )
            for record in flrw_campaign.get("candidates", [])
            if record.get("status")
            == "pass_flrw_interval_cubic_weak_field_bounds_unresolved"
        }
        result = run_cubic_bssn_domain_campaign(
            ir, flrw_campaign, trajectory_certificates, config
        )
        manifest_path, certificate_paths = write_cubic_bssn_domain_campaign(
            result, args.output
        )
        manifest = result["manifest"]
        print(f"status={manifest['status']}")
        print(f"screened={manifest['counts']['screened_cubic_candidates']}")
        print(f"certified={manifest['counts']['uniform_domain_certified']}")
        print(f"rejected={manifest['counts']['rejected']}")
        print(f"certificates={len(certificate_paths)}")
        print(f"report={manifest_path}")
        return (
            0
            if manifest["status"]
            == "pass_all_screened_cubic_candidates_have_uniform_local_jet_boxes"
            else 1
        )
    if args.command == "quartic-linear-x-symbol-campaign":
        ir = json.loads(args.ir.read_text(encoding="utf-8"))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_linear_x_symbol_campaign(ir, config)
        path = write_quartic_linear_x_symbol_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(f"exactly_bound={result['counts']['exactly_bound']}")
        print(f"canonical_G2={result['counts']['canonical_G2']}")
        print(f"quadratic_kessence_G2={result['counts']['quadratic_kessence_G2']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_exact_symbol_binding_uniform_symmetrizer_unresolved"
            else 1
        )
    if args.command == "quartic-symmetrizer-domain-campaign":
        ir = json.loads(args.ir.read_text(encoding="utf-8"))
        bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_symmetrizer_domain_campaign(ir, bindings, config)
        path = write_quartic_symmetrizer_domain_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "strongly_hyperbolic="
            f"{result['counts']['uniform_local_jet_strong_hyperbolicity_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_linear_X_quartic_candidates_strongly_hyperbolic_on_local_boxes"
            else 1
        )
    if args.command == "quartic-dirac-hamiltonian-campaign":
        ir = json.loads(args.ir.read_text(encoding="utf-8"))
        bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
        symmetrizers = json.loads(args.symmetrizers.read_text(encoding="utf-8"))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_dirac_hamiltonian_campaign(
            ir, bindings, symmetrizers, config
        )
        path = write_quartic_dirac_hamiltonian_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "adm_dirac_hamiltonian_passed="
            f"{result['counts']['local_on_shell_adm_dirac_hamiltonian_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
            else 1
        )
    if args.command == "quartic-linearized-energy-campaign":
        dirac_campaign = json.loads(
            args.dirac_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_linearized_energy_campaign(dirac_campaign, config)
        path = write_quartic_linearized_energy_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "linearized_energy_passed="
            f"{result['counts']['finite_horizon_linearized_energy_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_finite_horizon_linearized_inhomogeneous_energies"
            else 1
        )
    if args.command == "quartic-constraint-reconstruction-campaign":
        dirac_campaign = json.loads(
            args.dirac_campaign.read_text(encoding="utf-8")
        )
        energy_campaign = json.loads(
            args.energy_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_constraint_reconstruction_campaign(
            dirac_campaign, energy_campaign, config
        )
        path = write_quartic_constraint_reconstruction_campaign(
            result, args.output
        )
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "constraint_reconstruction_passed="
            f"{result['counts']['linear_constraint_reconstruction_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"] == "pass_all_12_linear_constraint_reconstructions"
            else 1
        )
    if args.command == "quartic-auxiliary-time-campaign":
        dirac_campaign = json.loads(
            args.dirac_campaign.read_text(encoding="utf-8")
        )
        energy_campaign = json.loads(
            args.energy_campaign.read_text(encoding="utf-8")
        )
        reconstruction_campaign = json.loads(
            args.reconstruction_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_auxiliary_time_campaign(
            dirac_campaign, energy_campaign, reconstruction_campaign, config
        )
        path = write_quartic_auxiliary_time_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "auxiliary_time_passed="
            f"{result['counts']['linear_auxiliary_time_reconstruction_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_linear_auxiliary_time_reconstructions"
            else 1
        )
    if args.command == "quartic-quasilinear-moser-campaign":
        symmetrizer_campaign = json.loads(
            args.symmetrizer_campaign.read_text(encoding="utf-8")
        )
        auxiliary_time_campaign = json.loads(
            args.auxiliary_time_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_quasilinear_moser_campaign(
            symmetrizer_campaign, auxiliary_time_campaign, config
        )
        path = write_quartic_quasilinear_moser_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "quasilinear_coefficient_envelopes_passed="
            f"{result['counts']['quasilinear_coefficient_envelopes_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_quasilinear_coefficient_derivative_envelopes"
            else 1
        )
    if args.command == "quartic-first-order-reduction-campaign":
        symmetrizer_campaign = json.loads(
            args.symmetrizer_campaign.read_text(encoding="utf-8")
        )
        moser_campaign = json.loads(args.moser_campaign.read_text(encoding="utf-8"))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_first_order_reduction_campaign(
            symmetrizer_campaign, moser_campaign, config
        )
        path = write_quartic_first_order_reduction_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "exact_55_variable_reductions_passed="
            f"{result['counts']['exact_55_variable_reductions_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_exact_55_variable_principal_first_order_reductions"
            else 1
        )
    if args.command == "quartic-geometric-jet-campaign":
        first_order_campaign = json.loads(
            args.first_order_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_geometric_jet_campaign(first_order_campaign, config)
        path = write_quartic_geometric_jet_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "geometric_state_to_jet_maps_passed="
            f"{result['counts']['geometric_state_to_jet_maps_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_exact_nonlinear_geometric_state_to_jet_maps"
            else 1
        )
    if args.command == "quartic-nonlinear-evolution-campaign":
        geometric_campaign = json.loads(
            args.geometric_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_nonlinear_evolution_campaign(
            geometric_campaign, config
        )
        path = write_quartic_nonlinear_evolution_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "nonlinear_time_acceleration_eliminations_passed="
            f"{result['counts']['nonlinear_time_acceleration_eliminations_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations"
            else 1
        )
    if args.command == "quartic-nonquasilinear-pde-campaign":
        symmetrizer_campaign = json.loads(
            args.symmetrizer_campaign.read_text(encoding="utf-8")
        )
        moser_campaign = json.loads(args.moser_campaign.read_text(encoding="utf-8"))
        first_order_campaign = json.loads(
            args.first_order_campaign.read_text(encoding="utf-8")
        )
        geometric_campaign = json.loads(
            args.geometric_campaign.read_text(encoding="utf-8")
        )
        nonlinear_campaign = json.loads(
            args.nonlinear_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_nonquasilinear_pde_campaign(
            symmetrizer_campaign,
            moser_campaign,
            first_order_campaign,
            geometric_campaign,
            nonlinear_campaign,
            config,
        )
        path = write_quartic_nonquasilinear_pde_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "full_55_state_symmetrizer_lifts_passed="
            f"{result['counts']['full_55_state_symmetrizer_lifts_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts"
            else 1
        )
    if args.command == "quartic-coordinate-jet-tube-campaign":
        nonquasilinear_pde_campaign = json.loads(
            args.nonquasilinear_pde_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_coordinate_jet_tube_campaign(
            nonquasilinear_pde_campaign, config
        )
        path = write_quartic_coordinate_jet_tube_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "coordinate_jet_tubes_passed="
            f"{result['counts']['coordinate_jet_tubes_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes"
            else 1
        )
    if args.command == "quartic-euler-remainder-majorant-campaign":
        nonquasilinear_pde_campaign = json.loads(
            args.nonquasilinear_pde_campaign.read_text(encoding="utf-8")
        )
        coordinate_tube_campaign = json.loads(
            args.coordinate_tube_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_euler_remainder_majorant_campaign(
            nonquasilinear_pde_campaign, coordinate_tube_campaign, config
        )
        path = write_quartic_euler_remainder_majorant_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "Euler_remainder_majorants_passed="
            f"{result['counts']['Euler_remainder_majorants_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_complete_coordinate_tube_euler_remainder_majorants"
            else 1
        )
    if args.command == "quartic-solved-source-moser-campaign":
        moser_campaign = json.loads(args.moser_campaign.read_text(encoding="utf-8"))
        nonquasilinear_pde_campaign = json.loads(
            args.nonquasilinear_pde_campaign.read_text(encoding="utf-8")
        )
        coordinate_tube_campaign = json.loads(
            args.coordinate_tube_campaign.read_text(encoding="utf-8")
        )
        euler_remainder_campaign = json.loads(
            args.euler_remainder_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_solved_source_moser_campaign(
            moser_campaign,
            nonquasilinear_pde_campaign,
            coordinate_tube_campaign,
            euler_remainder_campaign,
            config,
        )
        path = write_quartic_solved_source_moser_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "solved_source_moser_envelopes_passed="
            f"{result['counts']['solved_source_moser_envelopes_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes"
            else 1
        )
    if args.command == "quartic-full-symmetrizer-moser-campaign":
        symmetrizer_campaign = json.loads(
            args.symmetrizer_campaign.read_text(encoding="utf-8")
        )
        moser_campaign = json.loads(args.moser_campaign.read_text(encoding="utf-8"))
        nonquasilinear_pde_campaign = json.loads(
            args.nonquasilinear_pde_campaign.read_text(encoding="utf-8")
        )
        coordinate_tube_campaign = json.loads(
            args.coordinate_tube_campaign.read_text(encoding="utf-8")
        )
        solved_source_campaign = json.loads(
            args.solved_source_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_full_symmetrizer_moser_campaign(
            symmetrizer_campaign,
            moser_campaign,
            nonquasilinear_pde_campaign,
            coordinate_tube_campaign,
            solved_source_campaign,
            config,
        )
        path = write_quartic_full_symmetrizer_moser_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "full_K55_C4_derivative_envelopes_passed="
            f"{result['counts']['full_K55_C4_derivative_envelopes_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_full_K55_coordinate_atom_C4_derivative_envelopes"
            else 1
        )
    if args.command == "quartic-symmetrizer-symbol-moser-campaign":
        symmetrizer_campaign = json.loads(
            args.symmetrizer_campaign.read_text(encoding="utf-8")
        )
        moser_campaign = json.loads(args.moser_campaign.read_text(encoding="utf-8"))
        nonquasilinear_pde_campaign = json.loads(
            args.nonquasilinear_pde_campaign.read_text(encoding="utf-8")
        )
        coordinate_tube_campaign = json.loads(
            args.coordinate_tube_campaign.read_text(encoding="utf-8")
        )
        solved_source_campaign = json.loads(
            args.solved_source_campaign.read_text(encoding="utf-8")
        )
        full_symmetrizer_campaign = json.loads(
            args.full_symmetrizer_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_symmetrizer_symbol_moser_campaign(
            symmetrizer_campaign,
            moser_campaign,
            nonquasilinear_pde_campaign,
            coordinate_tube_campaign,
            solved_source_campaign,
            full_symmetrizer_campaign,
            config,
        )
        path = write_quartic_symmetrizer_symbol_moser_campaign(result, args.output)
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "mixed_symbol_envelopes_passed="
            f"{result['counts']['mixed_symbol_envelopes_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_full_K55_mixed_state_direction_C4_symbol_envelopes"
            else 1
        )
    if args.command == "quartic-homogeneous-frequency-symbol-campaign":
        symbol_campaign = json.loads(
            args.symbol_campaign.read_text(encoding="utf-8")
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_quartic_homogeneous_frequency_symbol_campaign(
            symbol_campaign, config
        )
        path = write_quartic_homogeneous_frequency_symbol_campaign(
            result, args.output
        )
        print(f"status={result['status']}")
        print(f"selected={result['counts']['selected']}")
        print(
            "homogeneous_frequency_bounds_passed="
            f"{result['counts']['homogeneous_frequency_bounds_passed']}"
        )
        print(f"rejected={result['counts']['rejected']}")
        print(f"report={path}")
        return (
            0
            if result["status"]
            == "pass_all_12_full_K55_homogeneous_frequency_C4_bounds"
            else 1
        )
    if args.command == "dhost-pack-compile":
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        result = compile_reduced_dhost_pack(spec)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"status={result['status']}")
        print(f"declared_mutations={result['mutation_space']['declared_cardinality']}")
        print(f"determinant_residual={result['determinant_residual']}")
        print(f"report={args.output}")
        return 0 if result["status"] != "reject" else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
