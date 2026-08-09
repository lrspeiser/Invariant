from __future__ import annotations

from sigma_theory_compiler.g4_curved_witness import (
    generic_g4_curved_rnc_witness_control,
    generic_g4_curved_symbolic_rnc_control,
)


def test_generic_g4_curved_exact_witnesses_close_noether_and_reject_omission() -> None:
    passed, evidence = generic_g4_curved_rnc_witness_control()
    assert passed, evidence
    assert evidence["generic_all_jet_theorem"].startswith("proved_by_")
    assert len(evidence["witnesses"]) == 3
    for witness in evidence["witnesses"]:
        assert witness["curvature_nonzero"]
        assert witness["curvature_gradient_nonzero"]
        assert witness["weyl_component_nonzero"]
        assert witness["algebraic_bianchi_residuals_zero"]
        assert witness["differential_bianchi_residuals_zero"]
        assert witness["scalar_hessian_commutator_residuals_zero"]
        assert witness["contracted_bianchi_residuals"] == ["0"] * 4
        assert witness["metric_symmetry_residuals"] == ["0"] * 6
        assert witness["combined_noether_residuals"] == ["0"] * 4
        assert witness["omitted_term_rejected"]


def test_generic_g4_curved_symbolic_all_jet_noether_identity() -> None:
    passed, evidence = generic_g4_curved_symbolic_rnc_control()
    assert passed, evidence
    assert evidence["proof_kind"].startswith("exact symbolic")
    assert evidence["independent_local_data"]["total_independent_symbols"] == 345
    assert evidence["metric_symmetry_residuals"] == ["0"] * 6
    assert evidence["combined_noether_residuals"] == ["0"] * 4
    assert evidence["omitted_G4_XX_q_mu_q_nu_negative"]["rejected"]
    assert evidence["independent_backend_metric_variation"] == "unresolved"
