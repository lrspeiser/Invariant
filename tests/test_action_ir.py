from __future__ import annotations

import copy
from pathlib import Path

from sigma_theory_compiler.action_ir import compile_action_spec, load_action_grammar
from sigma_theory_compiler.formal_backend import load_field_contract

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = load_action_grammar(ROOT / "configs" / "covariant_action_grammar.json")
CONTRACT = load_field_contract(ROOT / "configs" / "covariant_field_contract.json")


def _scalar_spec(terms=None):
    return {
        "schema_version": "sigma-action-spec-1.0",
        "role": "candidate",
        "fields": ["g_mu_nu", "phi"],
        "matter_metric": "g_mu_nu",
        "terms": terms or ["EH_R", "SCALAR_X", "SCALAR_MASS"],
        "coefficients": {},
        "universal_constants": ["M_Pl", "Lambda_phi", "m_phi", "c_X"],
        "static_dictionary_status": "derived",
    }


def test_action_ir_is_deterministic_and_canonical() -> None:
    first = compile_action_spec(_scalar_spec(), GRAMMAR, CONTRACT)
    second = compile_action_spec(
        _scalar_spec(["SCALAR_MASS", "EH_R", "SCALAR_X"]), GRAMMAR, CONTRACT
    )
    assert first["valid"] and second["valid"]
    assert first["content_sha256"] == second["content_sha256"]
    assert [term["id"] for term in first["canonical"]["terms"]] == [
        "EH_R",
        "SCALAR_MASS",
        "SCALAR_X",
    ]


def test_unknown_baryonic_term_cannot_enter_ir() -> None:
    result = compile_action_spec(_scalar_spec(["EH_R", "BARYON_Z"]), GRAMMAR, CONTRACT)
    assert not result["valid"]
    assert any("unknown action terms" in error for error in result["errors"])


def test_unit_vector_requires_constraint() -> None:
    result = compile_action_spec(
        {
            "schema_version": "sigma-action-spec-1.0",
            "role": "candidate",
            "fields": ["g_mu_nu", "u_mu"],
            "matter_metric": "g_mu_nu",
            "terms": ["EH_R", "AETHER_K1"],
            "universal_constants": ["M_Pl", "L_u", "c1"],
            "static_dictionary_status": "derived",
        },
        GRAMMAR,
        CONTRACT,
    )
    assert not result["valid"]
    assert "unit timelike u_mu requires UNIT_VECTOR_CONSTRAINT" in result["errors"]


def test_complete_unit_vector_action_compiles() -> None:
    result = compile_action_spec(
        {
            "schema_version": "sigma-action-spec-1.0",
            "role": "candidate",
            "fields": ["g_mu_nu", "u_mu", "lambda_u"],
            "matter_metric": "g_mu_nu",
            "terms": ["EH_R", "AETHER_K1", "AETHER_K2", "UNIT_VECTOR_CONSTRAINT"],
            "universal_constants": ["M_Pl", "L_u", "c1", "c2"],
            "static_dictionary_status": "derived",
        },
        GRAMMAR,
        CONTRACT,
    )
    assert result["valid"], result["errors"]


def test_multiple_extra_dynamical_fields_are_rejected() -> None:
    result = compile_action_spec(
        {
            "schema_version": "sigma-action-spec-1.0",
            "role": "candidate",
            "fields": ["g_mu_nu", "phi", "A_mu"],
            "matter_metric": "g_mu_nu",
            "terms": ["EH_R", "SCALAR_X", "PROCA_F2"],
            "universal_constants": ["M_Pl", "Lambda_phi"],
            "static_dictionary_status": "derived",
        },
        GRAMMAR,
        CONTRACT,
    )
    assert not result["valid"]
    assert "action exceeds the extra-dynamical-field bound: A_mu, phi" in result["errors"]


def test_proca_library_terms_are_reserved_for_known_answer_controls() -> None:
    result = compile_action_spec(
        {
            "schema_version": "sigma-action-spec-1.0",
            "role": "candidate",
            "fields": ["g_mu_nu", "A_mu"],
            "matter_metric": "g_mu_nu",
            "terms": ["EH_R", "PROCA_F2", "PROCA_MASS"],
            "universal_constants": ["M_Pl", "m_A"],
            "static_dictionary_status": "derived",
        },
        GRAMMAR,
        CONTRACT,
    )
    assert not result["valid"]
    assert (
        "control-only terms cannot enter candidate generation: PROCA_F2, PROCA_MASS"
        in result["errors"]
    )


def test_coefficients_are_safe_and_use_only_declared_constants() -> None:
    undeclared = _scalar_spec()
    undeclared["coefficients"] = {"SCALAR_X": "new_coupling"}
    result = compile_action_spec(undeclared, GRAMMAR, CONTRACT)
    assert not result["valid"]
    assert (
        "coefficient symbols must be declared universal constants: new_coupling" in result["errors"]
    )

    executable = _scalar_spec()
    executable["coefficients"] = {"SCALAR_X": "__import__('os').system('echo unsafe')"}
    rejected = compile_action_spec(executable, GRAMMAR, CONTRACT)
    assert not rejected["valid"]
    assert any("invalid coefficient for SCALAR_X" in item for item in rejected["errors"])


def test_parameter_domain_is_canonical_safe_and_hash_bound() -> None:
    first = _scalar_spec()
    first["parameter_domain"] = {
        "positive": ["m_phi", "M_Pl", "m_phi"],
        "nonzero": ["Lambda_phi"],
        "inequalities": ["c_X > 0", "1 - c_X/2 >= 0"],
    }
    second = _scalar_spec()
    second["parameter_domain"] = {
        "positive": ["M_Pl", "m_phi"],
        "nonzero": ["Lambda_phi"],
        "inequalities": ["1-c_X/2>=0", "c_X>0"],
    }
    compiled_first = compile_action_spec(first, GRAMMAR, CONTRACT)
    compiled_second = compile_action_spec(second, GRAMMAR, CONTRACT)
    assert compiled_first["valid"], compiled_first["errors"]
    assert compiled_first["content_sha256"] == compiled_second["content_sha256"]
    assert compiled_first["canonical"]["parameter_domain"]["positive"] == [
        "M_Pl",
        "m_phi",
    ]
    assert compiled_first["canonical"]["parameter_domain"]["inequalities"] == [
        "1 - c_X/2 >= 0",
        "c_X > 0",
    ]


def test_parameter_domain_rejects_unknown_unsafe_and_contradictory_claims() -> None:
    unknown = _scalar_spec()
    unknown["parameter_domain"] = {"positive": ["not_declared"]}
    unknown_result = compile_action_spec(unknown, GRAMMAR, CONTRACT)
    assert not unknown_result["valid"]
    assert any("uses undeclared constants" in item for item in unknown_result["errors"])

    unsafe = _scalar_spec()
    unsafe["parameter_domain"] = {"inequalities": ["__import__('os').system('echo unsafe') > 0"]}
    unsafe_result = compile_action_spec(unsafe, GRAMMAR, CONTRACT)
    assert not unsafe_result["valid"]
    assert any(
        "unsupported parameter inequality syntax" in item for item in unsafe_result["errors"]
    )

    contradictory = _scalar_spec()
    contradictory["parameter_domain"] = {
        "positive": ["c_X"],
        "nonpositive": ["c_X"],
    }
    contradictory_result = compile_action_spec(contradictory, GRAMMAR, CONTRACT)
    assert not contradictory_result["valid"]
    assert any("contradictory parameter signs" in item for item in contradictory_result["errors"])

    composite = _scalar_spec()
    composite["parameter_domain"] = {
        "inequalities": ["c_X > 0", "c_X <= 0"],
    }
    composite_result = compile_action_spec(composite, GRAMMAR, CONTRACT)
    assert not composite_result["valid"]
    assert any(
        "contradictory parameter-domain claims" in item
        for item in composite_result["errors"]
    )

    cross_form = _scalar_spec()
    cross_form["parameter_domain"] = {
        "positive": ["c_X"],
        "inequalities": ["c_X < 0"],
    }
    cross_result = compile_action_spec(cross_form, GRAMMAR, CONTRACT)
    assert not cross_result["valid"]
    assert any(
        "contradictory parameter-domain claims" in item for item in cross_result["errors"]
    )


def test_background_domain_is_covariant_local_safe_and_hash_bound() -> None:
    first = _scalar_spec()
    first["background_domain"] = {
        "variables": [
            {
                "id": "Y_phi",
                "covariant_definition": "-g^{mu nu} nabla_mu(phi) nabla_nu(phi)",
                "unitary_gauge_identification": "A_star^2",
                "mass_dimension": 4,
                "nonnegative": True,
                "locally_measurable": True,
            }
        ],
        "inequalities": ["Y_phi*c_X < M_Pl**4"],
        "preservation": {
            "status": "unresolved",
            "statement": "Pointwise domain declared; evolution preservation remains open.",
            "required_controls": ["scalar_background_domain_preservation"],
        },
    }
    second = copy.deepcopy(first)
    second["background_domain"]["inequalities"] = ["Y_phi * c_X<M_Pl^4"]
    compiled_first = compile_action_spec(first, GRAMMAR, CONTRACT)
    compiled_second = compile_action_spec(second, GRAMMAR, CONTRACT)
    assert compiled_first["valid"], compiled_first["errors"]
    assert compiled_first["content_sha256"] == compiled_second["content_sha256"]
    domain = compiled_first["canonical"]["background_domain"]
    assert domain["variables"][0]["id"] == "Y_phi"
    assert domain["variables"][0]["locally_measurable"]
    assert domain["inequalities"] == ["Y_phi*c_X < M_Pl**4"]
    assert domain["preservation"]["status"] == "unresolved"


def test_background_domain_rejects_unknown_unsafe_and_nonlocal_declarations() -> None:
    base = _scalar_spec()
    variable = {
        "id": "Y_phi",
        "covariant_definition": "-g^{mu nu} nabla_mu(phi) nabla_nu(phi)",
        "unitary_gauge_identification": "A_star^2",
        "mass_dimension": 4,
        "nonnegative": True,
        "locally_measurable": True,
    }
    base["background_domain"] = {
        "variables": [variable],
        "inequalities": ["unknown_field < M_Pl"],
        "preservation": {
            "status": "unresolved",
            "statement": "not yet proved",
            "required_controls": [],
        },
    }
    unknown = compile_action_spec(base, GRAMMAR, CONTRACT)
    assert not unknown["valid"]
    assert any("uses undeclared names" in item for item in unknown["errors"])

    unsafe_spec = copy.deepcopy(base)
    unsafe_spec["background_domain"]["inequalities"] = [
        "__import__('os').system('echo unsafe') > 0"
    ]
    unsafe = compile_action_spec(unsafe_spec, GRAMMAR, CONTRACT)
    assert not unsafe["valid"]
    assert any("unsupported background inequality syntax" in item for item in unsafe["errors"])

    nonlocal_spec = copy.deepcopy(base)
    nonlocal_spec["background_domain"]["inequalities"] = ["Y_phi < M_Pl**4"]
    nonlocal_spec["background_domain"]["variables"][0]["locally_measurable"] = False
    nonlocal_result = compile_action_spec(nonlocal_spec, GRAMMAR, CONTRACT)
    assert not nonlocal_result["valid"]
    assert any("locally_measurable must be true" in item for item in nonlocal_result["errors"])


def test_einstein_hilbert_control_compiles_as_metric_only_action() -> None:
    import json

    spec = json.loads(
        (ROOT / "configs" / "actions" / "einstein_hilbert_control.json").read_text(encoding="utf-8")
    )
    result = compile_action_spec(spec, GRAMMAR, CONTRACT)
    assert result["valid"], result["errors"]
    assert result["canonical"]["fields"] == ["g_mu_nu"]
    assert [term["id"] for term in result["canonical"]["terms"]] == ["EH_R"]


def test_generated_q_candidate_is_origin_bound_and_baryon_free() -> None:
    import json

    spec = json.loads(
        (
            ROOT
            / "configs"
            / "actions"
            / "generated_gf_cb4ebf3da5a74582_q_q2_candidate.json"
        ).read_text(encoding="utf-8")
    )
    result = compile_action_spec(spec, GRAMMAR, CONTRACT)
    assert result["valid"], result["errors"]
    assert result["canonical"]["source_role"] == "candidate"
    assert result["canonical"]["generator_origin"]["ordinal"] == 723
    assert {term["invariant"] for term in result["canonical"]["terms"]} == {
        None,
        "Q_a_u",
    }
    assert "z_b" not in str(result["canonical"])


def test_generator_origin_is_fail_closed() -> None:
    spec = _scalar_spec()
    spec["generator_origin"] = {"family_id": "missing-required-fields"}
    result = compile_action_spec(spec, GRAMMAR, CONTRACT)
    assert not result["valid"]
    assert any("generator_origin.ordinal" in error for error in result["errors"])
