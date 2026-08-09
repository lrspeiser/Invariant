from pathlib import Path

import pytest

from sigma_theory_compiler.action_ir import compile_action_file
from sigma_theory_compiler.adm_ir import compile_adm_ir
from sigma_theory_compiler.formal_backend import run_formal_control_suite
from sigma_theory_compiler.legendre_ir import compile_legendre_ir
from sigma_theory_compiler.static_dictionary import compile_static_dictionary_ir

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"
SPECS = (
    "generated_cv3_x_p1_2_static_null.json",
    "generated_cv3_x_p2_3_static_null.json",
    "generated_cv3_x_p3_4_static_null.json",
)
MATCHED_SPECS = (
    "generated_cv3_x_p1_2_matched.json",
    "generated_cv3_x_p2_3_matched.json",
    "generated_cv3_x_p3_4_matched.json",
)


@pytest.mark.parametrize("name", (*SPECS, *MATCHED_SPECS))
def test_covariant_first_x_candidates_compile_and_match_exact_static_origin(name: str) -> None:
    action = compile_action_file(ROOT / "configs" / "actions" / name, GRAMMAR, CONTRACT)
    static = compile_static_dictionary_ir(action)
    assert action["valid"], action["errors"]
    assert "Q_a_u" not in {term["invariant"] for term in action["canonical"]["terms"]}
    assert static["status"] == "pass"
    assert static["legacy_generator_dictionary"]["q"]["exact_shape_match"]
    assert static["legacy_generator_dictionary"]["x"]["status"] == (
        "derived_and_generator_matched"
    )
    assert static["aether_static_acceleration_sector"][
        "action_density_coefficient_of_a_i_a^i"
    ] == "0"


def test_new_nonlinear_x_adm_templates_pass_but_legendre_stays_fail_closed() -> None:
    formal = run_formal_control_suite(CONTRACT, ROOT)
    controls = {item["name"]: item["status"] == "pass" for item in formal["checks"]}
    for name in SPECS[1:]:
        action = compile_action_file(ROOT / "configs" / "actions" / name, GRAMMAR, CONTRACT)
        adm = compile_adm_ir(action, controls)
        legendre = compile_legendre_ir(action, adm)
        assert adm["status"] == "pass"
        assert adm["higher_time_derivative_channels"] == []
        assert legendre["status"] == "unresolved"
        assert legendre["unsupported_kinetic_terms"] == [
            "AETHER_X_P2_3" if "p2_3" in name else "AETHER_X_P3_4"
        ]

    for name in MATCHED_SPECS:
        action = compile_action_file(ROOT / "configs" / "actions" / name, GRAMMAR, CONTRACT)
        adm = compile_adm_ir(action, controls)
        legendre = compile_legendre_ir(action, adm)
        assert adm["status"] == "pass"
        assert adm["higher_time_derivative_channels"] == []
        assert legendre["status"] == "unresolved"
        assert any(term.startswith("AETHER_MATCHED_K14") for term in legendre["unsupported_kinetic_terms"])
