from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_split_gate_range_source_tradeoff as tradeoff,
)

ROOT = Path(__file__).resolve().parents[1]


def test_symbolic_theorem_routes_are_exact() -> None:
    checks = tradeoff.symbolic_checks()
    assert checks == {
        "T02_C_BRANCH_AMPLITUDE_LIMIT": True,
        "T03_K_BRANCH_AMPLITUDE_LIMIT": True,
        "T04_RANGE_LIMIT": True,
        "T05_C_SOURCE_LIMIT": True,
        "T06_K_SOURCE_LIMIT": True,
        "T07_C_PRODUCT_LIMIT": True,
        "T08_K_PRODUCT_LIMIT": True,
        "T09_CRITICAL_EXPONENT": True,
        "T10_SHIFTED_POWER_MAPPING": True,
        "T11_COMMITTED_P2_RECOVERY": True,
        "T12_FIXED_DISTANCE_SCREENING": True,
    }


def test_shifted_family_converges_to_all_four_exponents() -> None:
    config = tradeoff.load_config(ROOT)
    numeric = tradeoff.numeric_probes(config)
    assert numeric["all_passed"] is True
    assert len(numeric["exponent_records"]) == 4
    for record in numeric["exponent_records"]:
        power = record["power"]
        assert record["slopes"]["ell"] == pytest.approx(-power, abs=2e-6)
        assert record["slopes"]["chi_max"] == pytest.approx(0.5 - power, abs=2e-6)
        assert record["slopes"]["q_max"] == pytest.approx(power + 0.5, abs=2e-6)
        assert record["slopes"]["product"] == pytest.approx(0.5, abs=2e-6)


def test_source_threshold_has_all_three_behaviors() -> None:
    rows = tradeoff.numeric_probes(tradeoff.load_config(ROOT))["threshold_records"]
    assert len(rows) == 12
    assert {row["behavior"] for row in rows} == {
        "DECREASING_SUBCRITICAL",
        "COEFFICIENT_DEPENDENT_CRITICAL",
        "INCREASING_SUPERCRITICAL",
    }
    assert all(row["passed"] for row in rows)
    assert sum(row["behavior"] == "INCREASING_SUPERCRITICAL" for row in rows) == 4


def test_committed_p2_exponents_are_recovered() -> None:
    config = tradeoff.load_config(ROOT)
    p2 = next(
        row for row in tradeoff.numeric_probes(config)["exponent_records"] if row["power"] == 2.0
    )
    assert p2["targets"] == {
        "ell": -2.0,
        "chi_max": -1.5,
        "q_max": 2.5,
        "product": 0.5,
    }


def test_universal_product_and_no_free_range_are_scoped() -> None:
    theorem = tradeoff.load_config(ROOT)["asymptotic_theorem"]
    assert "sqrt(X)" in theorem["universal_product"]
    assert "independent of the gate amplitude and power" in theorem["universal_product"]
    assert "s>2*r-1" in theorem["no_free_range_corollary"]
    assert "sufficient bounded-source guarantee" in theorem["sharpness_boundary"]
    assert "not a necessary failure condition" in theorem["sharpness_boundary"]


def test_all_predecessor_bytes_match_their_commits() -> None:
    config = tradeoff.load_config(ROOT)
    for binding in config["bindings"]:
        for role in ("config", "module", "test", "receipt"):
            relative = binding[f"{role}_path"]
            expected = binding[f"{role}_sha256"]
            assert tradeoff._sha256_file(ROOT / relative) == expected
            assert (
                tradeoff._sha256_bytes(tradeoff._git_show(ROOT, binding["commit"], relative))
                == expected
            )


def test_receipt_is_deterministic_and_restrictive() -> None:
    first = tradeoff.build_receipt(ROOT)
    second = tradeoff.build_receipt(ROOT)
    assert first == second
    assert first["checks_passed"] == 15
    assert all(first["checks"].values())
    assert first["status"] == "PROMISING_SECOND_ASYMPTOTIC_THEOREM_CANDIDATE_NOT_PREPRINT_READY"
    assert first["claim_boundary"]["candidate_original_asymptotic_theorem"] is True
    assert first["claim_boundary"]["historical_novelty_established"] is False
    assert first["claim_boundary"]["publication_ready"] is False
    assert first["content_sha256"] == tradeoff._self_hash(first)


def test_conformal_source_is_bound_but_its_x_scaling_is_not_assumed() -> None:
    config = tradeoff.load_config(ROOT)
    assert "Q_chi=-(partial_chi ln A)*T_E" in config["theory_class"]["source_identity"]
    assert config["claim_boundary"]["physical_source_X_scaling_derived"] is False
    assert config["claim_boundary"]["on_shell_material_solution"] is False


def test_restricted_fifth_force_is_not_promoted_to_solar_success() -> None:
    link = tradeoff.load_config(ROOT)["fifth_force_link"]
    assert "epsilon_chi" in link["restricted_ratio"]
    assert "does not screen phi" in link["not_claimed"]
    receipt = tradeoff.build_receipt(ROOT)
    assert receipt["claim_boundary"]["observational_support"] is False
    assert receipt["claim_boundary"]["modified_gravity_success"] is False


def test_config_mutation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = copy.deepcopy(tradeoff.load_config(ROOT))
    config["claim_boundary"]["publication_ready"] = True
    monkeypatch.setattr(tradeoff, "_read_json", lambda _path: config)
    with pytest.raises(tradeoff.SplitGateTradeoffError, match="config changed"):
        tradeoff.load_config(ROOT)


def test_receipt_theorem_forgery_fails_after_coherent_rehash() -> None:
    config = tradeoff.load_config(ROOT)
    receipt = copy.deepcopy(tradeoff.build_receipt(ROOT))
    receipt["maximal_theorem"]["universal_product"] = "Q_max*ell=constant"
    receipt["content_sha256"] = tradeoff._self_hash(receipt)
    with pytest.raises(tradeoff.SplitGateTradeoffError, match="theorem changed"):
        tradeoff.validate_receipt(receipt, config)


def test_receipt_numeric_forgery_fails_after_coherent_rehash() -> None:
    config = tradeoff.load_config(ROOT)
    receipt = copy.deepcopy(tradeoff.build_receipt(ROOT))
    receipt["numeric_evidence"]["exponent_records"][0]["slopes"]["product"] = 0.0
    receipt["content_sha256"] = tradeoff._self_hash(receipt)
    with pytest.raises(tradeoff.SplitGateTradeoffError, match="numeric evidence changed"):
        tradeoff.validate_receipt(receipt, config)


def test_noncanonical_root_is_rejected_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = 0

    def forbidden_read(_path: Path) -> dict[str, object]:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(tradeoff, "_read_json", forbidden_read)
    with pytest.raises(tradeoff.SplitGateTradeoffError, match="noncanonical repository root"):
        tradeoff.load_config(tmp_path)
    assert reads == 0


def test_module_semantic_hash_ignores_only_config_seal(tmp_path: Path) -> None:
    original = (ROOT / tradeoff.SOURCE_PATH).read_text(encoding="utf-8")
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    third = tmp_path / "third.py"
    first.write_text(original, encoding="utf-8")
    second.write_text(
        original.replace(tradeoff.CONFIG_CANONICAL_SHA256, "f" * 64), encoding="utf-8"
    )
    third.write_text(original + "\n# semantic mutation\n", encoding="utf-8")
    assert tradeoff._module_semantic_sha256(first) == tradeoff._module_semantic_sha256(second)
    assert tradeoff._module_semantic_sha256(first) != tradeoff._module_semantic_sha256(third)


def test_atomic_receipt_publication_is_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert tradeoff._atomic_no_replace(path, b"sealed\n") == "CREATED"
    assert tradeoff._atomic_no_replace(path, b"sealed\n") == "EXISTING_IDENTICAL"
    with pytest.raises(tradeoff.SplitGateTradeoffError, match="refusing to overwrite"):
        tradeoff._atomic_no_replace(path, b"changed\n")
    assert path.read_bytes() == b"sealed\n"


def test_zero_access_and_finite_numeric_evidence() -> None:
    receipt = tradeoff.build_receipt(ROOT)
    assert all(value == 0 for value in receipt["zero_access"].values())
    for row in receipt["numeric_evidence"]["exponent_records"]:
        assert math.isfinite(row["max_abs_error"])
    assert len(receipt["literature_positioning"]) == 5
    assert receipt["novelty_search_scope"]["limitation"].startswith("This is not an exhaustive")


def test_cli_has_no_arbitrary_input_or_output_path() -> None:
    parser_source = (ROOT / tradeoff.SOURCE_PATH).read_text(encoding="utf-8")
    assert 'choices=("build", "check", "status")' in parser_source
    assert 'add_argument("--root"' not in parser_source
    assert 'add_argument("--output"' not in parser_source
    assert 'add_argument("--receipt"' not in parser_source
