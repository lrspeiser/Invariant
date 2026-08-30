from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_shared_quadrature_lensing_backreaction as lensing

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_predecessor_are_exactly_bound() -> None:
    config = lensing.load_config(ROOT)
    predecessor = lensing.validate_predecessor(ROOT, config["predecessor_binding"])
    assert predecessor == {
        "binding_id": "shared_quadrature_covariant_action",
        "git_commit": "28c5d95c8ddc263cfdbd6c83f99b177a2a282b4f",
        "artifact_count": 4,
        "receipt_content_sha256": "7503b0b833eef1e84f3db97245abcaa7f3f1b6fde1aa6b3678d04e4f155e5398",
        "valid": True,
    }


@pytest.mark.parametrize("section", list(lensing.EXPECTED_SECTION_HASHES))
def test_every_frozen_config_section_rejects_mutation(section: str) -> None:
    config = lensing.load_config(ROOT)
    changed = copy.deepcopy(config)
    value = changed[section]
    if isinstance(value, dict):
        value["unexpected"] = False
    else:  # pragma: no cover - every frozen section is a dictionary
        changed[section] = None
    with pytest.raises(lensing.QuadratureLensingBackreactionError, match=f"config {section}"):
        lensing.validate_config(changed)


def test_linearized_einstein_and_asymptotic_checks_pass_exactly() -> None:
    checks, expressions = lensing.symbolic_checks()
    assert len(checks) == 16
    assert all(row["passed"] and row["residual"] == "0" for row in checks)
    assert checks[0]["check_id"] == "S01_LINEARIZED_G00"
    assert checks[-1]["check_id"] == "S16_DIRECT_CONFORMAL_LENSING_CANCELLATION"
    assert "log" in expressions["kinetic_density"]
    assert "sqrt" in expressions["exterior_branch"]


def test_numeric_exterior_probes_are_positive_and_convergent() -> None:
    rows = lensing.numeric_probes(lensing.load_config(ROOT))
    assert len(rows) == 4
    assert all(row["passed"] for row in rows)
    assert all(row["normalized_energy_density"] > 0 for row in rows)
    assert all(row["normalized_lensing_source"] > 0 for row in rows)
    errors = [row["branch_asymptotic_error"] for row in rows]
    assert errors == sorted(errors, reverse=True)


def test_receipt_is_exact_and_preserves_failure_ceiling() -> None:
    receipt = lensing.build_receipt(ROOT)
    assert receipt["status"] == lensing.STATUS
    assert receipt["decision"] == lensing.DECISION
    assert receipt["counts"] == {
        "predecessor_artifacts": 4,
        "symbolic_checks": 16,
        "symbolic_checks_passed": 16,
        "numeric_probes": 4,
        "numeric_probes_passed": 4,
        "observational_files_opened": 0,
        "observational_rows_opened": 0,
        "network_calls": 0,
        "model_or_paid_calls": 0,
        "gpu_calls": 0,
    }
    assert receipt["adjudication"]["restricted_linearized_metric_backreaction_derived"] is True
    assert receipt["adjudication"]["direct_conformal_lensing_shift_cancels"] is True
    assert receipt["adjudication"]["scalar_lensing_backreaction_compactness_suppressed"] is True
    assert (
        receipt["adjudication"][
            "same_action_lensing_matches_scalar_motion_enhancement_asymptotically"
        ]
        is False
    )
    assert receipt["adjudication"]["finite_isolated_scalar_energy"] is False
    assert receipt["claim_boundary"]["same_action_quantitative_lensing_success"] is False
    assert receipt["claim_boundary"]["scientific_observational_claim_allowed"] is False
    assert set(receipt["zero_access_and_compute"].values()) == {0}


def test_stored_receipt_rebuilds_and_second_write_is_identical() -> None:
    stored = json.loads((ROOT / lensing.OUTPUT_PATH).read_text(encoding="utf-8"))
    lensing.validate_receipt(stored, ROOT)
    assert stored == lensing.build_receipt(ROOT)
    path, publication = lensing.write_receipt(ROOT)
    assert path == ROOT / lensing.OUTPUT_PATH
    assert publication == "EXISTING_IDENTICAL"


def test_changed_receipt_claim_fails_even_if_rehashed() -> None:
    receipt = lensing.build_receipt(ROOT)
    receipt["claim_boundary"]["same_action_quantitative_lensing_success"] = True
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = lensing._sha(body)
    with pytest.raises(lensing.QuadratureLensingBackreactionError, match="evidence changed"):
        lensing.validate_receipt(receipt, ROOT)


def test_source_has_no_observational_or_network_loader() -> None:
    source = (
        ROOT / "src/sigma_theory_compiler/gravity_shared_quadrature_lensing_backreaction.py"
    ).read_text(encoding="utf-8")
    assert "urllib" not in source
    assert "requests" not in source
    assert "pandas" not in source
    assert "astropy" not in source
