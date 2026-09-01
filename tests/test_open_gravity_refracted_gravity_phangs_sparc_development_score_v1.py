from __future__ import annotations

import copy
from functools import lru_cache

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_sparc_development_score_v1 as subject,
)


@lru_cache(maxsize=1)
def _receipt() -> dict[str, object]:
    return subject.build_receipt(subject.load_config(verify_package=False))


def test_config_freezes_no_tuning_and_paper_source_gates() -> None:
    config = subject.load_config(verify_package=False)
    subject.validate_config(config)
    assert config["method_rule"]["real_source_required"] is True
    assert config["method_rule"]["primary_paper_or_analytic_benchmark_required"] is True
    assert config["method_rule"]["response_parameter_source_radius_or_threshold_tuning"] is False
    assert config["method_rule"]["published_prior_corners_may_compete"] is False
    assert config["candidate_contract"]["per_object_fitted_parameters"] == 0
    assert config["access_scope"]["confirmation_rows_opened"] == 0


def test_predecessors_and_response_contracts_are_exact() -> None:
    evidence = subject.validate_predecessors(subject.load_config(verify_package=False))
    assert evidence["field_receipt"]["all_object_gates_pass"] is True
    assert evidence["field_receipt"]["response_blind_radius_summary"]["eligible_points"] == 852
    assert evidence["source_envelope_receipt"]["registered_source_parameter_pairs"] == 2025


def test_candidate_formulas_have_expected_limits() -> None:
    newton = np.asarray([0.0, 1.2e-10, 1.0e-4])
    refracted = np.asarray([0.0, 2.0e-10, 1.1e-4])
    assert np.array_equal(
        subject._candidate_acceleration("NEWTON_3D_DST", newton, refracted, a0_m_s2=1.2e-10),
        newton,
    )
    assert np.array_equal(
        subject._candidate_acceleration(
            "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG",
            newton,
            refracted,
            a0_m_s2=1.2e-10,
        ),
        refracted,
    )
    rar = subject._candidate_acceleration(
        "RAR_2016_ON_NEWTON_3D", newton, refracted, a0_m_s2=1.2e-10
    )
    mond = subject._candidate_acceleration(
        "MOND_STANDARD_MU_ON_NEWTON_3D", newton, refracted, a0_m_s2=1.2e-10
    )
    assert rar[0] == mond[0] == 0.0
    assert np.all(rar[1:] >= newton[1:])
    assert np.all(mond[1:] >= newton[1:])
    assert abs(rar[-1] / newton[-1] - 1.0) < 1.0e-3
    assert abs(mond[-1] / newton[-1] - 1.0) < 1.0e-6


def test_fixed_development_score_covers_every_object_and_common_rows() -> None:
    receipt = _receipt()
    assert len(receipt["phangs_object_scores"]) == 3
    assert len(receipt["sparc_object_scores"]) == 1
    assert {row["object_id"] for row in receipt["phangs_object_scores"]} == {
        "NGC2903",
        "NGC3351",
        "NGC3627",
    }
    assert receipt["sparc_object_scores"][0]["object_id"] == "NGC2903"
    candidates = set(receipt["candidate_contract"]["candidate_ids"])
    for object_row in receipt["phangs_object_scores"] + receipt["sparc_object_scores"]:
        assert set(object_row["candidates"]) == candidates
        row_counts = {row["rows_scored"] for row in object_row["candidates"].values()}
        assert row_counts == {object_row["eligibility"]["rows_scored_common"]}
        assert object_row["eligibility"]["eligibility_used_velocity_values"] is False
        assert object_row["eligibility"]["rows_scored_common"] >= 5


def test_receipt_preserves_scope_and_source_uncertainty_warning() -> None:
    receipt = _receipt()
    accounting = receipt["access_accounting"]
    assert accounting["phangs"]["container_objects_opened"] == 33
    assert accounting["phangs"]["container_response_rows_opened"] == 1321
    assert accounting["sparc"]["container_objects_opened"] == 175
    assert accounting["sparc"]["container_response_rows_opened"] == 3391
    for key in (
        "confirmation_rows_opened",
        "independent_rows_opened",
        "group_rows_opened",
        "lensing_rows_opened",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        assert accounting[key] == 0
    assert receipt["adjudication"]["source_systematic_robustness_established"] is False
    assert receipt["source_systematic_context"]["loss_robustness_recomputed"] is False
    assert receipt["claim_boundary"]["publication_ready"] is False


def test_adjudication_is_recomputed_from_all_object_losses() -> None:
    receipt = _receipt()
    adjudication = receipt["adjudication"]
    candidate = adjudication["candidate_id"]
    phangs = adjudication["phangs"]
    best = phangs["best_comparator_id"]
    observed = (
        phangs["candidate_aggregates"][best]["loss"]
        - phangs["candidate_aggregates"][candidate]["loss"]
    ) / phangs["candidate_aggregates"][best]["loss"]
    assert observed == phangs["fractional_improvement_over_best_comparator"]
    assert adjudication["development_signal"] == all(adjudication["checks"].values())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "PUBLICATION_READY"),
        (("method_rule", "real_source_required"), False),
        (("method_rule", "published_prior_corners_may_compete"), True),
        (("candidate_contract", "per_object_fitted_parameters"), 1),
        (("radius_gate", "velocity_values_used_by_gate"), True),
        (("radius_gate", "fine_vs_convergence_maximum_relative_difference"), 1.0),
        (("access_scope", "confirmation_rows_opened"), 1),
    ],
)
def test_config_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(subject.load_config(verify_package=False))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(subject.DevelopmentScoreError):
        subject.validate_config(config)


def test_receipt_self_hash_and_determinism() -> None:
    receipt = _receipt()
    assert receipt["content_sha256"] == subject.content_sha256({**receipt, "content_sha256": ""})
    rebuilt = subject.build_receipt(subject.load_config(verify_package=False))
    assert subject.canonical_bytes(rebuilt) == subject.canonical_bytes(receipt)


def test_atomic_no_clobber(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    assert subject._atomic_no_clobber(path, b"a\n") == "CREATED"
    assert subject._atomic_no_clobber(path, b"a\n") == "EXISTING_IDENTICAL"
    with pytest.raises(subject.DevelopmentScoreError):
        subject._atomic_no_clobber(path, b"b\n")
