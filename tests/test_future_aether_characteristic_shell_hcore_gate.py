from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_characteristic_shell_hcore_gate import (
    BLOCKER,
    CHARACTERISTIC_BLOCKER,
    TARGET_ID,
    YORK_SHELL_BLOCKER,
    _validate_result,
    build_future_aether_characteristic_shell_hcore_gate,
    exact_characteristic_shell_control,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/future_aether_characteristic_shell_hcore_gate.json"
ARTIFACT = ROOT / "runs/engine/future-aether-characteristic-shell-hcore-gate.json"
SOURCE = ROOT / "runs/engine/future-aether-canonical-seed-constraint-dag-gate.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_future_aether_characteristic_shell_hcore_gate(_load(CONFIG), ROOT)


def _target(artifact: dict[str, object]) -> dict[str, object]:
    return next(record for record in artifact["candidate_records"] if record["candidate_id"] == TARGET_ID)


def test_exact_rebuild_and_candidate_partition(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["candidate_count"] == 14
    assert artifact["decision_counts"] == {"blocked": 14}
    assert artifact["first_blocker_counts"] == {
        CHARACTERISTIC_BLOCKER: 11,
        YORK_SHELL_BLOCKER: 2,
        BLOCKER: 1,
    }
    assert artifact["formal_pass_count"] == 0
    assert artifact["candidate_rejection_authorized_count"] == 0


def test_target_hessian_determinant_and_characteristic_shell_are_exact(
    rebuilt: dict[str, object],
) -> None:
    control = _target(rebuilt)["characteristic_shell_H_core_certificate"][
        "characteristic_shell_control"
    ]
    assert control["passed"] is True
    assert control["hessian_determinant"] == (
        "-31*(F**2 - 31)**2*(33*F**2 + 65)*(61*F**2 + 124)**2/"
        "(8796093022208*(F**2 + 1))"
    )
    assert control["only_real_characteristic_condition"] == "F**2=31"
    shell = control["declared_profile_characteristic_shell"]
    assert shell["F_squared_residual"] == "0"
    assert shell["radius_interval"] == "0<r_characteristic<1"
    assert shell["hessian_rank"] == 7
    assert shell["hessian_nullity"] == 2
    assert shell["nullspace"] == [
        ["0", "-1", "1", "0", "0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0", "1", "0", "0", "0"],
    ]


def test_seed_momenta_are_compatible_but_inverse_is_nonunique(
    rebuilt: dict[str, object],
) -> None:
    certificate = _target(rebuilt)["characteristic_shell_H_core_certificate"]
    control = certificate["characteristic_shell_control"]
    image = control["seed_legendre_image"]
    assert image["affine_residual"] == ["0"] * 9
    assert image["shell_primary_compatibility_residuals"] == ["0", "0"]
    assert "not uniquely recoverable" in image["interpretation"]
    regular = control["regular_stratum_H_core"]
    assert regular["domain"] == "F**2 != 31"
    assert regular["registered"] is True
    assert regular["global_on_declared_profile"] is False
    assert certificate["regular_stratum_flat_chart_H_core_contract_registered"] is True
    assert certificate["declared_profile_global_flat_chart_H_core_registered"] is False


def test_exact_positive_and_negative_controls() -> None:
    control = exact_characteristic_shell_control()
    negative = control["incompatible_momentum_negative_control"]
    assert negative["rejected"] is True
    assert negative["null_projection_residuals"] == ["0", "1"]
    noncrossing = control["noncrossing_profile_control"]
    assert noncrossing["uniform_F_squared_upper_bound"] == "25"
    assert noncrossing["distance_to_characteristic_F_squared"] == "6"
    assert noncrossing["avoids_shell"] is True
    assert "not substituted" in noncrossing["scope"]


def test_eleven_characteristic_and_two_york_records_are_unchanged(
    rebuilt: dict[str, object],
) -> None:
    source = _load(SOURCE)
    source_records = {record["candidate_id"]: record for record in source["candidate_records"]}
    for record in rebuilt["candidate_records"]:
        if record["candidate_id"] == TARGET_ID:
            continue
        predecessor = source_records[record["candidate_id"]]
        assert record["first_blocker"] == predecessor["first_blocker"]
        assert record["source_record_sha256"] == predecessor["content_sha256"]
        assert record["typed_action_ir_sha256"] == predecessor["typed_action_ir_sha256"]
        assert record["parameters"] == predecessor["parameters"]


def test_all_downstream_and_observation_seals_remain_closed(
    rebuilt: dict[str, object],
) -> None:
    assert rebuilt["declared_profile_global_flat_chart_H_core_registered_count"] == 0
    assert rebuilt["off_flat_metric_covariantization_registered_count"] == 0
    assert rebuilt["metric_covariantized_H_D_Frechet_DAG_registered_count"] == 0
    assert rebuilt["constraint_satisfying_negative_total_energy_datum_count"] == 0
    assert rebuilt["automatic_downstream_enqueue_performed"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["data_eligibility"] == ELIGIBILITY
    for record in rebuilt["candidate_records"]:
        assert record["decision"] == "blocked"
        assert record["formal_pass"] is False
        assert record["candidate_rejection_authorized"] is False
        assert record["constraint_satisfying_negative_total_energy_datum_proven"] is False
        assert record["automatic_downstream_enqueue_performed"] is False
        assert record["solar_bundle_generated"] is False
        assert record["data_eligibility"] == ELIGIBILITY


def test_source_and_target_tampering_fail_before_execution() -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["source_canonical_artifact"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="canonical artifact file hash mismatch"):
        build_future_aether_characteristic_shell_hcore_gate(tampered, ROOT)

    tampered = copy.deepcopy(config)
    tampered["exact_target"]["profile"] = "F=5*(1-r^2)^4_+"
    with pytest.raises(ValueError, match="characteristic-shell target changed"):
        build_future_aether_characteristic_shell_hcore_gate(tampered, ROOT)


def test_artifact_tampering_cannot_authorize_hcore_or_rejection() -> None:
    artifact = _load(ARTIFACT)
    tampered = copy.deepcopy(artifact)
    certificate = _target(tampered)["characteristic_shell_H_core_certificate"]
    certificate["declared_profile_global_flat_chart_H_core_registered"] = True
    cert_body = {key: item for key, item in certificate.items() if key != "content_sha256"}
    certificate["content_sha256"] = _sha(cert_body)
    record = _target(tampered)
    record_body = {key: item for key, item in record.items() if key != "content_sha256"}
    record["content_sha256"] = _sha(record_body)
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = _sha(body)
    with pytest.raises(ValueError, match="certificate overclaimed"):
        _validate_result(tampered)
