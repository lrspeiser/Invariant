from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.g4_phi_seed_formal_audit import (
    _sha,
    _validate_target,
    build_g4_phi_seed_formal_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g4_phi_seed_formal_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g4-phi-seed-formal-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g4_phi_seed_formal_audit(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "30ad295116926bfd8f1e2c612f36fdae1148fc317cc3c45e5840ec20c08ff531"
    )


def test_conformal_and_scalar_kinetic_intervals_are_exact(rebuilt: dict) -> None:
    y = sp.symbols("y", nonnegative=True)
    f = 1 + y / 50
    kinetic = sp.factor(1 / f + sp.Rational(3, 2) * (sp.Rational(1, 25) ** 2 * y) / f**2)
    assert sp.factor(kinetic - (56 * y + 2500) / (y + 50) ** 2) == 0
    assert sp.factor(
        sp.diff(kinetic, y) + (56 * y + 2200) / (y + 50) ** 3
    ) == 0
    assert kinetic.subs(y, 0) == 1
    assert kinetic.subs(y, 1) == sp.Rational(284, 289)
    certificate = rebuilt["candidate_records"][0]["candidate_certificate"]
    transform = certificate["invertible_field_transformation"]
    assert transform["f_interval"] == ["1", "51/50"]
    assert transform["K_E_interval"] == ["284/289", "1"]
    assert transform["regular_on_entire_phi_domain"] is True


def test_local_lapse_operator_is_positive_without_claiming_global_inverse(rebuilt: dict) -> None:
    certificate = rebuilt["candidate_records"][0]["candidate_certificate"]
    lapse = certificate["unitary_lapse_operator"]
    assert lapse["local_Delta_N_kernel"] == "sqrt(q)*f*K_E/N^3*delta(x-y)"
    assert lapse["f_times_K_E_interval"] == ["1", "426/425"]
    assert lapse["pointwise_nonzero_for_finite_N_positive"] is True
    assert lapse["local_constraint_matrix_rank"] == 2
    assert lapse["local_physical_dof"] == 3
    assert lapse["global_bounded_inverse"] == "blocked"
    gates = rebuilt["candidate_records"][0]["gate_ledger"]
    assert gates["candidate_local_lapse_dirac_pair"]["status"] == "pass"
    assert gates["global_lapse_operator_invertibility"]["status"] == "blocked"


def test_inhomogeneous_common_cone_passes_but_global_energy_stays_blocked(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    principal = record["candidate_certificate"]["inhomogeneous_principal_common_cone"]
    assert principal["status"] == "pass"
    assert principal["jordan_and_einstein_null_cones_identical"] is True
    assert principal["uniform_tensor_margin_f"] == "1"
    assert principal["uniform_scalar_margin_K_E"] == "284/289"
    assert record["gate_ledger"]["inhomogeneous_principal_symbol"]["status"] == "pass"
    assert record["gate_ledger"]["common_time_cone"]["status"] == "pass"
    assert record["gate_ledger"]["global_positive_energy"]["status"] == "blocked"
    assert record["decision"] == "blocked"
    assert record["necessary_condition_rejection_found"] is False


def test_adapter_replays_known_answer_and_eligibility_are_sealed(rebuilt: dict) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert rebuilt["invoked_adapter_entrypoints"] == [
        item["entrypoint"] for item in config["formal_adapters"]
    ]
    assert rebuilt["known_answer_control"]["eligible_as_candidate_evidence"] is False
    assert rebuilt["decision_counts"] == {"blocked": 1}
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_action_binding_is_rejected() -> None:
    predecessor = json.loads(
        (ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-compilation-campaign.json").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    target = copy.deepcopy(config["target_seed"])
    record = next(item for item in predecessor["candidate_records"] if item["seed_id"] == target["seed_id"])
    target["action_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="action hash mismatch"):
        _validate_target(record, target)
