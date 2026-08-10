from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.aether_twist_sector_energy_audit import (
    _sha,
    _validate_target,
    build_aether_twist_sector_energy_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "aether_twist_sector_energy_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "aether-twist-sector-energy-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_aether_twist_sector_energy_audit(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "38497dd37c9f197a3e0e61ffdd7552fab82f7cf07f22ed375f15ea5a044a1725"
    )


def test_quadratic_reduced_energy_and_dirac_signs_are_candidate_specific(
    rebuilt: dict,
) -> None:
    expected = {
        "G3-0b8cb2d5591bf50d2465978d": {
            "energy": {"spin_2": "1", "spin_1": "7/12", "spin_0": "135/256"},
            "kinetic": {"tensor": "3/4", "vector": "5/8", "scalar": "525/1024"},
            "gradient": {"tensor": "1", "vector": "7/12", "scalar": "135/256"},
        },
        "G3-94086fa702500475b35ab002": {
            "energy": {"spin_2": "1", "spin_1": "107/120", "spin_0": "2231/3600"},
            "kinetic": {
                "tensor": "5/8",
                "vector": "23/30",
                "scalar": "3703/7040",
            },
            "gradient": {"tensor": "1", "vector": "107/120", "scalar": "2231/3600"},
        },
    }
    by_id = {item["seed_id"]: item for item in rebuilt["candidate_records"]}
    for seed_id, coefficients in expected.items():
        record = by_id[seed_id]
        quadratic = record["twist_sector_certificate"]["quadratic_constraint_reduced_energy"]
        assert quadratic["energy_coefficients"] == coefficients["energy"]
        assert quadratic["kinetic_coefficients"] == coefficients["kinetic"]
        assert quadratic["gradient_coefficients"] == coefficients["gradient"]
        assert quadratic["all_positive"] is True
        assert record["gate_ledger"][
            "regular_patch_legendre_and_dirac_constraint_algebra"
        ]["status"] == "pass"


def test_exact_static_twist_energy_is_coercive_but_does_not_promote(rebuilt: dict) -> None:
    expected = {
        "G3-0b8cb2d5591bf50d2465978d": ("1/4", "-5/32", "3/32"),
        "G3-94086fa702500475b35ab002": ("7/24", "-23/120", "1/10"),
    }
    for record in rebuilt["candidate_records"]:
        sector = record["twist_sector_certificate"]["nonlinear_static_pure_twist_sector"]
        c_zero, first_nonlinear, infimum = expected[record["seed_id"]]
        assert sector["C_at_zero"] == c_zero
        assert sector["first_nonlinear_term_in_C"] == f"({first_nonlinear})*y"
        assert sector["C_infimum_y_to_infinity"] == infimum
        assert Fraction(infimum) > 0
        assert sector["positive_for_all_tilts_and_twist_orientations"] is True
        assert sector["negative_energy_mode_found"] is False
        assert record["decision"] == "blocked"
        assert record["first_missing_premise"] == "complete_generic_twisting_reduced_hamiltonian"


def test_frobenius_witness_remains_kinematic_and_maxwell_is_inference_control(
    rebuilt: dict,
) -> None:
    assert rebuilt["prior_frobenius_witness_treatment"] == {
        "status": "kinematic_noncoverage_witness",
        "establishes": (
            "generic unit-timelike Aether configurations need not be hypersurface orthogonal"
        ),
        "negative_energy_mode_found": False,
        "candidate_rejection_authorized": False,
    }
    assert rebuilt["inference_negative_control"]["full_hamiltonian_status"] == "reject"
    assert rebuilt["decision_counts"] == {"blocked": 2}
    assert rebuilt["formal_pass_count"] == 0
    for record in rebuilt["candidate_records"]:
        assert record["negative_energy_mode_found"] is False
        assert record["gate_ledger"]["complete_generic_twisting_reduced_hamiltonian"][
            "status"
        ] == "blocked"
        assert record["gate_ledger"]["global_positive_energy"]["status"] == "blocked"


def test_adapters_and_external_data_remain_sealed(rebuilt: dict) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert rebuilt["invoked_adapter_entrypoints"] == [
        item["entrypoint"] for item in config["formal_adapters"]
    ]
    assert rebuilt["solar_bundle_count"] == 0
    assert all(not item["solar_bundle"]["generated"] for item in rebuilt["candidate_records"])
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_action_binding_is_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seeds = json.loads(
        (ROOT / config["seed_predecessor"]["path"]).read_text(encoding="utf-8")
    )
    premises = json.loads(
        (ROOT / config["premise_predecessor"]["path"]).read_text(encoding="utf-8")
    )
    target = copy.deepcopy(config["target_seeds"][0])
    target["action_sha256"] = "0" * 64
    seed = next(item for item in seeds["candidate_records"] if item["seed_id"] == target["seed_id"])
    premise = next(
        item for item in premises["candidate_records"] if item["seed_id"] == target["seed_id"]
    )
    with pytest.raises(ValueError, match="action hash mismatch"):
        _validate_target(seed, premise, target)
