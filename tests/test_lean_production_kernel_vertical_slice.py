from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.lean_production_kernel_vertical_slice import (
    CONFIG_PATH,
    OUTPUT_PATH,
    THEOREM_PATH,
    _content_sha,
    build_receipt,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def receipt():
    return build_receipt(ROOT / CONFIG_PATH)


def _reseal(value):
    value["content_sha256"] = _content_sha(value)
    return value


def test_artifact_matches_live_adapter_receipt(receipt):
    artifact = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    assert artifact == receipt
    validate_receipt(artifact, root=ROOT)


def test_real_known_answer_source_is_bound(receipt):
    source = (ROOT / THEOREM_PATH).read_text(encoding="utf-8")
    assert "theorem kernelSmoke (n : Nat) : n = n := Eq.refl n" in source
    assert "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN" in source
    assert receipt["theorem_contract"]["proof_term"] == "Eq.refl n"
    assert receipt["dependency_closure"]["declared_dependencies"] == ["Eq.refl"]


def test_pinned_real_lean_kernel_and_dependency_closure_pass(receipt):
    assert receipt["decision"] == "pass_real_lean_kernel_vertical_slice"
    assert receipt["first_blocker"] == "none_for_bounded_kernel_smoke_theorem"
    adapter = receipt["adapter_receipt"]
    assert adapter["decision"] == "pass_lean_checked_closed_premise"
    assert adapter["executable"] == {
        "configured": True,
        "discovered": True,
        "discovery_source": "explicit",
        "identity_sha256": ("dd86e9b24990b1da425ea4af910f016e4db8f9a25c9ddad27bc6bee3690e677f"),
    }
    assert adapter["execution"]["attempted"] is True
    assert adapter["execution"]["exit_code"] == 0
    assert adapter["dependency_audit"] == {
        "protocol_version": "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1",
        "reported_target": "Invariant.kernelSmoke",
        "dependencies": ["Eq.refl"],
        "closure_valid": True,
    }


def test_official_toolchain_identity_is_live_and_outside_git(receipt):
    toolchain = receipt["toolchain_receipt"]
    assert toolchain["official_release"] == "leanprover/lean4:v4.33.0"
    assert toolchain["archive_bytes"] == 583425362
    assert toolchain["archive_sha256"] == (
        "60d045a2ef45fca55a620b7d55be682e8439ec8d1fc9a8bcd2615da7dffba26a"
    )
    assert toolchain["live_executable_sha256"] == toolchain["executable_sha256"]
    assert toolchain["version_probe_exit_code"] == 0
    assert toolchain["archive_and_binary_outside_git_history"] is True
    assert ".cache\\invariant\\lean\\v4.33.0" in toolchain["executable_path"]


def test_counts_and_claims_are_conservative(receipt):
    assert receipt["readiness_counts"] == {
        "theorems_presented": 1,
        "kernel_executions_attempted": 1,
        "kernel_checked_theorems": 1,
        "blocked_theorems": 0,
        "rejected_theorems": 0,
    }
    assert receipt["claim_seals"] == {
        "lean_executable_discovered": True,
        "bounded_theorem_kernel_checked": True,
        "dependency_closure_validated": True,
        "scientific_truth_inferred": False,
        "general_formal_completion_claimed": False,
        "physics_claimed": False,
    }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["claim_seals"].__setitem__("bounded_theorem_kernel_checked", False),
        lambda value: value["readiness_counts"].__setitem__("kernel_checked_theorems", 0),
        lambda value: value["dependency_closure"].__setitem__("closure_checked_by_adapter", False),
        lambda value: value["adapter_receipt"]["claims"].__setitem__(
            "formal_target_checked", False
        ),
    ],
)
def test_resealed_overclaim_tamper_fails_closed(receipt, mutator):
    tampered = copy.deepcopy(receipt)
    mutator(tampered)
    _reseal(tampered)
    with pytest.raises(ValueError, match="differs from live adapter result"):
        validate_receipt(tampered, root=ROOT)


def test_unknown_key_fails_closed(receipt):
    tampered = copy.deepcopy(receipt)
    tampered["unknown"] = True
    _reseal(tampered)
    with pytest.raises(ValueError, match="result keys changed"):
        validate_receipt(tampered, root=ROOT)
