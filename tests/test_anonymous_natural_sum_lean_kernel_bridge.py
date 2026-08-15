from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.anonymous_natural_sum_lean_kernel_bridge import (
    CONFIG_PATH,
    OUTPUT_PATH,
    THEOREM_PATH,
    WINNER,
    _content_sha,
    build_live_receipt,
    validate_checked_receipt,
    validate_live_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def checked_receipt():
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_checked_receipt(value, root=ROOT)
    return value


def _reseal(value):
    value["content_sha256"] = _content_sha(value)
    return value


def _live_environment() -> dict[str, str] | None:
    environment = dict(os.environ)
    if environment.get("INVARIANT_LEAN_EXECUTABLE") or shutil.which("lean"):
        return environment
    candidate = (
        Path.home()
        / ".cache"
        / "invariant"
        / "lean"
        / "v4.33.0"
        / "lean-4.33.0-windows"
        / "bin"
        / "lean.exe"
    )
    if candidate.is_file():
        environment["INVARIANT_LEAN_EXECUTABLE"] = str(candidate)
        return environment
    return None


def test_checked_receipt_binds_exact_blinded_winner(checked_receipt):
    evidence = checked_receipt["rediscovery_evidence"]
    assert evidence["winner"] == WINNER
    assert evidence["blinded_pre_unseal_root_sha256"] == (
        "63a89eef6375f105175497614900e64acf60be4fee63842ecacf1d601a16bf10"
    )
    assert evidence["sealed_before_unseal"] is True
    assert evidence["novel_theorem_claimed"] is False


def test_theorem_is_real_recursive_induction_not_reflexivity(checked_receipt):
    source = (ROOT / THEOREM_PATH).read_text(encoding="utf-8")
    assert "def anonymousNaturalSum : Nat → Nat" in source
    assert "theorem anonymousNaturalSumClosedForm (n : Nat)" in source
    assert "2 * anonymousNaturalSum n = n * (n + 1)" in source
    assert "induction n with" in source
    assert "rw [ih]" in source
    lowered = source.lower()
    assert "sorry" not in lowered
    assert "axiom " not in lowered
    assert checked_receipt["theorem_contract"]["sorry_or_axiom_used"] is False


def test_real_kernel_pass_and_exact_dependency_closure(checked_receipt):
    adapter = checked_receipt["adapter_receipt"]
    assert adapter["decision"] == "pass_lean_checked_closed_premise"
    assert adapter["execution"]["exit_code"] == 0
    assert adapter["dependency_audit"] == {
        "protocol_version": "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1",
        "reported_target": "Invariant.anonymousNaturalSumClosedForm",
        "dependencies": [
            "Invariant.anonymousNaturalSum",
            "Nat.add_mul",
            "Nat.mul_add",
            "Nat.mul_comm",
            "Nat.rec",
        ],
        "closure_valid": True,
    }
    assert checked_receipt["claim_seals"]["withheld_theorem_used_as_premise"] is False


def test_checked_historical_receipt_is_host_path_free(checked_receipt):
    assert checked_receipt["receipt_role"] == "checked_windows_historical"
    assert checked_receipt["toolchain_receipt"]["platform"] == "windows-x86_64"
    encoded = json.dumps(checked_receipt, sort_keys=True)
    assert "C:\\Users\\" not in encoded
    assert checked_receipt["toolchain_receipt"]["executable_path_persisted"] is False


def test_claims_are_bounded(checked_receipt):
    claims = checked_receipt["claim_seals"]
    assert claims["rediscovery_winner_bound"] is True
    assert claims["closed_form_induction_kernel_checked"] is True
    assert claims["dependency_closure_validated"] is True
    for key in (
        "withheld_theorem_used_as_premise",
        "novel_theorem_claimed",
        "general_rediscovery_claimed",
        "scientific_or_physics_truth_inferred",
    ):
        assert claims[key] is False


def test_optional_live_replay_passes_with_registered_lean():
    environment = _live_environment()
    if environment is None:
        pytest.skip("registered Lean is unavailable; checked receipt remains valid")
    live = build_live_receipt(ROOT / CONFIG_PATH, environment=environment)
    validate_live_receipt(live, ROOT / CONFIG_PATH, environment=environment)
    assert live["receipt_role"] == "live_replay"
    assert live["decision"] == (
        "pass_rediscovered_natural_sum_closed_form_checked_by_real_lean_kernel"
    )
    assert "C:\\Users\\" not in json.dumps(live, sort_keys=True)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["rediscovery_evidence"]["winner"].__setitem__("candidate_id", "forged"),
        lambda value: value["toolchain_receipt"].__setitem__("commit", "0" * 40),
        lambda value: value["adapter_receipt"]["dependency_audit"].__setitem__(
            "dependencies", ["False.elim"]
        ),
        lambda value: value["claim_seals"].__setitem__("withheld_theorem_used_as_premise", True),
        lambda value: value["claim_seals"].__setitem__("novel_theorem_claimed", True),
    ],
)
def test_resealed_winner_version_dependency_and_claim_tampers_fail_closed(checked_receipt, mutator):
    tampered = copy.deepcopy(checked_receipt)
    mutator(tampered)
    tampered["adapter_receipt"]["content_sha256"] = _content_sha(tampered["adapter_receipt"])
    _reseal(tampered)
    with pytest.raises(ValueError):
        validate_checked_receipt(tampered, root=ROOT)


def test_unknown_key_and_host_path_fail_closed(checked_receipt):
    unknown = copy.deepcopy(checked_receipt)
    unknown["unknown"] = True
    _reseal(unknown)
    with pytest.raises(ValueError, match="keys or seal"):
        validate_checked_receipt(unknown, root=ROOT)

    leaked = copy.deepcopy(checked_receipt)
    leaked["scope"] = r"host executable C:\Users\someone\lean.exe"
    _reseal(leaked)
    with pytest.raises(ValueError, match="host path"):
        validate_checked_receipt(leaked, root=ROOT)
