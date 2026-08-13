from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.synthetic_modular_exponent_lean_kernel_bridge import (
    CLAIMS,
    CONFIG_PATH,
    OUTPUT_PATH,
    THEOREM_PATH,
    _content_sha,
    build_live_receipt,
    validate_checked_receipt,
    validate_live_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str] | None:
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
    if not candidate.is_file():
        return None
    environment["INVARIANT_LEAN_EXECUTABLE"] = str(candidate)
    return environment


@pytest.fixture(scope="module")
def checked() -> dict:
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_checked_receipt(value, root=ROOT)
    return value


def test_receipt_binds_blinded_winner_and_all_obligations(checked: dict) -> None:
    evidence = checked["rediscovery_evidence"]
    assert evidence["modulus"] == 11
    assert evidence["exponent"] == 10
    assert evidence["residues_checked"] == 10
    assert evidence["reference_unseal_ordinal"] == 6
    assert checked["counts"]["finite_residue_obligations"] == 10


def test_theorem_is_nontrivial_finite_modular_statement(checked: dict) -> None:
    source = (ROOT / THEOREM_PATH).read_text(encoding="utf-8")
    assert "∀ a : Fin 11, a ≠ 0 → a ^ 10 = 1" in source
    assert "native_decide" in source
    assert "target=Invariant.anonymousModularExponent" in source
    assert "dependency=of_decide_eq_true" in source
    assert "sorry" not in source.lower()
    assert "axiom " not in source.lower()
    assert checked["theorem_contract"]["sorry_or_axiom_used"] is False


def test_real_kernel_and_closed_dependency_receipt(checked: dict) -> None:
    adapter = checked["adapter_receipt"]
    assert adapter["decision"] == "pass_lean_checked_closed_premise"
    assert adapter["execution"]["exit_code"] == 0
    assert adapter["dependency_audit"] == {
        "protocol_version": "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1",
        "reported_target": "Invariant.anonymousModularExponent",
        "dependencies": ["of_decide_eq_true"],
        "closure_valid": True,
    }


def test_claims_are_bounded(checked: dict) -> None:
    assert checked["claims"] == CLAIMS
    assert CLAIMS["ten_residue_obligations_kernel_checked"] is True
    assert CLAIMS["withheld_theorem_used_as_premise"] is False
    assert CLAIMS["novel_theorem_claimed"] is False
    assert CLAIMS["general_number_theory_claimed"] is False


def test_checked_receipt_is_host_path_free(checked: dict) -> None:
    encoded = json.dumps(checked, sort_keys=True)
    assert "C:\\Users\\" not in encoded
    assert checked["toolchain_receipt"]["executable_path_persisted"] is False


def test_optional_live_replay() -> None:
    environment = _environment()
    if environment is None:
        pytest.skip("registered Lean unavailable; portable checked receipt remains valid")
    live = build_live_receipt(ROOT / CONFIG_PATH, environment=environment)
    validate_live_receipt(live, ROOT / CONFIG_PATH, environment=environment)
    assert live["receipt_role"] == "live_replay"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["rediscovery_evidence"].__setitem__("exponent", 9),
        lambda value: value["adapter_receipt"]["dependency_audit"].__setitem__(
            "dependencies", ["False.elim"]
        ),
        lambda value: value["claims"].__setitem__("novel_theorem_claimed", True),
        lambda value: value["theorem_contract"].__setitem__("sorry_or_axiom_used", True),
    ],
)
def test_resealed_semantic_tampers_fail_closed(checked: dict, mutator) -> None:
    tampered = copy.deepcopy(checked)
    mutator(tampered)
    tampered["adapter_receipt"]["content_sha256"] = _content_sha(tampered["adapter_receipt"])
    tampered["content_sha256"] = _content_sha(tampered)
    with pytest.raises(ValueError):
        validate_checked_receipt(tampered, root=ROOT)
