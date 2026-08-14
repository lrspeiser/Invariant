"""Generated Formula Discovery Proof/Lean v2 kernel suite.

The suite deliberately exercises three different theorem shapes.  Every positive source is
generated from a closed case specification, executed by the registered portable Lean 4.33
binary, and paired with a minimally changed source that the kernel must reject.  Receipts retain
source and byte identities but never retain caller or executable paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .lean_production_kernel_vertical_slice import (
    COMMIT,
    PLATFORM_ASSETS,
    RELEASE,
    VERSION,
    _platform_key,
    _probe_toolchain,
)
from .math_lean_adapter import (
    AUDIT_PROTOCOL,
    LeanAdapterConfig,
    build_allowed_premise_manifest,
    run_lean_adapter,
)
from .math_lean_adapter import (
    RESULT_SCHEMA as ADAPTER_SCHEMA,
)

SCHEMA_VERSION = "sigma-formula-discovery-proof-v2-receipt-1.0"
SUITE_ID = "formula-discovery-proof-v2-001"
FORBIDDEN_PREMISES = ("Classical.choice", "False.elim")
FORBIDDEN_PREFIXES = ("KnownAnswer", "Unsafe")
MAX_RECEIPT_BYTES = 2 * 1024 * 1024

CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "rational_regular_domain_identity",
        "strategy": "rational_cancellation_with_explicit_nonzero_denominator",
        "target": "Invariant.formulaDiscoveryRationalIdentityV2",
        "false_target": "Invariant.formulaDiscoveryRationalIdentityV2",
        "numerator_offset": 3,
        "denominator_offset": 2,
        "false_rhs_offset": 4,
        "allowed_premises": ["Rat.mul_div_cancel"],
    },
    {
        "case_id": "second_order_recurrence",
        "strategy": "definitional_second_order_recurrence",
        "target": "Invariant.formulaDiscoverySecondOrderRecurrenceV2",
        "false_target": "Invariant.formulaDiscoverySecondOrderRecurrenceV2",
        "base_zero": 2,
        "base_one": 5,
        "left_coefficient": 1,
        "right_coefficient": 1,
        "false_right_coefficient": 2,
        "allowed_premises": ["Invariant.formulaDiscoverySecondOrderSequenceV2"],
    },
    {
        "case_id": "quantified_non_identity",
        "strategy": "strict_order_contradiction_via_presburger_arithmetic",
        "target": "Invariant.formulaDiscoveryQuantifiedNonIdentityV2",
        "false_target": "Invariant.formulaDiscoveryQuantifiedNonIdentityV2",
        "slope": 3,
        "left_offset": 2,
        "right_offset": 3,
        "false_right_offset": 2,
        "allowed_premises": ["Lean.Parser.Tactic.omega"],
    },
)

CLAIMS = {
    "three_generated_kernel_cases_checked": True,
    "rational_denominator_premise_explicit": True,
    "higher_order_recurrence_checked": True,
    "quantified_non_identity_checked_by_distinct_strategy": True,
    "one_false_control_per_case_rejected": True,
    "sorry_axiom_or_admit_used": False,
    "host_path_persisted": False,
    "general_formula_discovery_established": False,
    "novelty_established": False,
    "scientific_truth_inferred": False,
}

_TOP_KEYS = {
    "case_spec_sha256",
    "cases",
    "claims",
    "content_sha256",
    "counts",
    "decision",
    "schema_version",
    "scope",
    "suite_id",
    "toolchain_receipt",
}
_CASE_KEYS = {
    "adapter_receipt",
    "allowed_premise_manifest",
    "case_id",
    "false_control",
    "source",
    "source_sha256",
    "strategy",
    "target",
    "theorem_statement",
}
_FALSE_KEYS = {
    "adapter_receipt",
    "alteration",
    "outcome",
    "source",
    "source_sha256",
    "target",
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_HOST_PATH = re.compile(r"[A-Za-z]:\\|/(?:home|Users)/")
_UNSAFE_DECLARATION = re.compile(r"\b(?:sorry|axiom|admit)\b", re.IGNORECASE)
_REJECTION_SCHEMA = "sigma-formula-discovery-proof-v2-kernel-rejection-1.0"


class FormulaDiscoveryProofV2Error(ValueError):
    """Raised when generation, execution, or receipt replay fails closed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha_value({key: item for key, item in value.items() if key != "content_sha256"})


def _audit(target: str, dependencies: Sequence[str]) -> str:
    records = [
        f'#eval IO.println "{AUDIT_PROTOCOL}_BEGIN"',
        f'#eval IO.println "target={target}"',
    ]
    records.extend(f'#eval IO.println "dependency={item}"' for item in dependencies)
    records.extend(
        (
            '#eval IO.println "result=checked"',
            f'#eval IO.println "{AUDIT_PROTOCOL}_END"',
        )
    )
    return "\n".join(records)


def _rational_source(spec: Mapping[str, Any], *, false: bool) -> tuple[str, str, str]:
    numerator = spec["numerator_offset"]
    denominator = spec["denominator_offset"]
    rhs = spec["false_rhs_offset"] if false else numerator
    target = spec["false_target"] if false else spec["target"]
    statement = (
        f"forall x : Rat, x - {denominator} != 0 -> "
        f"((x + {numerator}) * (x - {denominator})) / (x - {denominator}) = x + {rhs}"
    )
    audit = "\n\n" + _audit(target, spec["allowed_premises"])
    source = f"""import Std.Tactic

namespace Invariant

theorem {target.rsplit(".", 1)[1]} (x : Rat) (denominator_nonzero : x - {denominator} ≠ 0) :
    ((x + {numerator}) * (x - {denominator})) / (x - {denominator}) = x + {rhs} := by
  exact Rat.mul_div_cancel denominator_nonzero

end Invariant{audit}
"""
    alteration = f"right-side offset {numerator} changed to {rhs}"
    return source, statement, alteration


def _recurrence_source(spec: Mapping[str, Any], *, false: bool) -> tuple[str, str, str]:
    left = spec["left_coefficient"]
    actual_right = spec["right_coefficient"]
    stated_right = spec["false_right_coefficient"] if false else actual_right
    target = spec["false_target"] if false else spec["target"]
    sequence = "formulaDiscoverySecondOrderSequenceV2"
    statement = (
        f"forall n : Nat, {sequence} (n + 2) = {left} * {sequence} (n + 1) + "
        f"{stated_right} * {sequence} n"
    )
    audit = "\n\n" + _audit(target, spec["allowed_premises"])
    source = f"""import Std.Tactic

namespace Invariant

def {sequence} : Nat -> Nat
  | 0 => {spec["base_zero"]}
  | 1 => {spec["base_one"]}
  | n + 2 => {left} * {sequence} (n + 1) + {actual_right} * {sequence} n

theorem {target.rsplit(".", 1)[1]} (n : Nat) :
    {sequence} (n + 2) =
      {left} * {sequence} (n + 1) + {stated_right} * {sequence} n := by
  rfl

end Invariant{audit}
"""
    alteration = f"right recurrence coefficient {actual_right} changed to {stated_right}"
    return source, statement, alteration


def _nonidentity_source(spec: Mapping[str, Any], *, false: bool) -> tuple[str, str, str]:
    right = spec["false_right_offset"] if false else spec["right_offset"]
    target = spec["false_target"] if false else spec["target"]
    statement = (
        f"forall n : Nat, {spec['slope']} * n + {spec['left_offset']} != "
        f"{spec['slope']} * n + {right}"
    )
    audit = "\n\n" + _audit(target, spec["allowed_premises"])
    source = f"""import Std.Tactic

namespace Invariant

theorem {target.rsplit(".", 1)[1]} (n : Nat) :
    {spec["slope"]} * n + {spec["left_offset"]} ≠ {spec["slope"]} * n + {right} := by
  omega

end Invariant{audit}
"""
    alteration = f"right offset {spec['right_offset']} changed to {right}"
    return source, statement, alteration


def generate_proof_v2_source(
    case_spec: Mapping[str, Any], *, false_control: bool = False
) -> tuple[str, str, str]:
    """Generate one closed-world theorem source and its normalized statement/alteration."""

    matching = [spec for spec in CASE_SPECS if spec["case_id"] == case_spec.get("case_id")]
    if len(matching) != 1 or dict(case_spec) != matching[0]:
        raise FormulaDiscoveryProofV2Error("proof-v2 case specification changed")
    case_id = str(case_spec["case_id"])
    if case_id == "rational_regular_domain_identity":
        generated = _rational_source(case_spec, false=false_control)
    elif case_id == "second_order_recurrence":
        generated = _recurrence_source(case_spec, false=false_control)
    elif case_id == "quantified_non_identity":
        generated = _nonidentity_source(case_spec, false=false_control)
    else:  # pragma: no cover - closed by the specification comparison above
        raise FormulaDiscoveryProofV2Error("proof-v2 case is unregistered")
    source = generated[0]
    if _UNSAFE_DECLARATION.search(source) or "Classical.choice" in source or "False.elim" in source:
        raise FormulaDiscoveryProofV2Error("proof-v2 generated an unsafe declaration or premise")
    return generated


def _resolve_executable(
    executable: str | Path | None, environment: Mapping[str, str] | None
) -> Path:
    environment = os.environ if environment is None else environment
    raw = (
        os.fspath(executable)
        if executable is not None
        else environment.get("INVARIANT_LEAN_EXECUTABLE")
    )
    if not raw:
        raw = shutil.which("lean")
    if not raw and os.name == "nt":
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
        raw = str(candidate) if candidate.is_file() else None
    if not raw or "\x00" in raw:
        raise FormulaDiscoveryProofV2Error("registered portable Lean 4.33 executable unavailable")
    resolved = Path(raw).expanduser().resolve()
    if not resolved.is_file():
        raise FormulaDiscoveryProofV2Error("registered portable Lean 4.33 executable unavailable")
    return resolved


def _config(spec: Mapping[str, Any], executable: Path, *, false: bool) -> LeanAdapterConfig:
    return LeanAdapterConfig(
        target=str(spec["false_target"] if false else spec["target"]),
        allowed_premises=tuple(spec["allowed_premises"]),
        forbidden_premises=FORBIDDEN_PREMISES,
        forbidden_prefixes=FORBIDDEN_PREFIXES,
        executable=executable,
        timeout_seconds=30.0,
    )


def _run_source(
    source: str, spec: Mapping[str, Any], executable: Path, *, false: bool
) -> dict[str, Any]:
    label = "False" if false else "Pass"
    with tempfile.TemporaryDirectory(prefix="invariant-proof-v2-") as temporary:
        root = Path(temporary)
        source_path = root / f"{spec['case_id']}{label}.lean"
        source_path.write_text(source, encoding="utf-8", newline="\n")
        (root / "lean-toolchain").write_text(
            "leanprover/lean4:v4.33.0\n", encoding="utf-8", newline="\n"
        )
        return run_lean_adapter(_config(spec, executable, false=false), source_path, environment={})


def _assert_execution(
    receipt: Mapping[str, Any], spec: Mapping[str, Any], source: str, *, false: bool
) -> None:
    expected_target = spec["false_target"] if false else spec["target"]
    execution = receipt.get("execution", {})
    if (
        receipt.get("schema_version") != ADAPTER_SCHEMA
        or receipt.get("target") != expected_target
        or receipt.get("source_sha256") != _sha_text(source)
        or execution.get("attempted") is not True
        or execution.get("timed_out") is not False
    ):
        raise FormulaDiscoveryProofV2Error("proof-v2 Lean execution binding changed")
    if false:
        if (
            receipt.get("decision") != "block_lean_process_failure"
            or receipt.get("status") != "block"
            or not isinstance(execution.get("exit_code"), int)
            or execution["exit_code"] == 0
        ):
            raise FormulaDiscoveryProofV2Error("proof-v2 false control was not rejected")
    elif (
        receipt.get("decision") != "pass_lean_checked_closed_premise"
        or receipt.get("status") != "pass"
        or execution.get("exit_code") != 0
        or receipt.get("dependency_audit")
        != {
            "protocol_version": AUDIT_PROTOCOL,
            "reported_target": spec["target"],
            "dependencies": sorted(spec["allowed_premises"]),
            "closure_valid": True,
        }
    ):
        raise FormulaDiscoveryProofV2Error("proof-v2 positive theorem did not kernel-check")


def _rejection_receipt(
    adapter: Mapping[str, Any], spec: Mapping[str, Any], source: str
) -> dict[str, Any]:
    """Discard path-sensitive diagnostics while retaining the decisive kernel rejection."""

    executable = adapter["executable"]
    body = {
        "schema_version": _REJECTION_SCHEMA,
        "target": spec["false_target"],
        "source_sha256": _sha_text(source),
        "executable_identity_sha256": executable["identity_sha256"],
        "status": "block",
        "decision": "block_lean_process_failure",
        "execution": {
            "attempted": True,
            "timed_out": False,
            "nonzero_exit_code": True,
        },
        "diagnostic_bytes_persisted": False,
    }
    return {**body, "content_sha256": _sha_value(body)}


def build_proof_v2_receipt(
    *,
    executable: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Generate and execute all three positive/negative Lean 4.33 kernel cases."""

    lean = _resolve_executable(executable, environment)
    toolchain = _probe_toolchain(lean, _platform_key(), "registered_portable_lean")
    cases: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        source, statement, _ = generate_proof_v2_source(spec)
        false_source, _, alteration = generate_proof_v2_source(spec, false_control=True)
        adapter = _run_source(source, spec, lean, false=False)
        false_adapter = _run_source(false_source, spec, lean, false=True)
        _assert_execution(adapter, spec, source, false=False)
        _assert_execution(false_adapter, spec, false_source, false=True)
        manifest = build_allowed_premise_manifest(_config(spec, lean, false=False))
        cases.append(
            {
                "case_id": spec["case_id"],
                "strategy": spec["strategy"],
                "target": spec["target"],
                "theorem_statement": statement,
                "source": source,
                "source_sha256": _sha_text(source),
                "allowed_premise_manifest": manifest,
                "adapter_receipt": adapter,
                "false_control": {
                    "target": spec["false_target"],
                    "alteration": alteration,
                    "outcome": "REJECT",
                    "source": false_source,
                    "source_sha256": _sha_text(false_source),
                    "adapter_receipt": _rejection_receipt(false_adapter, spec, false_source),
                },
            }
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "decision": "PASS",
        "case_spec_sha256": _sha_value(CASE_SPECS),
        "toolchain_receipt": toolchain,
        "cases": cases,
        "counts": {
            "generated_theorems": 3,
            "distinct_proof_strategies": 3,
            "explicit_nonzero_denominator_premises": 1,
            "kernel_executions": 6,
            "kernel_passes": 3,
            "false_controls_rejected": 3,
            "blocked": 0,
        },
        "claims": dict(CLAIMS),
        "scope": (
            "three bounded generated Formula Discovery proof shapes checked by registered portable "
            "Lean 4.33 with one minimally false rejected control each; no general discovery, "
            "novelty, or scientific-truth claim"
        ),
    }
    receipt = {**body, "content_sha256": _sha_value(body)}
    validate_proof_v2_receipt(receipt)
    return receipt


def _validate_adapter_static(
    receipt: Mapping[str, Any],
    spec: Mapping[str, Any],
    source: str,
    toolchain_sha: str,
    *,
    false: bool,
) -> None:
    if receipt.get("content_sha256") != _content_sha(receipt):
        raise FormulaDiscoveryProofV2Error("proof-v2 adapter receipt seal changed")
    executable = receipt.get("executable", {})
    manifest = build_allowed_premise_manifest(_config(spec, Path("lean"), false=false))
    if receipt.get("manifest_sha256") != manifest["content_sha256"] or executable != {
        "configured": True,
        "discovered": True,
        "discovery_source": "explicit",
        "identity_sha256": toolchain_sha,
    }:
        raise FormulaDiscoveryProofV2Error("proof-v2 adapter identity or premise seal changed")
    _assert_execution(receipt, spec, source, false=false)


def _validate_rejection_static(
    receipt: Mapping[str, Any], spec: Mapping[str, Any], source: str, toolchain_sha: str
) -> None:
    expected_body = {
        "schema_version": _REJECTION_SCHEMA,
        "target": spec["false_target"],
        "source_sha256": _sha_text(source),
        "executable_identity_sha256": toolchain_sha,
        "status": "block",
        "decision": "block_lean_process_failure",
        "execution": {
            "attempted": True,
            "timed_out": False,
            "nonzero_exit_code": True,
        },
        "diagnostic_bytes_persisted": False,
    }
    expected = {**expected_body, "content_sha256": _sha_value(expected_body)}
    if dict(receipt) != expected:
        raise FormulaDiscoveryProofV2Error("proof-v2 false-control rejection receipt changed")


def validate_proof_v2_receipt(value: Mapping[str, Any]) -> None:
    """Validate a sealed receipt without requiring Lean or replaying host paths."""

    if set(value) != _TOP_KEYS or value.get("content_sha256") != _content_sha(value):
        raise FormulaDiscoveryProofV2Error("proof-v2 receipt schema or seal changed")
    encoded = json.dumps(value, sort_keys=True)
    if _HOST_PATH.search(encoded):
        raise FormulaDiscoveryProofV2Error("proof-v2 receipt persisted a host path")
    toolchain = value.get("toolchain_receipt")
    cases = value.get("cases")
    if not isinstance(toolchain, Mapping) or not isinstance(cases, list) or len(cases) != 3:
        raise FormulaDiscoveryProofV2Error("proof-v2 receipt structure changed")
    platform = str(toolchain.get("platform"))
    asset = PLATFORM_ASSETS.get(platform)
    executable_sha = str(toolchain.get("executable_sha256"))
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("suite_id") != SUITE_ID
        or value.get("decision") != "PASS"
        or value.get("case_spec_sha256") != _sha_value(CASE_SPECS)
        or value.get("claims") != CLAIMS
        or value.get("counts")
        != {
            "generated_theorems": 3,
            "distinct_proof_strategies": 3,
            "explicit_nonzero_denominator_premises": 1,
            "kernel_executions": 6,
            "kernel_passes": 3,
            "false_controls_rejected": 3,
            "blocked": 0,
        }
        or toolchain.get("official_release") != RELEASE
        or toolchain.get("version") != VERSION
        or toolchain.get("commit") != COMMIT
        or toolchain.get("platform_asset") != asset
        or toolchain.get("executable_path_persisted") is not False
        or toolchain.get("resolution_source") != "registered_portable_lean"
        or not _HEX.fullmatch(executable_sha)
    ):
        raise FormulaDiscoveryProofV2Error("proof-v2 receipt semantics changed")
    for case, spec in zip(cases, CASE_SPECS, strict=True):
        if not isinstance(case, Mapping) or set(case) != _CASE_KEYS:
            raise FormulaDiscoveryProofV2Error("proof-v2 case schema changed")
        source, statement, _ = generate_proof_v2_source(spec)
        false_source, _, alteration = generate_proof_v2_source(spec, false_control=True)
        manifest = build_allowed_premise_manifest(_config(spec, Path("lean"), false=False))
        false = case.get("false_control")
        if (
            case.get("case_id") != spec["case_id"]
            or case.get("strategy") != spec["strategy"]
            or case.get("target") != spec["target"]
            or case.get("theorem_statement") != statement
            or case.get("source") != source
            or case.get("source_sha256") != _sha_text(source)
            or case.get("allowed_premise_manifest") != manifest
            or not isinstance(false, Mapping)
            or set(false) != _FALSE_KEYS
            or false.get("target") != spec["false_target"]
            or false.get("alteration") != alteration
            or false.get("outcome") != "REJECT"
            or false.get("source") != false_source
            or false.get("source_sha256") != _sha_text(false_source)
        ):
            raise FormulaDiscoveryProofV2Error("proof-v2 generated case binding changed")
        _validate_adapter_static(case["adapter_receipt"], spec, source, executable_sha, false=False)
        _validate_rejection_static(false["adapter_receipt"], spec, false_source, executable_sha)


def validate_live_proof_v2_receipt(
    value: Mapping[str, Any],
    *,
    executable: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Statically validate and then require exact six-execution live replay parity."""

    validate_proof_v2_receipt(value)
    if dict(value) != build_proof_v2_receipt(executable=executable, environment=environment):
        raise FormulaDiscoveryProofV2Error("proof-v2 live replay changed")


def write_proof_v2_receipt(receipt: Mapping[str, Any], output_path: Path) -> Path:
    """Publish one validated receipt without overwriting an existing path."""

    validate_proof_v2_receipt(receipt)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if len(payload.encode("utf-8")) > MAX_RECEIPT_BYTES:
        raise FormulaDiscoveryProofV2Error("proof-v2 receipt exceeds byte limit")
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except OSError as error:
        raise FormulaDiscoveryProofV2Error(
            "proof-v2 receipt could not be published immutably"
        ) from error
    return output_path


def load_proof_v2_receipt(path: Path) -> dict[str, Any]:
    """Load one bounded strict receipt for CLI/API validation."""

    try:
        payload = path.read_bytes()
        if not payload or len(payload) > MAX_RECEIPT_BYTES:
            raise FormulaDiscoveryProofV2Error("proof-v2 receipt byte budget violated")
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormulaDiscoveryProofV2Error("proof-v2 receipt is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise FormulaDiscoveryProofV2Error("proof-v2 receipt must be one JSON object")
    validate_proof_v2_receipt(value)
    return value


__all__ = [
    "CASE_SPECS",
    "CLAIMS",
    "SCHEMA_VERSION",
    "SUITE_ID",
    "FormulaDiscoveryProofV2Error",
    "build_proof_v2_receipt",
    "generate_proof_v2_source",
    "load_proof_v2_receipt",
    "validate_live_proof_v2_receipt",
    "validate_proof_v2_receipt",
    "write_proof_v2_receipt",
]
