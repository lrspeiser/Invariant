"""Bridge the sealed natural-sum rediscovery winner into a real Lean kernel proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .anonymous_natural_sum_blind_rediscovery import validate_result as validate_rediscovery
from .lean_production_kernel_vertical_slice import (
    ADAPTER_BINDING,
    COMMIT,
    PLATFORM_ASSETS,
    RELEASE,
    VERSION,
    _platform_key,
    _probe_toolchain,
    _resolve_executable,
)
from .lean_production_kernel_vertical_slice import (
    _validate_config as validate_lean_runtime_config,
)
from .lean_production_kernel_vertical_slice import (
    validate_checked_receipt as validate_lean_runtime_receipt,
)
from .math_lean_adapter import (
    AUDIT_PROTOCOL,
    LeanAdapterConfig,
    build_allowed_premise_manifest,
    run_lean_adapter,
)
from .math_lean_adapter import (
    RESULT_SCHEMA as LEAN_RESULT_SCHEMA,
)

CONFIG_SCHEMA = "invariant-anonymous-natural-sum-lean-kernel-bridge-config-1.0"
RESULT_SCHEMA = "invariant-anonymous-natural-sum-lean-kernel-bridge-result-1.0"
CAMPAIGN_ID = "anonymous-natural-sum-lean-kernel-bridge-001"
CONFIG_PATH = "configs/math/anonymous_natural_sum_lean_kernel_bridge.json"
SOURCE_PATH = "src/sigma_theory_compiler/anonymous_natural_sum_lean_kernel_bridge.py"
TEST_PATH = "tests/test_anonymous_natural_sum_lean_kernel_bridge.py"
THEOREM_PATH = "formal/lean/AnonymousNaturalSumClosedForm.lean"
OUTPUT_PATH = "runs/math/anonymous-natural-sum-lean-kernel-bridge/receipt.json"
TARGET = "Invariant.anonymousNaturalSumClosedForm"
ALLOWED_PREMISES = (
    "Invariant.anonymousNaturalSum",
    "Nat.add_mul",
    "Nat.mul_add",
    "Nat.mul_comm",
    "Nat.rec",
)
FORBIDDEN_PREMISES = ("Classical.choice", "False.elim")
FORBIDDEN_PREFIXES = ("KnownAnswer", "Unsafe")
WINNER = {
    "candidate_id": "f48406b1add8e3e88d5b86c0",
    "coefficients": {
        "constant": {"denominator": 1, "numerator": 0},
        "linear": {"denominator": 2, "numerator": 1},
        "square": {"denominator": 2, "numerator": 1},
    },
}
WINNER_CONTRACT = {
    "candidate_id": WINNER["candidate_id"],
    "square": "1/2",
    "linear": "1/2",
    "constant": "0",
    "scaled_lean_identity": "2*S(n)=n*(n+1)",
}
REDISCOVERY_BINDING = {
    "source": {
        "path": "src/sigma_theory_compiler/anonymous_natural_sum_blind_rediscovery.py",
        "file_sha256": "0a965887d1b604d0b15ac9b60c766b567890a6089c9e52343f3e0479f67a023d",
    },
    "config": {
        "path": "configs/backgrounds/anonymous_natural_sum_blind_rediscovery.json",
        "file_sha256": "46a982033c92636a11293fec4cf4266ce99915f07a6fc47349269ff5551d8656",
    },
    "test": {
        "path": "tests/test_anonymous_natural_sum_blind_rediscovery.py",
        "file_sha256": "a0544115a5b40ea732259196e7024dce267784bd08e1c0f5e780f0b2acae5f13",
    },
    "artifact": {
        "path": "runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json",
        "file_sha256": "3158d6031ad0dbf1c3cb955c956af319f39c58d50982d2193a07b0f46c83e685",
        "content_sha256": "05b7ab12f3876513216394dd578258ea6af242d0b8e6a4911e281a4d713bd5be",
    },
}
LEAN_RUNTIME_BINDING = {
    "source": {
        "path": "src/sigma_theory_compiler/lean_production_kernel_vertical_slice.py",
        "file_sha256": "06db4e93bd277019f8c5f803a3b676f1953a909862e458a137da5ad868ae3d36",
    },
    "config": {
        "path": "configs/math/lean_production_kernel_vertical_slice.json",
        "file_sha256": "dd1c183ff0ace9be3c5688e029ead3bc34a7cf6cb16370cc989c243c26a39a0d",
    },
    "test": {
        "path": "tests/test_lean_production_kernel_vertical_slice.py",
        "file_sha256": "bdc5264da89ec86434478b7f0c54581ab915a897c498b2264b693979a9c9a039",
    },
    "receipt": {
        "path": "runs/math/lean-production-kernel-vertical-slice/receipt.json",
        "file_sha256": "f34d10c772282ca8a832d848d29d90ee354a98371922d660daa38b4b105f530a",
        "content_sha256": "2ab97edf548703abfe05e03e606fbbd6842384f57ff82a5d306cbf15ac51bc55",
    },
}
_RESULT_KEYS = {
    "adapter_receipt",
    "campaign_id",
    "claim_seals",
    "content_sha256",
    "decision",
    "dependency_closure",
    "first_blocker",
    "readiness_counts",
    "receipt_role",
    "rediscovery_evidence",
    "schema_version",
    "scope",
    "source_bindings",
    "theorem_contract",
    "toolchain_receipt",
}
_ADAPTER_KEYS = {
    "claims",
    "content_sha256",
    "decision",
    "dependency_audit",
    "executable",
    "execution",
    "manifest_sha256",
    "reason",
    "schema_version",
    "source_sha256",
    "status",
    "target",
}
_AUDIT_OUTPUT = (
    f"{AUDIT_PROTOCOL}_BEGIN\n"
    f"target={TARGET}\n"
    "dependency=Invariant.anonymousNaturalSum\n"
    "dependency=Nat.rec\n"
    "dependency=Nat.mul_add\n"
    "dependency=Nat.add_mul\n"
    "dependency=Nat.mul_comm\n"
    "result=checked\n"
    f"{AUDIT_PROTOCOL}_END\n"
).encode()
_AUDIT_STDOUT_SHA256 = hashlib.sha256(_AUDIT_OUTPUT).hexdigest()
_VERSION_PATTERN = re.compile(
    rf"Lean \(version {re.escape(VERSION)}, [^,]+, commit {COMMIT}, Release\)\Z"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("natural-sum Lean bridge path escapes project root")
    return path


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("natural-sum Lean bridge artifact file binding changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise ValueError("natural-sum Lean bridge artifact content binding changed")
    return value


def _validate_file_bundle(root: Path, bundle: Mapping[str, Any]) -> None:
    for label, binding in bundle.items():
        if label in {"artifact", "receipt"}:
            continue
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("natural-sum Lean bridge dependency file changed")


def _validate_config(config: Mapping[str, Any], *, root: Path) -> None:
    if config != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "theorem_source_path": THEOREM_PATH,
        "target": TARGET,
        "timeout_seconds": 30,
        "premise_policy": {
            "allowed_premises": list(ALLOWED_PREMISES),
            "equivalent_targets": [],
            "forbidden_premises": list(FORBIDDEN_PREMISES),
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        },
        "winner_contract": WINNER_CONTRACT,
        "rediscovery_binding": REDISCOVERY_BINDING,
        "lean_runtime_binding": LEAN_RUNTIME_BINDING,
    }:
        raise ValueError("natural-sum Lean bridge config boundary changed")
    _validate_file_bundle(root, REDISCOVERY_BINDING)
    _validate_file_bundle(root, LEAN_RUNTIME_BINDING)


def _load_evidence(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rediscovery = _load_bound(root, REDISCOVERY_BINDING["artifact"])
    validate_rediscovery(
        rediscovery,
        root,
        _inside(root, REDISCOVERY_BINDING["config"]["path"]),
    )
    if (
        rediscovery.get("decision")
        != "pass_blind_bounded_grammar_rediscovery_independently_proved_before_unseal"
        or rediscovery.get("winner") != WINNER
        or rediscovery.get("induction_proof", {}).get("successor_identity")
        != "candidate(n+1)-candidate(n)=n+1"
        or rediscovery.get("induction_proof", {}).get("successor_identity_proved") is not True
        or rediscovery.get("claims", {}).get("winner_sealed_before_reference_access") is not True
        or rediscovery.get("claims", {}).get("novel_theorem_claimed") is not False
    ):
        raise ValueError("natural-sum Lean bridge rediscovery evidence changed")
    lean_config_path = _inside(root, LEAN_RUNTIME_BINDING["config"]["path"])
    lean_config = json.loads(lean_config_path.read_text(encoding="utf-8"))
    validate_lean_runtime_config(lean_config, root=root)
    lean_receipt = _load_bound(root, LEAN_RUNTIME_BINDING["receipt"])
    validate_lean_runtime_receipt(lean_receipt, root=root)
    return rediscovery, lean_config, lean_receipt


def _adapter_config(config: Mapping[str, Any], executable: Path) -> LeanAdapterConfig:
    policy = config["premise_policy"]
    return LeanAdapterConfig(
        target=TARGET,
        allowed_premises=tuple(policy["allowed_premises"]),
        equivalent_targets=tuple(policy["equivalent_targets"]),
        forbidden_premises=tuple(policy["forbidden_premises"]),
        forbidden_prefixes=tuple(policy["forbidden_prefixes"]),
        executable=executable,
        timeout_seconds=float(config["timeout_seconds"]),
    )


def build_live_receipt(
    config_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
    receipt_role: str = "live_replay",
) -> dict[str, Any]:
    if receipt_role not in {"live_replay", "checked_windows_historical"}:
        raise ValueError("natural-sum Lean bridge receipt role changed")
    root = config_path.resolve().parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, root=root)
    rediscovery, lean_config, lean_receipt = _load_evidence(root)
    del lean_receipt
    executable, resolution_source = _resolve_executable(lean_config, environment=environment)
    if executable is None or resolution_source is None:
        raise ValueError("natural-sum Lean bridge executable was not resolved")
    toolchain_receipt = _probe_toolchain(executable, _platform_key(), resolution_source)
    theorem_path = _inside(root, THEOREM_PATH)
    adapter_config = _adapter_config(config, executable)
    manifest = build_allowed_premise_manifest(adapter_config)
    adapter_receipt = run_lean_adapter(adapter_config, theorem_path, environment={})
    if adapter_receipt.get("decision") != "pass_lean_checked_closed_premise":
        raise ValueError(
            f"natural-sum Lean bridge adapter did not pass: {adapter_receipt.get('decision')}"
        )
    rediscovery_summary = {
        "benchmark_id": rediscovery["benchmark_id"],
        "artifact_content_sha256": rediscovery["content_sha256"],
        "blinded_pre_unseal_root_sha256": rediscovery["blinded_pre_unseal_root_sha256"],
        "winner": rediscovery["winner"],
        "winner_formula": rediscovery["pre_unseal"]["discovery"]["winner_formula"],
        "successor_identity": rediscovery["induction_proof"]["successor_identity"],
        "sealed_before_unseal": rediscovery["claims"]["winner_sealed_before_reference_access"],
        "novel_theorem_claimed": False,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "receipt_role": receipt_role,
        "decision": "pass_rediscovered_natural_sum_closed_form_checked_by_real_lean_kernel",
        "first_blocker": "none_for_the_bounded_rediscovery_to_kernel_bridge",
        "rediscovery_evidence": rediscovery_summary,
        "theorem_contract": {
            "target": TARGET,
            "recursive_definition": (
                "S(0)=0 and S(n+1)=S(n)+(n+1), matching the anonymous benchmark"
            ),
            "winner_formula": "S(n)=(n^2+n)/2",
            "kernel_statement": "2*S(n)=n*(n+1) for every n:Nat",
            "proof_method": "Nat induction with explicit distributivity and commutativity rewrites",
            "source_sha256": _file_sha(theorem_path),
            "source_bytes": theorem_path.stat().st_size,
            "sorry_or_axiom_used": False,
        },
        "dependency_closure": {
            "allowed_premise_manifest": manifest,
            "declared_dependencies": sorted(ALLOWED_PREMISES),
            "closure_checked_by_adapter": True,
            "withheld_theorem_used_as_premise": False,
            "out_of_manifest_dependency_allowed": False,
        },
        "adapter_receipt": adapter_receipt,
        "toolchain_receipt": toolchain_receipt,
        "readiness_counts": {
            "rediscovery_winners_bound": 1,
            "induction_theorems_presented": 1,
            "kernel_executions_attempted": 1,
            "kernel_checked_theorems": 1,
            "blocked_theorems": 0,
            "rejected_theorems": 0,
        },
        "claim_seals": {
            "rediscovery_winner_bound": True,
            "closed_form_induction_kernel_checked": True,
            "dependency_closure_validated": True,
            "withheld_theorem_used_as_premise": False,
            "novel_theorem_claimed": False,
            "general_rediscovery_claimed": False,
            "scientific_or_physics_truth_inferred": False,
        },
        "source_bindings": {
            "source": {
                "path": SOURCE_PATH,
                "file_sha256": _file_sha(_inside(root, SOURCE_PATH)),
            },
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {
                "path": TEST_PATH,
                "file_sha256": _file_sha(_inside(root, TEST_PATH)),
            },
            "theorem": {
                "path": THEOREM_PATH,
                "file_sha256": _file_sha(theorem_path),
            },
            "rediscovery": REDISCOVERY_BINDING,
            "lean_runtime": LEAN_RUNTIME_BINDING,
            "math_lean_adapter": ADAPTER_BINDING,
        },
        "scope": (
            "the single sealed winner f48406b1add8e3e88d5b86c0 and its natural-number "
            "recursive sum closed form only; no novelty, unbounded grammar, general rediscovery, "
            "scientific, or physics claim"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_adapter(
    receipt: Mapping[str, Any], theorem_sha: str, toolchain: Mapping[str, Any]
) -> None:
    if set(receipt) != _ADAPTER_KEYS or receipt.get("content_sha256") != _content_sha(receipt):
        raise ValueError("natural-sum Lean bridge adapter schema or seal changed")
    expected_manifest = build_allowed_premise_manifest(
        LeanAdapterConfig(
            target=TARGET,
            allowed_premises=ALLOWED_PREMISES,
            forbidden_premises=FORBIDDEN_PREMISES,
            forbidden_prefixes=FORBIDDEN_PREFIXES,
        )
    )
    if (
        receipt.get("schema_version") != LEAN_RESULT_SCHEMA
        or receipt.get("status") != "pass"
        or receipt.get("decision") != "pass_lean_checked_closed_premise"
        or receipt.get("reason") != "Lean checked the target and reported only allowed premises"
        or receipt.get("target") != TARGET
        or receipt.get("manifest_sha256") != expected_manifest["content_sha256"]
        or receipt.get("source_sha256") != theorem_sha
        or receipt.get("executable")
        != {
            "configured": True,
            "discovered": True,
            "discovery_source": "explicit",
            "identity_sha256": toolchain["executable_sha256"],
        }
        or receipt.get("execution")
        != {
            "attempted": True,
            "timed_out": False,
            "exit_code": 0,
            "stdout_sha256": _AUDIT_STDOUT_SHA256,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        }
        or receipt.get("dependency_audit")
        != {
            "protocol_version": AUDIT_PROTOCOL,
            "reported_target": TARGET,
            "dependencies": sorted(ALLOWED_PREMISES),
            "closure_valid": True,
        }
        or receipt.get("claims")
        != {
            "lean_available": True,
            "formal_target_checked": True,
            "scientific_truth_inferred": False,
        }
    ):
        raise ValueError("natural-sum Lean bridge adapter semantics changed")


def validate_checked_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    if set(value) != _RESULT_KEYS or value.get("content_sha256") != _content_sha(value):
        raise ValueError("natural-sum Lean bridge result keys or seal changed")
    encoded = json.dumps(value, sort_keys=True)
    if re.search(r"[A-Za-z]:\\|/(?:home|Users)/", encoded):
        raise ValueError("natural-sum Lean bridge receipt persisted a host path")
    config_path = _inside(root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config, root=root)
    rediscovery, _lean_config, _lean_receipt = _load_evidence(root)
    theorem_path = _inside(root, THEOREM_PATH)
    theorem_sha = _file_sha(theorem_path)
    toolchain = value.get("toolchain_receipt")
    if not isinstance(toolchain, Mapping):
        raise TypeError("natural-sum Lean bridge toolchain receipt missing")
    platform_key = str(toolchain.get("platform"))
    asset = PLATFORM_ASSETS.get(platform_key)
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("receipt_role")
        not in {
            "checked_windows_historical",
            "live_replay",
        }
        or value.get("decision")
        != "pass_rediscovered_natural_sum_closed_form_checked_by_real_lean_kernel"
        or value.get("first_blocker") != "none_for_the_bounded_rediscovery_to_kernel_bridge"
        or asset is None
        or toolchain.get("official_release") != RELEASE
        or toolchain.get("version") != VERSION
        or toolchain.get("commit") != COMMIT
        or toolchain.get("platform_asset") != asset
        or not _VERSION_PATTERN.fullmatch(str(toolchain.get("version_output")))
        or toolchain.get("version_probe_exit_code") != 0
        or toolchain.get("resolution_source") not in {"environment", "PATH"}
        or toolchain.get("executable_path_persisted") is not False
        or (
            value.get("receipt_role") == "checked_windows_historical"
            and platform_key != "windows-x86_64"
        )
    ):
        raise ValueError("natural-sum Lean bridge toolchain semantics changed")
    executable_sha = toolchain.get("executable_sha256")
    if (
        not isinstance(executable_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", executable_sha)
        or (asset["executable_sha256"] is not None and executable_sha != asset["executable_sha256"])
    ):
        raise ValueError("natural-sum Lean bridge executable identity changed")
    _validate_adapter(value["adapter_receipt"], theorem_sha, toolchain)
    expected_summary = {
        "benchmark_id": rediscovery["benchmark_id"],
        "artifact_content_sha256": rediscovery["content_sha256"],
        "blinded_pre_unseal_root_sha256": rediscovery["blinded_pre_unseal_root_sha256"],
        "winner": WINNER,
        "winner_formula": "a*n^2+b*n+c",
        "successor_identity": "candidate(n+1)-candidate(n)=n+1",
        "sealed_before_unseal": True,
        "novel_theorem_claimed": False,
    }
    expected_manifest = build_allowed_premise_manifest(
        LeanAdapterConfig(
            target=TARGET,
            allowed_premises=ALLOWED_PREMISES,
            forbidden_premises=FORBIDDEN_PREMISES,
            forbidden_prefixes=FORBIDDEN_PREFIXES,
        )
    )
    if (
        value.get("rediscovery_evidence") != expected_summary
        or value.get("theorem_contract")
        != {
            "target": TARGET,
            "recursive_definition": (
                "S(0)=0 and S(n+1)=S(n)+(n+1), matching the anonymous benchmark"
            ),
            "winner_formula": "S(n)=(n^2+n)/2",
            "kernel_statement": "2*S(n)=n*(n+1) for every n:Nat",
            "proof_method": (
                "Nat induction with explicit distributivity and commutativity rewrites"
            ),
            "source_sha256": theorem_sha,
            "source_bytes": theorem_path.stat().st_size,
            "sorry_or_axiom_used": False,
        }
        or value.get("dependency_closure")
        != {
            "allowed_premise_manifest": expected_manifest,
            "declared_dependencies": sorted(ALLOWED_PREMISES),
            "closure_checked_by_adapter": True,
            "withheld_theorem_used_as_premise": False,
            "out_of_manifest_dependency_allowed": False,
        }
        or value.get("readiness_counts")
        != {
            "rediscovery_winners_bound": 1,
            "induction_theorems_presented": 1,
            "kernel_executions_attempted": 1,
            "kernel_checked_theorems": 1,
            "blocked_theorems": 0,
            "rejected_theorems": 0,
        }
        or value.get("claim_seals")
        != {
            "rediscovery_winner_bound": True,
            "closed_form_induction_kernel_checked": True,
            "dependency_closure_validated": True,
            "withheld_theorem_used_as_premise": False,
            "novel_theorem_claimed": False,
            "general_rediscovery_claimed": False,
            "scientific_or_physics_truth_inferred": False,
        }
        or value.get("source_bindings")
        != {
            "source": {
                "path": SOURCE_PATH,
                "file_sha256": _file_sha(_inside(root, SOURCE_PATH)),
            },
            "config": {
                "path": CONFIG_PATH,
                "file_sha256": _file_sha(config_path),
            },
            "test": {
                "path": TEST_PATH,
                "file_sha256": _file_sha(_inside(root, TEST_PATH)),
            },
            "theorem": {"path": THEOREM_PATH, "file_sha256": theorem_sha},
            "rediscovery": REDISCOVERY_BINDING,
            "lean_runtime": LEAN_RUNTIME_BINDING,
            "math_lean_adapter": ADAPTER_BINDING,
        }
    ):
        raise ValueError("natural-sum Lean bridge evidence bindings changed")


def validate_live_receipt(
    value: Mapping[str, Any],
    config_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    root = config_path.resolve().parents[2]
    validate_checked_receipt(value, root=root)
    if value.get("receipt_role") != "live_replay":
        raise ValueError("natural-sum Lean bridge live receipt role changed")
    expected = build_live_receipt(config_path, environment=environment)
    if value != expected:
        raise ValueError("natural-sum Lean bridge live replay changed")


def write_receipt(
    config_path: Path,
    output_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
    receipt_role: str = "live_replay",
) -> Path:
    receipt = build_live_receipt(config_path, environment=environment, receipt_role=receipt_role)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument(
        "--mode", choices=("validate-checked", "emit-live"), default="validate-checked"
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.config.resolve().parents[2]
    if arguments.mode == "validate-checked":
        checked = json.loads(_inside(root, OUTPUT_PATH).read_text(encoding="utf-8"))
        validate_checked_receipt(checked, root=root)
        print(_inside(root, OUTPUT_PATH))
        return 0
    if arguments.output is None:
        parser.error("--output is required for emit-live")
    output = arguments.output.resolve()
    write_receipt(arguments.config, output)
    live = json.loads(output.read_text(encoding="utf-8"))
    validate_live_receipt(live, arguments.config)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
