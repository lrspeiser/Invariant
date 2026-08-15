"""Check one constraint-recovered recurrence with the portable real Lean kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    validate_checked_receipt as validate_runtime_receipt,
)
from .math_lean_adapter import (
    AUDIT_PROTOCOL,
    LeanAdapterConfig,
    build_allowed_premise_manifest,
    run_lean_adapter,
)
from .math_lean_adapter import RESULT_SCHEMA as ADAPTER_SCHEMA

CONFIG_SCHEMA = "invariant-constraint-recovered-recurrence-lean-kernel-bridge-config-1.0"
RESULT_SCHEMA = "invariant-constraint-recovered-recurrence-lean-kernel-bridge-result-1.0"
CAMPAIGN_ID = "constraint-recovered-recurrence-lean-kernel-bridge-001"
CONFIG_PATH = "configs/math/constraint_recovered_recurrence_lean_kernel_bridge.json"
SOURCE_PATH = "src/sigma_theory_compiler/constraint_recovered_recurrence_lean_kernel_bridge.py"
TEST_PATH = "tests/test_constraint_recovered_recurrence_lean_kernel_bridge.py"
THEOREM_PATH = "formal/lean/ConstraintRecoveredRecurrence.lean"
OUTPUT_PATH = "runs/math/constraint-recovered-recurrence-lean-kernel-bridge/receipt.json"
TARGET = "Invariant.constraintRecoveredSequenceClosedForm"
FALSE_TARGET = "Invariant.constraintRecoveredSequenceFalseControl"
CONFIG_SHA256 = "5f8c08f2ced4a00c8f31a3579b00505d13d86974bfd7ed6ed182199742ff61bd"
THEOREM_SHA256 = "09b1ab65ebe9de65a564124753b0aedd4f35e27e773f7a13467833a7b6cbd642"

ALLOWED_PREMISES = (
    "Invariant.constraintRecoveredSequence",
    "Invariant.recoveredPolynomialSuccessor",
    "Nat.rec",
    "Nat.add_mul",
    "Nat.mul_add",
    "Nat.mul_assoc",
    "Nat.mul_one",
    "Nat.pow_succ",
    "Nat.pow_zero",
    "Lean.Parser.Tactic.omega",
)
FORBIDDEN_PREMISES = ("Classical.choice", "False.elim")
FORBIDDEN_PREFIXES = ("KnownAnswer", "Unsafe")

RECOVERY_BINDING = {
    "source": {
        "path": "src/sigma_theory_compiler/constraint_conditioned_semantic_recovery_tournament.py",
        "file_sha256": "b2fadd569871498e6351e4b087552850a2b3cb53a299e9456cb869b8184bc4d5",
    },
    "config": {
        "path": "configs/constraint_conditioned_semantic_recovery_tournament.json",
        "file_sha256": "64434d7ec2a78272cf2842219f89910ea77e9f1527846de8153d0a21b3063fb7",
    },
    "test": {
        "path": "tests/test_constraint_conditioned_semantic_recovery_tournament.py",
        "file_sha256": "f5c0f63955850607a207216932064ce2545a756f0d4d984977d649920d30b335",
    },
    "artifact": {
        "path": "runs/math/constraint-conditioned-semantic-recovery-tournament/campaign.json",
        "file_sha256": "cb5b0963c52887c93008a23d87925dea8b9fd394063ff5d724b56c0a587ef54e",
        "content_sha256": "300eee2de03b328da94cad8337aba11f3e31dde17ffb97e016d6cc922a43f37d",
    },
}
RUNTIME_BINDING = {
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
WINNER_CONTRACT = {
    "world_id": "constraint.hidden_recurrence",
    "public_constraints_sha256": "7f0fe4de903c11d14a01e2dfe0876ce9fcbe5a67d746438d4f8c0d5b4e83eb28",
    "candidate_artifact_id": "sig-01a4b312743385723df01633",
    "candidate_content_sha256": "01a4b312743385723df0163335d464011658a986782846ecb020a6fc3cb1e64e",
    "solver_result_sha256": "45a13350b70230aac8ed08fd4609d4ae330b1d0ffcc8b418acbde171d9b24e4a",
    "closed_form": "2*n**3 + 2*n**2 + n + 7",
    "increment": "6*n**2 + 10*n + 5",
    "certificate_content_sha256": "3a7c3610f0d73013241423bea5df17aea91bd9483ba8bb7f1cfe2e544564edce",
    "recurrence_sha256": "8de70d2c1e113843d00879a21dbc369e1db384ebffa6d99629c4cf001a87ef13",
    "statement_sha256": "c9cf6b4d4a6b3285816c1f3b46162c681d3e29461eccd70fe48558d2b73171af",
    "successor_rule_sha256": "d157dba6b572c9f0dc7da38c805ebd3078c4b71e53ba14df756ca2a46f3b9bdd",
}
CLAIMS = {
    "constraint_recovered_candidate_bound": True,
    "symbolic_certificate_bound": True,
    "recurrence_closed_form_kernel_checked": True,
    "false_closed_form_kernel_rejected": True,
    "dependency_closure_validated": True,
    "general_recovery_established": False,
    "novelty_established": False,
    "promotion_authorized": False,
    "scientific_or_physics_truth_inferred": False,
}

_AUDIT_OUTPUT = (
    f"{AUDIT_PROTOCOL}_BEGIN\n"
    f"target={TARGET}\n"
    + "".join(f"dependency={item}\n" for item in ALLOWED_PREMISES)
    + "result=checked\n"
    + f"{AUDIT_PROTOCOL}_END\n"
).encode()
_TOP_KEYS = {
    "adapter_receipt",
    "campaign_id",
    "claims",
    "content_sha256",
    "counts",
    "decision",
    "dependency_closure",
    "false_control",
    "first_blocker",
    "receipt_role",
    "recovery_evidence",
    "schema_version",
    "scope",
    "source_bindings",
    "theorem_contract",
    "toolchain_receipt",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("recurrence bridge path escapes root")
    return path


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("recurrence bridge bound file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("recurrence bridge bound content changed")
    return value


def _validate_files(root: Path) -> None:
    for bundle in (RECOVERY_BINDING, RUNTIME_BINDING):
        for role, binding in bundle.items():
            if role in {"artifact", "receipt"}:
                continue
            if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
                raise ValueError("recurrence bridge dependency changed")
    if _file_sha(_inside(root, CONFIG_PATH)) != CONFIG_SHA256:
        raise ValueError("recurrence bridge config changed")
    if _file_sha(_inside(root, THEOREM_PATH)) != THEOREM_SHA256:
        raise ValueError("recurrence bridge theorem changed")


def _load_config(root: Path) -> dict[str, Any]:
    _validate_files(root)
    config = json.loads(_inside(root, CONFIG_PATH).read_text(encoding="utf-8"))
    expected_policy = {
        "allowed_premises": list(ALLOWED_PREMISES),
        "equivalent_targets": [],
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "forbidden_premises": list(FORBIDDEN_PREMISES),
    }
    if (
        set(config)
        != {
            "campaign_id",
            "lean_runtime_binding",
            "output_path",
            "premise_policy",
            "recovery_binding",
            "schema_version",
            "target",
            "theorem_source_path",
            "timeout_seconds",
            "winner_contract",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("output_path") != OUTPUT_PATH
        or config.get("theorem_source_path") != THEOREM_PATH
        or config.get("target") != TARGET
        or config.get("timeout_seconds") != 30
        or config.get("winner_contract") != WINNER_CONTRACT
        or config.get("recovery_binding") != RECOVERY_BINDING
        or config.get("lean_runtime_binding") != RUNTIME_BINDING
        or config.get("premise_policy") != expected_policy
    ):
        raise ValueError("recurrence bridge config semantics changed")
    return config


def _evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    recovery = _load_bound(root, RECOVERY_BINDING["artifact"])
    if (
        recovery.get("decision") != "pass_three_of_three_constraint_conditioned_semantic_worlds"
        or recovery.get("claims", {}).get("scientific_truth_established") is not False
        or recovery.get("claims", {}).get("promotion_authorized") is not False
    ):
        raise ValueError("recurrence recovery campaign boundary changed")
    worlds = recovery.get("world_results")
    if not isinstance(worlds, list):
        raise TypeError("recurrence recovery worlds missing")
    matches = [item for item in worlds if item.get("world_id") == WINNER_CONTRACT["world_id"]]
    if len(matches) != 1:
        raise ValueError("recurrence recovery world changed")
    world = matches[0]
    candidates = world.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError("recurrence recovery candidates missing")
    selected = [
        item
        for item in candidates
        if item.get("artifact_id") == WINNER_CONTRACT["candidate_artifact_id"]
    ]
    if len(selected) != 1:
        raise ValueError("recurrence recovery selected candidate changed")
    candidate = selected[0]
    representation = candidate.get("representation", {})
    certificate = world.get("reference_proof_certificate", {})
    target = world.get("unsealed_target", {})
    if (
        world.get("public_constraints_sha256") != WINNER_CONTRACT["public_constraints_sha256"]
        or candidate.get("content_sha256") != WINNER_CONTRACT["candidate_content_sha256"]
        or representation.get("expression") != WINNER_CONTRACT["closed_form"]
        or representation.get("solver_receipt", {}).get("result_sha256")
        != WINNER_CONTRACT["solver_result_sha256"]
        or target.get("closed_form") != WINNER_CONTRACT["closed_form"]
        or target.get("increment") != WINNER_CONTRACT["increment"]
        or target.get("initial_value") != 7
        or certificate.get("content_sha256") != WINNER_CONTRACT["certificate_content_sha256"]
        or certificate.get("decision") != "proved_by_base_and_symbolic_successor_identity"
        or certificate.get("recurrence_sha256") != WINNER_CONTRACT["recurrence_sha256"]
        or certificate.get("statement_sha256") != WINNER_CONTRACT["statement_sha256"]
        or certificate.get("obligations", {}).get("successor", {}).get("recurrence_rule_sha256")
        != WINNER_CONTRACT["successor_rule_sha256"]
    ):
        raise ValueError("recurrence recovery candidate or certificate changed")
    runtime = _load_bound(root, RUNTIME_BINDING["receipt"])
    validate_runtime_receipt(runtime, root=root)
    return recovery, world, runtime


def _adapter_config(executable: Path, *, target: str = TARGET) -> LeanAdapterConfig:
    return LeanAdapterConfig(
        target=target,
        allowed_premises=ALLOWED_PREMISES if target == TARGET else (),
        forbidden_premises=FORBIDDEN_PREMISES,
        forbidden_prefixes=FORBIDDEN_PREFIXES,
        executable=executable,
        timeout_seconds=30.0,
    )


def _false_control_source() -> str:
    return """import Std.Tactic

namespace Invariant

def constraintRecoveredSequenceFalse : Nat → Nat
  | 0 => 7
  | n + 1 => constraintRecoveredSequenceFalse n + (6 * n ^ 2 + 10 * n + 5)

theorem constraintRecoveredSequenceFalseControl :
    constraintRecoveredSequenceFalse 0 = 8 := by
  rfl

end Invariant
"""


def _run_false_control(executable: Path) -> dict[str, Any]:
    source = _false_control_source()
    with tempfile.TemporaryDirectory(prefix="invariant-lean-false-control-") as temp:
        path = Path(temp) / "FalseControl.lean"
        path.write_text(source, encoding="utf-8", newline="\n")
        result = run_lean_adapter(
            _adapter_config(executable, target=FALSE_TARGET), path, environment={}
        )
    execution = result["execution"]
    if (
        result.get("decision") != "block_lean_process_failure"
        or result.get("status") != "block"
        or execution.get("attempted") is not True
        or execution.get("timed_out") is not False
        or not isinstance(execution.get("exit_code"), int)
        or execution["exit_code"] == 0
    ):
        raise ValueError("false recurrence control was not rejected by Lean")
    return {
        "target": FALSE_TARGET,
        "alteration": "base value 7 falsely claimed equal to 8",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "kernel_attempted": True,
        "adapter_status": result["status"],
        "adapter_decision": result["decision"],
        "nonzero_exit_code": True,
        "timed_out": False,
        "rejected_before_receipt_promotion": True,
    }


def build_live_receipt(
    config_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
    receipt_role: str = "live_replay",
) -> dict[str, Any]:
    if receipt_role not in {"live_replay", "checked_windows_historical"}:
        raise ValueError("recurrence bridge receipt role is invalid")
    root = config_path.resolve().parents[2]
    _load_config(root)
    recovery, world, _runtime = _evidence(root)
    runtime_config = json.loads(
        _inside(root, RUNTIME_BINDING["config"]["path"]).read_text(encoding="utf-8")
    )
    executable, resolution = _resolve_executable(runtime_config, environment=environment)
    if executable is None or resolution is None:
        raise ValueError("registered Lean executable unavailable")
    toolchain = _probe_toolchain(executable, _platform_key(), resolution)
    theorem = _inside(root, THEOREM_PATH)
    adapter_config = _adapter_config(executable)
    adapter = run_lean_adapter(adapter_config, theorem, environment={})
    if adapter.get("decision") != "pass_lean_checked_closed_premise":
        raise ValueError("recurrence Lean adapter did not pass")
    false_control = _run_false_control(executable)
    certificate = world["reference_proof_certificate"]
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "receipt_role": receipt_role,
        "decision": "pass_constraint_recovered_recurrence_checked_by_real_lean_kernel",
        "first_blocker": "none_for_single_bound_recurrence_closed_form",
        "recovery_evidence": {
            "campaign_id": recovery["campaign_id"],
            "artifact_content_sha256": recovery["content_sha256"],
            **WINNER_CONTRACT,
            "candidate_family": "llm",
            "certificate_decision": certificate["decision"],
            "certificate_kind": certificate["certificate_kind"],
            "base_index": certificate["base_index"],
        },
        "theorem_contract": {
            "target": TARGET,
            "initial_value": 7,
            "increment": WINNER_CONTRACT["increment"],
            "closed_form": WINNER_CONTRACT["closed_form"],
            "domain": "all natural-number indices reachable by successor from zero",
            "proof_method": "Nat induction plus explicit polynomial successor identity",
            "source_sha256": _file_sha(theorem),
            "source_bytes": theorem.stat().st_size,
            "sorry_or_axiom_used": False,
        },
        "dependency_closure": {
            "allowed_premise_manifest": build_allowed_premise_manifest(adapter_config),
            "declared_dependencies": sorted(ALLOWED_PREMISES),
            "closure_checked_by_adapter": True,
            "recovery_target_used_as_Lean_premise": False,
            "out_of_manifest_dependency_allowed": False,
        },
        "adapter_receipt": adapter,
        "false_control": false_control,
        "toolchain_receipt": toolchain,
        "counts": {
            "recovered_candidates_bound": 1,
            "symbolic_certificates_bound": 1,
            "kernel_executions": 2,
            "kernel_checked_theorems": 1,
            "false_controls_rejected": 1,
            "blocked": 0,
            "rejected": 0,
        },
        "claims": CLAIMS,
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": CONFIG_SHA256},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "theorem": {"path": THEOREM_PATH, "file_sha256": THEOREM_SHA256},
            "recovery": RECOVERY_BINDING,
            "lean_runtime": RUNTIME_BINDING,
            "math_lean_adapter": ADAPTER_BINDING,
        },
        "scope": (
            "one exact recovered first-order natural recurrence and its cubic closed form only; "
            "no general recovery, novelty, promotion, scientific, or physics claim"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def validate_checked_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    if set(value) != _TOP_KEYS or value.get("content_sha256") != _content_sha(value):
        raise ValueError("recurrence bridge receipt schema or seal changed")
    if re.search(r"[A-Za-z]:\\|/(?:home|Users)/", json.dumps(value, sort_keys=True)):
        raise ValueError("recurrence bridge persisted host path")
    _load_config(root)
    recovery, _world, _runtime = _evidence(root)
    toolchain = value.get("toolchain_receipt", {})
    platform = str(toolchain.get("platform"))
    asset = PLATFORM_ASSETS.get(platform)
    adapter = value.get("adapter_receipt", {})
    expected_manifest = build_allowed_premise_manifest(
        LeanAdapterConfig(
            target=TARGET,
            allowed_premises=ALLOWED_PREMISES,
            forbidden_premises=FORBIDDEN_PREMISES,
            forbidden_prefixes=FORBIDDEN_PREFIXES,
        )
    )
    expected_bindings = {
        "config": {"path": CONFIG_PATH, "file_sha256": CONFIG_SHA256},
        "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
        "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
        "theorem": {"path": THEOREM_PATH, "file_sha256": THEOREM_SHA256},
        "recovery": RECOVERY_BINDING,
        "lean_runtime": RUNTIME_BINDING,
        "math_lean_adapter": ADAPTER_BINDING,
    }
    false_control = value.get("false_control", {})
    expected_counts = {
        "recovered_candidates_bound": 1,
        "symbolic_certificates_bound": 1,
        "kernel_executions": 2,
        "kernel_checked_theorems": 1,
        "false_controls_rejected": 1,
        "blocked": 0,
        "rejected": 0,
    }
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("receipt_role") not in {"checked_windows_historical", "live_replay"}
        or value.get("decision")
        != "pass_constraint_recovered_recurrence_checked_by_real_lean_kernel"
        or value.get("claims") != CLAIMS
        or value.get("counts") != expected_counts
        or value.get("recovery_evidence", {}).get("artifact_content_sha256")
        != recovery["content_sha256"]
        or any(
            value.get("recovery_evidence", {}).get(key) != expected
            for key, expected in WINNER_CONTRACT.items()
        )
        or value.get("theorem_contract", {}).get("source_sha256") != THEOREM_SHA256
        or value.get("theorem_contract", {}).get("sorry_or_axiom_used") is not False
        or value.get("dependency_closure", {}).get("allowed_premise_manifest") != expected_manifest
        or value.get("dependency_closure", {}).get("declared_dependencies")
        != sorted(ALLOWED_PREMISES)
        or value.get("dependency_closure", {}).get("recovery_target_used_as_Lean_premise")
        is not False
        or false_control
        != {
            "target": FALSE_TARGET,
            "alteration": "base value 7 falsely claimed equal to 8",
            "source_sha256": hashlib.sha256(_false_control_source().encode()).hexdigest(),
            "kernel_attempted": True,
            "adapter_status": "block",
            "adapter_decision": "block_lean_process_failure",
            "nonzero_exit_code": True,
            "timed_out": False,
            "rejected_before_receipt_promotion": True,
        }
        or asset is None
        or toolchain.get("version") != VERSION
        or toolchain.get("commit") != COMMIT
        or toolchain.get("official_release") != RELEASE
        or toolchain.get("platform_asset") != asset
        or toolchain.get("executable_path_persisted") is not False
        or adapter.get("schema_version") != ADAPTER_SCHEMA
        or adapter.get("decision") != "pass_lean_checked_closed_premise"
        or adapter.get("target") != TARGET
        or adapter.get("source_sha256") != THEOREM_SHA256
        or adapter.get("manifest_sha256") != expected_manifest["content_sha256"]
        or adapter.get("execution")
        != {
            "attempted": True,
            "timed_out": False,
            "exit_code": 0,
            "stdout_sha256": hashlib.sha256(_AUDIT_OUTPUT).hexdigest(),
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        }
        or adapter.get("dependency_audit")
        != {
            "protocol_version": AUDIT_PROTOCOL,
            "reported_target": TARGET,
            "dependencies": sorted(ALLOWED_PREMISES),
            "closure_valid": True,
        }
        or value.get("source_bindings") != expected_bindings
    ):
        raise ValueError("recurrence bridge semantics changed")


def validate_live_receipt(
    value: Mapping[str, Any],
    config_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    root = config_path.resolve().parents[2]
    validate_checked_receipt(value, root=root)
    if value.get("receipt_role") != "live_replay" or value != build_live_receipt(
        config_path, environment=environment
    ):
        raise ValueError("recurrence bridge live replay changed")


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
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument(
        "--mode", choices=("validate-checked", "emit-live"), default="validate-checked"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.config.resolve().parents[2]
    if args.mode == "validate-checked":
        receipt = json.loads(_inside(root, OUTPUT_PATH).read_text(encoding="utf-8"))
        validate_checked_receipt(receipt, root=root)
        return 0
    if args.output is None:
        parser.error("--output is required for emit-live")
    write_receipt(args.config, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
