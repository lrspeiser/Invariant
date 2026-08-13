"""Replay two recovered identities and kernel-check the recovered quartic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
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
from .math_lean_adapter import (
    AUDIT_PROTOCOL,
    LeanAdapterConfig,
    build_allowed_premise_manifest,
    run_lean_adapter,
)
from .math_lean_adapter import RESULT_SCHEMA as ADAPTER_SCHEMA

CONFIG_SCHEMA = "invariant-constraint-recovered-identity-breadth-lean-bridge-config-1.0"
RESULT_SCHEMA = "invariant-constraint-recovered-identity-breadth-lean-bridge-result-1.0"
CAMPAIGN_ID = "constraint-recovered-identity-breadth-lean-bridge-001"
CONFIG_PATH = "configs/math/constraint_recovered_identity_breadth_lean_bridge.json"
SOURCE_PATH = "src/sigma_theory_compiler/constraint_recovered_identity_breadth_lean_bridge.py"
TEST_PATH = "tests/test_constraint_recovered_identity_breadth_lean_bridge.py"
THEOREM_PATH = "formal/lean/ConstraintRecoveredQuarticIdentity.lean"
OUTPUT_PATH = "runs/math/constraint-recovered-identity-breadth-lean-bridge/receipt.json"
TARGET = "Invariant.constraintRecoveredQuarticIdentity"
FALSE_TARGET = "Invariant.constraintRecoveredQuarticFalseControl"
CONFIG_SHA256 = "f8d525d57d9a89df85481bd884f2976b05119aa1bad807ee60f69c7b0fa8a88a"
THEOREM_SHA256 = "75721bd58a7d23c55449d114b0f6f97381410f1b23bad6a02acb14ccba02aa6f"

ALLOWED_PREMISES = (
    "Invariant.recoveredPolyAdd",
    "Invariant.recoveredPolyScale",
    "Invariant.recoveredPolyMul",
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

WORLD_CONTRACTS = {
    "constraint.hidden_quartic": {
        "public_constraints_sha256": "0671b1163577eb24816ba0f028b3385a611c35eae051d11cda3083dd541ca0bb",
        "candidate_artifact_id": "sig-e6827fb88a008c33844e3aad",
        "candidate_content_sha256": "e6827fb88a008c33844e3aad80caf01271dc964d31f8a79a7fbdd0e20c09aa76",
        "candidate_family": "symbolic",
        "solver_result_sha256": "e157f0a5b91825a83bb500956ef355bc94bfa61896503956db1e9b555fd6f0c8",
        "recovered_expression": "x**4 + 2*x**3 - x - 30",
        "expression_sha256": "c0c1107959645a75dbc4a8c776875f0531a738ded4a846f34a5aee1f8a6ab67a",
        "certificate_content_sha256": "2cc733210e1a6315a5f6d8f22fa0b2c11474c20f6c8b0a5cd2327107a51a2d6d",
        "certificate_statement_sha256": "2599e6d4bb986e455d0bdf5e02d8967e8e5e3a1df4d4cbfc9b87b18bcc12acbd",
        "recovered_coefficients": [-30, -1, 0, 2, 1],
    },
    "constraint.hidden_partial_fraction": {
        "public_constraints_sha256": "fee3826f0d5c80abf841ef7ebdb880812ac3906bd8cf9f708518c2323de26df5",
        "candidate_artifact_id": "sig-3e4e8e0839cbca9d4a4f57ba",
        "candidate_content_sha256": "3e4e8e0839cbca9d4a4f57bab4896bd0817c25eebe2dda9a1e9a3f196dbf0061",
        "candidate_family": "symbolic",
        "solver_result_sha256": "b9475922ac39bb0dc1ea625690565da46fb5f441360650b82870af4d77422487",
        "recovered_expression": "(6*x**2 + 53*x + 127)/(x**3 + 14*x**2 + 59*x + 70)",
        "target_expression": "3/(x + 2) - 2/(x + 5) + 5/(x + 7)",
        "expression_sha256": "80e7323ec67201555598867ed4f819809efad05bf513d270ee213dd43b4a7de6",
        "certificate_content_sha256": "b78c538f429342a3eeb353a8084ba10438c5ecc46336191f095e07767da9edfe",
        "certificate_statement_sha256": "45477ad3c2319349e0ba143e40f5005d568d2b098eeaadba695657af7a7165e4",
        "partial_weights": [3, -2, 5],
        "recovered_numerator_coefficients": [127, 53, 6],
        "recovered_denominator_coefficients": [70, 59, 14, 1],
    },
}

CLAIMS = {
    "two_constraint_recovered_worlds_bound": True,
    "two_symbolic_certificates_bound": True,
    "quartic_integer_polynomial_identity_replayed": True,
    "partial_fraction_integer_polynomial_identity_replayed": True,
    "quartic_identity_kernel_checked": True,
    "false_quartic_kernel_rejected": True,
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
    "integer_polynomial_replays",
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
    """Hash canonical UTF-8/LF text so bindings survive Git checkout policy."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("identity breadth bridge path escapes root")
    return path


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("identity breadth bridge bound file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("identity breadth bridge bound content changed")
    return value


def _validate_files(root: Path) -> None:
    for bundle in (RECOVERY_BINDING, RUNTIME_BINDING):
        for role, binding in bundle.items():
            if role in {"artifact", "receipt"}:
                continue
            if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
                raise ValueError("identity breadth bridge dependency changed")
    if _file_sha(_inside(root, CONFIG_PATH)) != CONFIG_SHA256:
        raise ValueError("identity breadth bridge config changed")
    if _file_sha(_inside(root, THEOREM_PATH)) != THEOREM_SHA256:
        raise ValueError("identity breadth bridge theorem changed")


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
            "world_contracts",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("output_path") != OUTPUT_PATH
        or config.get("theorem_source_path") != THEOREM_PATH
        or config.get("target") != TARGET
        or config.get("timeout_seconds") != 30
        or config.get("world_contracts") != WORLD_CONTRACTS
        or config.get("recovery_binding") != RECOVERY_BINDING
        or config.get("lean_runtime_binding") != RUNTIME_BINDING
        or config.get("premise_policy") != expected_policy
    ):
        raise ValueError("identity breadth bridge config semantics changed")
    return config


def _poly_add(left: Sequence[int], right: Sequence[int]) -> list[int]:
    size = max(len(left), len(right))
    return [
        (left[index] if index < len(left) else 0) + (right[index] if index < len(right) else 0)
        for index in range(size)
    ]


def _poly_scale(value: int, coefficients: Sequence[int]) -> list[int]:
    return [value * coefficient for coefficient in coefficients]


def _poly_mul(left: Sequence[int], right: Sequence[int]) -> list[int]:
    if not left or not right:
        return []
    result = [0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return result


def _sealed_replay(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _sha(body)}


def _integer_replays() -> list[dict[str, Any]]:
    quartic_expected = list(WORLD_CONTRACTS["constraint.hidden_quartic"]["recovered_coefficients"])
    quartic_product = _poly_mul(_poly_mul([-2, 1], [3, 1]), [5, 1, 1])
    partial = WORLD_CONTRACTS["constraint.hidden_partial_fraction"]
    denominators = ([2, 1], [5, 1], [7, 1])
    weights = partial["partial_weights"]
    partial_numerator = _poly_add(
        _poly_add(
            _poly_scale(weights[0], _poly_mul(denominators[1], denominators[2])),
            _poly_scale(weights[1], _poly_mul(denominators[0], denominators[2])),
        ),
        _poly_scale(weights[2], _poly_mul(denominators[0], denominators[1])),
    )
    partial_denominator = _poly_mul(_poly_mul(denominators[0], denominators[1]), denominators[2])
    expected_numerator = list(partial["recovered_numerator_coefficients"])
    expected_denominator = list(partial["recovered_denominator_coefficients"])
    if quartic_product != quartic_expected:
        raise ValueError("quartic closed integer polynomial replay failed")
    if partial_numerator != expected_numerator or partial_denominator != expected_denominator:
        raise ValueError("partial-fraction closed integer polynomial replay failed")
    return [
        _sealed_replay(
            {
                "world_id": "constraint.hidden_quartic",
                "method": "closed_integer_coefficient_convolution",
                "factor_coefficients_constant_first": [[-2, 1], [3, 1], [5, 1, 1]],
                "computed_coefficients_constant_first": quartic_product,
                "recovered_coefficients_constant_first": quartic_expected,
                "exact_equality": True,
                "floating_point_operations": 0,
            }
        ),
        _sealed_replay(
            {
                "world_id": "constraint.hidden_partial_fraction",
                "method": "closed_integer_common_denominator_convolution",
                "linear_denominator_coefficients_constant_first": [
                    list(item) for item in denominators
                ],
                "partial_fraction_weights": list(weights),
                "computed_numerator_coefficients_constant_first": partial_numerator,
                "recovered_numerator_coefficients_constant_first": expected_numerator,
                "computed_denominator_coefficients_constant_first": partial_denominator,
                "recovered_denominator_coefficients_constant_first": expected_denominator,
                "exact_equality": True,
                "regular_domain_exclusions": [-7, -5, -2],
                "floating_point_operations": 0,
            }
        ),
    ]


def _evidence(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    recovery = _load_bound(root, RECOVERY_BINDING["artifact"])
    if (
        recovery.get("decision") != "pass_three_of_three_constraint_conditioned_semantic_worlds"
        or recovery.get("claims", {}).get("scientific_truth_established") is not False
        or recovery.get("claims", {}).get("promotion_authorized") is not False
    ):
        raise ValueError("identity breadth recovery campaign boundary changed")
    worlds = recovery.get("world_results")
    if not isinstance(worlds, list):
        raise TypeError("identity breadth recovery worlds missing")
    selected_worlds: list[dict[str, Any]] = []
    expected_families = {
        "bayesian",
        "cross_domain",
        "egraph",
        "evolutionary",
        "grammar",
        "llm",
        "symbolic",
    }
    for world_id, contract in WORLD_CONTRACTS.items():
        matches = [item for item in worlds if item.get("world_id") == world_id]
        if len(matches) != 1:
            raise ValueError("identity breadth recovery world changed")
        world = matches[0]
        candidates = world.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 7:
            raise ValueError("identity breadth recovery candidates changed")
        selected = [
            item
            for item in candidates
            if item.get("artifact_id") == contract["candidate_artifact_id"]
        ]
        if len(selected) != 1:
            raise ValueError("identity breadth selected candidate changed")
        candidate = selected[0]
        representation = candidate.get("representation", {})
        certificate = world.get("reference_proof_certificate", {})
        target = world.get("unsealed_target", {})
        if (
            world.get("public_constraints_sha256") != contract["public_constraints_sha256"]
            or world.get("terminal_status_counts") != {"pass": 7}
            or {item.get("representation", {}).get("family") for item in candidates}
            != expected_families
            or any(
                item.get("representation", {}).get("target_fields_read") != []
                for item in candidates
            )
            or candidate.get("content_sha256") != contract["candidate_content_sha256"]
            or representation.get("family") != contract["candidate_family"]
            or representation.get("expression") != contract["recovered_expression"]
            or representation.get("expression_sha256") != contract["expression_sha256"]
            or representation.get("solver_receipt", {}).get("result_sha256")
            != contract["solver_result_sha256"]
            or certificate.get("content_sha256") != contract["certificate_content_sha256"]
            or certificate.get("statement_sha256") != contract["certificate_statement_sha256"]
            or certificate.get("decision") != "proved_exact_rational_identity_on_regular_domain"
            or certificate.get("witness", {}).get("cleared_numerator_zero") is not True
        ):
            raise ValueError("identity breadth recovery candidate or certificate changed")
        if world_id.endswith("quartic"):
            recovered = [item["numerator"] for item in representation.get("coefficients", [])]
            target_expression = target.get("closed_form")
            if (
                recovered != contract["recovered_coefficients"]
                or target_expression != contract["recovered_expression"]
            ):
                raise ValueError("quartic recovery semantics changed")
        else:
            recovered = [item["numerator"] for item in representation.get("coefficients", [])]
            if (
                recovered != contract["partial_weights"]
                or target.get("closed_form") != contract["target_expression"]
            ):
                raise ValueError("partial-fraction recovery semantics changed")
        selected_worlds.append(world)
    runtime = _load_bound(root, RUNTIME_BINDING["receipt"])
    runtime_toolchain = runtime.get("toolchain_receipt", {})
    if (
        runtime.get("schema_version")
        != "invariant-lean-production-kernel-vertical-slice-result-2.0"
        or runtime.get("campaign_id") != "lean-production-kernel-vertical-slice-001"
        or runtime.get("decision") != "pass_real_lean_kernel_vertical_slice"
        or runtime.get("adapter_receipt", {}).get("decision") != "pass_lean_checked_closed_premise"
        or runtime_toolchain.get("version") != VERSION
        or runtime_toolchain.get("commit") != COMMIT
        or runtime_toolchain.get("official_release") != RELEASE
        or runtime_toolchain.get("executable_path_persisted") is not False
    ):
        raise ValueError("identity breadth portable Lean runtime boundary changed")
    return recovery, selected_worlds, runtime


def _adapter_config(executable: Path | None = None, *, target: str = TARGET) -> LeanAdapterConfig:
    return LeanAdapterConfig(
        target=target,
        allowed_premises=ALLOWED_PREMISES if target == TARGET else (),
        forbidden_premises=FORBIDDEN_PREMISES,
        forbidden_prefixes=FORBIDDEN_PREFIXES,
        executable=executable,
        timeout_seconds=30.0,
    )


def _recovery_world_evidence(worlds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for world in worlds:
        world_id = str(world["world_id"])
        contract = WORLD_CONTRACTS[world_id]
        evidence.append(
            {
                "world_id": world_id,
                **contract,
                "certificate_kind": world["reference_proof_certificate"]["certificate_kind"],
                "certificate_decision": world["reference_proof_certificate"]["decision"],
                "candidate_count": len(world["candidates"]),
                "candidate_families": sorted(
                    item["representation"]["family"] for item in world["candidates"]
                ),
            }
        )
    return evidence


def _canonical_theorem_source(root: Path) -> str:
    return (
        _inside(root, THEOREM_PATH)
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def _run_theorem(executable: Path, root: Path) -> dict[str, Any]:
    source = _canonical_theorem_source(root)
    with tempfile.TemporaryDirectory(prefix="invariant-lean-quartic-") as temp:
        path = Path(temp) / "ConstraintRecoveredQuarticIdentity.lean"
        path.write_text(source, encoding="utf-8", newline="\n")
        return run_lean_adapter(_adapter_config(executable), path, environment={})


def _false_control_source() -> str:
    return """import Std.Tactic

namespace Invariant

theorem constraintRecoveredQuarticFalseControl :
    ([-30, -1, 0, 2, 1] : List Int) = [-29, -1, 0, 2, 1] := by
  rfl

end Invariant
"""


def _run_false_control(executable: Path) -> dict[str, Any]:
    source = _false_control_source()
    with tempfile.TemporaryDirectory(prefix="invariant-lean-quartic-false-") as temp:
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
        raise ValueError("false quartic control was not rejected by Lean")
    return {
        "target": FALSE_TARGET,
        "alteration": "constant coefficient -30 changed to -29",
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
        raise ValueError("identity breadth bridge receipt role is invalid")
    root = config_path.resolve().parents[2]
    _load_config(root)
    recovery, worlds, _runtime = _evidence(root)
    runtime_config = json.loads(
        _inside(root, RUNTIME_BINDING["config"]["path"]).read_text(encoding="utf-8")
    )
    executable, resolution = _resolve_executable(runtime_config, environment=environment)
    if executable is None or resolution is None:
        raise ValueError("registered Lean executable unavailable")
    toolchain = _probe_toolchain(executable, _platform_key(), resolution)
    adapter_config = _adapter_config(executable)
    adapter = _run_theorem(executable, root)
    if adapter.get("decision") != "pass_lean_checked_closed_premise":
        raise ValueError("quartic Lean adapter did not pass")
    false_control = _run_false_control(executable)
    replays = _integer_replays()
    recovery_worlds = _recovery_world_evidence(worlds)
    theorem = _inside(root, THEOREM_PATH)
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "receipt_role": receipt_role,
        "decision": "pass_two_recovered_identities_replayed_and_quartic_checked_by_real_lean_kernel",
        "first_blocker": "none_for_two_bound_recovered_identities",
        "recovery_evidence": {
            "campaign_id": recovery["campaign_id"],
            "artifact_content_sha256": recovery["content_sha256"],
            "worlds": recovery_worlds,
            "target_fields_read_before_unseal": 0,
        },
        "integer_polynomial_replays": replays,
        "theorem_contract": {
            "target": TARGET,
            "statement": "coefficient convolution of (x-2)(x+3)(x^2+x+5) equals [-30,-1,0,2,1]",
            "domain": "closed finite lists of integer coefficients in constant-first order",
            "proof_method": "kernel reduction of executable List Int coefficient convolution by rfl",
            "source_sha256": _file_sha(theorem),
            "source_bytes_canonical_lf": len(_canonical_theorem_source(root).encode()),
            "sorry_or_axiom_used": False,
        },
        "dependency_closure": {
            "allowed_premise_manifest": build_allowed_premise_manifest(adapter_config),
            "declared_dependencies": sorted(ALLOWED_PREMISES),
            "closure_checked_by_adapter": True,
            "recovery_targets_used_as_Lean_premises": False,
            "out_of_manifest_dependency_allowed": False,
        },
        "adapter_receipt": adapter,
        "false_control": false_control,
        "toolchain_receipt": toolchain,
        "counts": {
            "recovered_worlds_bound": 2,
            "recovered_candidates_bound": 14,
            "symbolic_certificates_bound": 2,
            "integer_polynomial_replays": 2,
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
            "two exact constraint-recovered synthetic identities only; one quartic is independently "
            "kernel-checked and both are replayed by closed integer coefficient arithmetic; no general "
            "recovery, novelty, promotion, scientific, or physics claim"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def validate_checked_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    if set(value) != _TOP_KEYS or value.get("content_sha256") != _content_sha(value):
        raise ValueError("identity breadth bridge receipt schema or seal changed")
    if re.search(r"[A-Za-z]:\\|/(?:home|Users)/", json.dumps(value, sort_keys=True)):
        raise ValueError("identity breadth bridge persisted host path")
    _load_config(root)
    recovery, source_worlds, _runtime = _evidence(root)
    toolchain = value.get("toolchain_receipt", {})
    platform = str(toolchain.get("platform"))
    asset = PLATFORM_ASSETS.get(platform)
    adapter = value.get("adapter_receipt", {})
    expected_manifest = build_allowed_premise_manifest(_adapter_config())
    expected_bindings = {
        "config": {"path": CONFIG_PATH, "file_sha256": CONFIG_SHA256},
        "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
        "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
        "theorem": {"path": THEOREM_PATH, "file_sha256": THEOREM_SHA256},
        "recovery": RECOVERY_BINDING,
        "lean_runtime": RUNTIME_BINDING,
        "math_lean_adapter": ADAPTER_BINDING,
    }
    expected_counts = {
        "recovered_worlds_bound": 2,
        "recovered_candidates_bound": 14,
        "symbolic_certificates_bound": 2,
        "integer_polynomial_replays": 2,
        "kernel_executions": 2,
        "kernel_checked_theorems": 1,
        "false_controls_rejected": 1,
        "blocked": 0,
        "rejected": 0,
    }
    false_control = value.get("false_control", {})
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("receipt_role") not in {"checked_windows_historical", "live_replay"}
        or value.get("decision")
        != "pass_two_recovered_identities_replayed_and_quartic_checked_by_real_lean_kernel"
        or value.get("first_blocker") != "none_for_two_bound_recovered_identities"
        or value.get("claims") != CLAIMS
        or value.get("counts") != expected_counts
        or value.get("recovery_evidence", {}).get("artifact_content_sha256")
        != recovery["content_sha256"]
        or value.get("recovery_evidence", {}).get("target_fields_read_before_unseal") != 0
        or value.get("recovery_evidence", {}).get("worlds")
        != _recovery_world_evidence(source_worlds)
        or value.get("integer_polynomial_replays") != _integer_replays()
        or value.get("theorem_contract", {}).get("source_sha256") != THEOREM_SHA256
        or value.get("theorem_contract", {}).get("sorry_or_axiom_used") is not False
        or value.get("dependency_closure", {}).get("allowed_premise_manifest") != expected_manifest
        or value.get("dependency_closure", {}).get("declared_dependencies")
        != sorted(ALLOWED_PREMISES)
        or value.get("dependency_closure", {}).get("recovery_targets_used_as_Lean_premises")
        is not False
        or false_control
        != {
            "target": FALSE_TARGET,
            "alteration": "constant coefficient -30 changed to -29",
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
        raise ValueError("identity breadth bridge semantics changed")


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
        raise ValueError("identity breadth bridge live replay changed")


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
