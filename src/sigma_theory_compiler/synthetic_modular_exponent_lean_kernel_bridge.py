"""Bind the blinded modular-exponent winner to an independent Lean kernel proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
from .math_lean_adapter import (
    RESULT_SCHEMA as ADAPTER_SCHEMA,
)
from .synthetic_modular_exponent_holdout_world import validate_campaign as validate_rediscovery

CONFIG_SCHEMA = "invariant-synthetic-modular-exponent-lean-kernel-bridge-config-1.0"
RESULT_SCHEMA = "invariant-synthetic-modular-exponent-lean-kernel-bridge-result-1.0"
CAMPAIGN_ID = "synthetic-modular-exponent-lean-kernel-bridge-001"
CONFIG_PATH = "configs/math/synthetic_modular_exponent_lean_kernel_bridge.json"
SOURCE_PATH = "src/sigma_theory_compiler/synthetic_modular_exponent_lean_kernel_bridge.py"
TEST_PATH = "tests/test_synthetic_modular_exponent_lean_kernel_bridge.py"
THEOREM_PATH = "formal/lean/AnonymousModularExponent.lean"
OUTPUT_PATH = "runs/math/synthetic-modular-exponent-lean-kernel-bridge/receipt.json"
TARGET = "Invariant.anonymousModularExponent"
ALLOWED_PREMISES = ("of_decide_eq_true",)
FORBIDDEN_PREMISES = ("Classical.choice", "False.elim")
FORBIDDEN_PREFIXES = ("KnownAnswer", "Unsafe")
CONFIG_SHA256 = "8ba66f92e89ca283b77739b6126d17a856a07ca85671182f5279148b450fab8a"
THEOREM_SHA256 = "2b2bb324c6b3bf3095d1db6bc644a99b650f43d31bb6195eee7be992c1b5e0fd"
REDISCOVERY_BINDING = {
    "source": {
        "path": "src/sigma_theory_compiler/synthetic_modular_exponent_holdout_world.py",
        "file_sha256": "08ed6541868b3cc28e08e435482c6ab7d6dcaf3a9e914fb3618dd59f76477e46",
    },
    "config": {
        "path": "configs/synthetic_modular_exponent_holdout_world.json",
        "file_sha256": "4091462af36ff4de5936bc49effd5eb8f42fc45a6ed4e0fe534751f7b98a78d4",
    },
    "test": {
        "path": "tests/test_synthetic_modular_exponent_holdout_world.py",
        "file_sha256": "c49397109f4c1d17946711d7e7c235b2fd2ed8a4624854cfa65479618b9e9567",
    },
    "artifact": {
        "path": "runs/math/synthetic-modular-exponent-holdout-world/campaign.json",
        "file_sha256": "bbf3c93295400c313997cbeacc865371dac720c5cf142c67eb4311c8d5aeffc3",
        "content_sha256": "fdb89f7499e44ae6979a94a31fc84a5e1bf4790e4ed5605ae1130f611d9479f5",
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
    "modulus": 11,
    "exponent": 10,
    "proof_obligations": 10,
    "kernel_statement": "forall a:Fin 11, a!=0 -> a^10=1",
    "winner_sealed_before_reference_unseal": True,
}
CLAIMS = {
    "rediscovery_winner_bound": True,
    "ten_residue_obligations_kernel_checked": True,
    "dependency_closure_validated": True,
    "withheld_theorem_used_as_premise": False,
    "novel_theorem_claimed": False,
    "general_number_theory_claimed": False,
    "scientific_or_physics_truth_inferred": False,
}
_AUDIT_OUTPUT = (
    f"{AUDIT_PROTOCOL}_BEGIN\n"
    f"target={TARGET}\n"
    "dependency=of_decide_eq_true\n"
    "result=checked\n"
    f"{AUDIT_PROTOCOL}_END\n"
).encode()


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
        raise ValueError("modular-exponent bridge path escapes root")
    return path


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("modular-exponent bridge bound file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("modular-exponent bridge bound content changed")
    return value


def _validate_files(root: Path) -> None:
    for bundle in (REDISCOVERY_BINDING, RUNTIME_BINDING):
        for role, binding in bundle.items():
            if role in {"artifact", "receipt"}:
                continue
            if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
                raise ValueError("modular-exponent bridge dependency changed")
    if _file_sha(_inside(root, CONFIG_PATH)) != CONFIG_SHA256:
        raise ValueError("modular-exponent bridge config changed")
    if _file_sha(_inside(root, THEOREM_PATH)) != THEOREM_SHA256:
        raise ValueError("modular-exponent bridge theorem changed")


def _load_config(root: Path) -> dict[str, Any]:
    _validate_files(root)
    config = json.loads(_inside(root, CONFIG_PATH).read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("output_path") != OUTPUT_PATH
        or config.get("theorem_source_path") != THEOREM_PATH
        or config.get("target") != TARGET
        or config.get("timeout_seconds") != 30
        or config.get("winner_contract") != WINNER_CONTRACT
        or config.get("rediscovery_binding") != REDISCOVERY_BINDING
        or config.get("lean_runtime_binding") != RUNTIME_BINDING
        or config.get("premise_policy")
        != {
            "allowed_premises": list(ALLOWED_PREMISES),
            "equivalent_targets": [],
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "forbidden_premises": list(FORBIDDEN_PREMISES),
        }
    ):
        raise ValueError("modular-exponent bridge config semantics changed")
    return config


def _evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rediscovery = _load_bound(root, REDISCOVERY_BINDING["artifact"])
    validate_rediscovery(
        rediscovery,
        root,
        _inside(root, REDISCOVERY_BINDING["config"]["path"]),
    )
    if (
        rediscovery.get("decision")
        != "pass_one_of_one_synthetic_modular_exponent_holdout_rediscovered_and_proved"
        or rediscovery.get("proof", {}).get("winning_exponent") != 10
        or rediscovery.get("proof", {}).get("winning_residues_checked") != 10
        or rediscovery.get("claims", {}).get("winner_sealed_before_post_unseal_comparison")
        is not True
        or rediscovery.get("claims", {}).get("formal_proof_assistant_kernel_checked") is not False
    ):
        raise ValueError("modular-exponent rediscovery evidence changed")
    runtime = _load_bound(root, RUNTIME_BINDING["receipt"])
    validate_runtime_receipt(runtime, root=root)
    return rediscovery, runtime


def _adapter_config(executable: Path) -> LeanAdapterConfig:
    return LeanAdapterConfig(
        target=TARGET,
        allowed_premises=ALLOWED_PREMISES,
        forbidden_premises=FORBIDDEN_PREMISES,
        forbidden_prefixes=FORBIDDEN_PREFIXES,
        executable=executable,
        timeout_seconds=30.0,
    )


def build_live_receipt(
    config_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
    receipt_role: str = "live_replay",
) -> dict[str, Any]:
    root = config_path.resolve().parents[2]
    _load_config(root)
    rediscovery, _runtime = _evidence(root)
    runtime_config = json.loads(
        _inside(root, RUNTIME_BINDING["config"]["path"]).read_text(encoding="utf-8")
    )
    executable, resolution = _resolve_executable(runtime_config, environment=environment)
    if executable is None or resolution is None:
        raise ValueError("registered Lean executable unavailable")
    toolchain = _probe_toolchain(executable, _platform_key(), resolution)
    theorem = _inside(root, THEOREM_PATH)
    adapter = run_lean_adapter(_adapter_config(executable), theorem, environment={})
    if adapter.get("decision") != "pass_lean_checked_closed_premise":
        raise ValueError("modular-exponent Lean adapter did not pass")
    winner = rediscovery["pre_unseal"]["discovery"]["winner"]
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "receipt_role": receipt_role,
        "decision": "pass_rediscovered_modular_exponent_checked_by_real_lean_kernel",
        "first_blocker": "none_for_bounded_modulus_eleven_kernel_bridge",
        "rediscovery_evidence": {
            "benchmark_id": rediscovery["benchmark_id"],
            "artifact_content_sha256": rediscovery["content_sha256"],
            "winner_seal": rediscovery["chronology"][4]["seal"],
            "reference_unseal_ordinal": rediscovery["chronology"][5]["ordinal"],
            "modulus": 11,
            "exponent": winner["exponent"],
            "residues_checked": winner["residues_checked"],
            "exact_rows_sha256": winner["proof_sha256"],
        },
        "theorem_contract": {
            **WINNER_CONTRACT,
            "target": TARGET,
            "proof_method": "kernel-checked finite decision over all Fin 11 residues",
            "source_sha256": _file_sha(theorem),
            "source_bytes": theorem.stat().st_size,
            "sorry_or_axiom_used": False,
        },
        "dependency_closure": {
            "allowed_premise_manifest": build_allowed_premise_manifest(_adapter_config(executable)),
            "declared_dependencies": list(ALLOWED_PREMISES),
            "closure_checked_by_adapter": True,
            "withheld_theorem_used_as_premise": False,
        },
        "adapter_receipt": adapter,
        "toolchain_receipt": toolchain,
        "counts": {
            "rediscovery_winners_bound": 1,
            "finite_residue_obligations": 10,
            "kernel_executions": 1,
            "kernel_checked_theorems": 1,
            "blocked": 0,
            "rejected": 0,
        },
        "claims": CLAIMS,
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": CONFIG_SHA256},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "theorem": {"path": THEOREM_PATH, "file_sha256": THEOREM_SHA256},
            "rediscovery": REDISCOVERY_BINDING,
            "lean_runtime": RUNTIME_BINDING,
            "math_lean_adapter": ADAPTER_BINDING,
        },
        "scope": (
            "the single blinded modulus-11 exponent-10 winner and its ten finite nonzero-residue "
            "obligations only; no novelty, unbounded number theory, scientific, or physics claim"
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def validate_checked_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("modular-exponent bridge receipt seal changed")
    if re.search(r"[A-Za-z]:\\|/(?:home|Users)/", json.dumps(value, sort_keys=True)):
        raise ValueError("modular-exponent bridge persisted host path")
    _load_config(root)
    rediscovery, _runtime = _evidence(root)
    toolchain = value.get("toolchain_receipt", {})
    platform = str(toolchain.get("platform"))
    asset = PLATFORM_ASSETS.get(platform)
    expected_manifest = build_allowed_premise_manifest(
        LeanAdapterConfig(
            target=TARGET,
            allowed_premises=ALLOWED_PREMISES,
            forbidden_premises=FORBIDDEN_PREMISES,
            forbidden_prefixes=FORBIDDEN_PREFIXES,
        )
    )
    adapter = value.get("adapter_receipt", {})
    expected_rows = rediscovery["pre_unseal"]["discovery"]["winner"]
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("receipt_role") not in {"checked_windows_historical", "live_replay"}
        or value.get("decision") != "pass_rediscovered_modular_exponent_checked_by_real_lean_kernel"
        or value.get("claims") != CLAIMS
        or value.get("counts")
        != {
            "rediscovery_winners_bound": 1,
            "finite_residue_obligations": 10,
            "kernel_executions": 1,
            "kernel_checked_theorems": 1,
            "blocked": 0,
            "rejected": 0,
        }
        or value.get("rediscovery_evidence", {}).get("exponent") != expected_rows["exponent"]
        or value.get("rediscovery_evidence", {}).get("exact_rows_sha256")
        != expected_rows["proof_sha256"]
        or value.get("theorem_contract", {}).get("source_sha256") != THEOREM_SHA256
        or value.get("theorem_contract", {}).get("sorry_or_axiom_used") is not False
        or value.get("dependency_closure", {}).get("allowed_premise_manifest") != expected_manifest
        or value.get("dependency_closure", {}).get("withheld_theorem_used_as_premise") is not False
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
            "dependencies": list(ALLOWED_PREMISES),
            "closure_valid": True,
        }
        or value.get("source_bindings")
        != {
            "config": {"path": CONFIG_PATH, "file_sha256": CONFIG_SHA256},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "theorem": {"path": THEOREM_PATH, "file_sha256": THEOREM_SHA256},
            "rediscovery": REDISCOVERY_BINDING,
            "lean_runtime": RUNTIME_BINDING,
            "math_lean_adapter": ADAPTER_BINDING,
        }
    ):
        raise ValueError("modular-exponent bridge semantics changed")


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
        raise ValueError("modular-exponent bridge live replay changed")


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
