"""Recovery-option-1 V2 pre-score contract for real B+E+N development.

V2 never loads SPARC or X-COP scientific payloads and never computes a real
metric.  It reclassifies all locally accessible SPARC material as development
only, freezes a predictor-only input mapping, and blocks because the X-COP
physical output mapping still needs response-derived quantities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_shared_target_blind_ben_real_development_preflight_v2.json")
TEST_PATH = Path("tests/test_gravity_shared_target_blind_ben_real_development_preflight_v2.py")
AUTHORIZATION_PATH = Path(
    "runs/gravity/shared-target-blind-ben-real-development-preflight-v2/authorization-v1.json"
)
RECEIPT_PATH = Path("runs/gravity/shared-target-blind-ben-real-development-preflight-v2.json")

CONFIG_SCHEMA = "invariant-gravity-ben-real-development-preflight-config-2.0"
AUTHORIZATION_SCHEMA = "invariant-gravity-ben-real-development-authorization-2.0"
RECEIPT_SCHEMA = "invariant-gravity-ben-real-development-preflight-receipt-2.0"
DECISION = "BLOCKED_XCOP_RESPONSE_DERIVED_OUTPUT_MAPPING_UNAUTHORIZED_NO_SCORE"
CONFIG_FILE_SHA256 = "5970f7a898a1661a3cdd228d56cac1a112f6a0d2c2691dc190a18257dd01a804"
TEST_FILE_SHA256 = "51af62d12bdf3df81f31c0e263cf615ce252989d123f0eabffe2d139dbcde472"

SECTION_SHA256 = {
    "source_bindings": "06a1e63b958328b6aec86d811e4ce145d52ab904dd998099882356acca70b6c6",
    "lineage": "6bf70891808d1747e9b8bfac419fe3464ba8153a42b64e85be1af18a36262ec3",
    "sparc_reclassification": "b2d0c567d9b075b2eaf09af63d784ba07bafadd5c9971d5dfb012b3f56ee397c",
    "planned_populations": "4663e4ce8229ed1e760bd78fb5d540c91cbfa8fc7d831fd99fe470f12cb4f255",
    "item61_metric_reuse": "c34610eff3cc06617394a19b7a1d92816d57426b5a1ded7177a33535dc20554a",
    "predictor_only_mapping": "7d2ca417cca9ded2a20f4371c8032677a0974b72b85ba90a555adfc9b9b845a5",
    "candidate_registry": "4afa8d0151bc9a26dd7580d60cbbba123131331bd6b8796f6bff3cb4060e03b9",
    "selection_and_ablation": "135d4c089cb53ae364a0bccbea72c7f00dab7f05b3a43d176f5ed5e8eec11907",
    "compute_ceiling": "52106211b638cb4bcc51e70ced3a7857b329ca33d440f32ab6230b30dc4fd35c",
    "zero_access_chronology": "7b0c937362f7ec4a627d315ac56b473574bbe6d79bcf87b4a939cb9bc0dbe0cb",
    "production_gate": "43546bb52a35da2c049b4081b778ae8f257ec236d48f6cc94b6707147e78b53e",
    "approval_schema": "6992890be6ccd24df944daee077589aef0a041609b5296d9ac7684defe5bbe57",
    "claim_boundary": "00f3e70997856257c0843990593a50f479afb496a14461b530c54f348cdd5b3d",
    "output_paths": "da444763575a8f5cffc51da32f5708584dc14909325b4293361379a03b336561",
}

EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "purpose",
    "implementation_source",
    "verifier_test",
    *SECTION_SHA256,
}
EXPECTED_XCOP_OBJECTS = [
    "A1644",
    "A1795",
    "A2142",
    "A2255",
    "A2319",
    "A3266",
    "A85",
    "ZW1215",
]
EXPECTED_AUTHORIZATION_KEYS = {
    "schema_version",
    "authorization_id",
    "authorized",
    "approved_by",
    "approved_at",
    "config_file_sha256",
    "preflight_receipt_file_sha256",
    "preflight_receipt_content_sha256",
    "incident_commit",
    "candidate_registry_content_sha256",
    "sparc_role",
    "sparc_objects",
    "sparc_rows",
    "xcop_role",
    "xcop_objects",
    "compute_ceiling",
    "mapping_resolution_contract",
    "claim_acknowledgements",
}
EXPECTED_ACKNOWLEDGEMENTS = {
    "no_local_sparc_confirmation_claim": True,
    "development_only_selection": True,
    "no_single_counterexample_veto": True,
    "no_family_pruning": True,
    "no_publication_or_gr_replacement_claim": True,
}


class BENRealDevelopmentPreflightV2Error(RuntimeError):
    """Raised when the V2 pre-score boundary changes or a gate fails."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256((canonical_json(value) + "\n").encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section_sha256(value: Any) -> str:
    return hashlib.sha256((canonical_json(value) + "\n").encode()).hexdigest()


def confined(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise BENRealDevelopmentPreflightV2Error(f"path escaped repository: {path}") from error
    return target


def read_json(path: Path) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file():
        raise BENRealDevelopmentPreflightV2Error(f"required artifact absent: {path}")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BENRealDevelopmentPreflightV2Error(f"expected JSON object: {path}")
    return value


def strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise BENRealDevelopmentPreflightV2Error(f"{label} keys changed")


def validate_config(config: Mapping[str, Any]) -> None:
    strict_keys(config, EXPECTED_TOP_LEVEL_KEYS, "V2 config")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "blocked_predictor_only_xcop_output_mapping_and_unauthorized"
        or config["purpose"]
        != (
            "Prepare recovery option 1 without opening or scoring real rows: every "
            "locally accessible SPARC row is development-only for this descendant, the "
            "planned SPARC score remains restricted to the historical 139-object/2720-row "
            "subset, and no local confirmation claim survives."
        )
        or config["implementation_source"]
        != "src/sigma_theory_compiler/gravity_shared_target_blind_ben_real_development_preflight_v2.py"
        or config["verifier_test"]
        != {"path": TEST_PATH.as_posix(), "file_sha256": TEST_FILE_SHA256}
    ):
        raise BENRealDevelopmentPreflightV2Error("V2 identity changed")
    for section, expected in SECTION_SHA256.items():
        if section_sha256(config[section]) != expected:
            raise BENRealDevelopmentPreflightV2Error(f"frozen {section} changed")
    reclassified = config["sparc_reclassification"]
    if (
        reclassified["all_rows_in_locally_accessible_mixed_packet_role"]
        != "development_only_for_this_descendant"
        or reclassified["local_confirmation_role_exists"] is not False
        or reclassified["local_confirmation_claim_allowed"] is not False
        or reclassified["planned_score_subset_objects"] != 139
        or reclassified["planned_score_subset_rows"] != 2720
        or reclassified["rows_outside_planned_score_subset_scored"] != 0
    ):
        raise BENRealDevelopmentPreflightV2Error("SPARC reclassification weakened")
    mapping = config["predictor_only_mapping"]
    if (
        mapping["sparc"]["response_fields_used_in_mapping"] != []
        or mapping["xcop"]["response_fields_used_in_input_mapping"] != []
        or mapping["xcop"]["candidate_output_projection"] is not None
        or mapping["xcop"]["output_mapping_ready"] is not False
        or mapping["xcop"]["failure_action"] != "BLOCK_BEFORE_PAYLOAD_LOAD_AND_SCORE"
        or mapping["response_used_for_candidate_generation"] is not False
    ):
        raise BENRealDevelopmentPreflightV2Error("predictor-only mapping boundary weakened")
    if config["planned_populations"]["xcop"]["objects"] != EXPECTED_XCOP_OBJECTS:
        raise BENRealDevelopmentPreflightV2Error("X-COP development population changed")
    if (
        config["production_gate"]["payload_loader_present_in_v2"] is not False
        or config["production_gate"]["authorization_cannot_override_mapping_blocker"] is not True
        or config["claim_boundary"]["production_authorized"] is not False
        or config["claim_boundary"]["real_scoring_executed"] is not False
        or config["claim_boundary"]["local_sparc_confirmation_claim_survives"] is not False
    ):
        raise BENRealDevelopmentPreflightV2Error("production or claim gate weakened")


def validate_bound_files(config: Mapping[str, Any]) -> None:
    test = confined(ROOT / str(config["verifier_test"]["path"]))
    if not test.is_file() or file_sha256(test) != TEST_FILE_SHA256:
        raise BENRealDevelopmentPreflightV2Error("V2 test binding changed")
    for label, binding in config["source_bindings"].items():
        target = confined(ROOT / str(binding["path"]))
        if not target.is_file() or file_sha256(target) != binding["file_sha256"]:
            raise BENRealDevelopmentPreflightV2Error(f"source binding missing or changed: {label}")


def load_config() -> dict[str, Any]:
    target = confined(ROOT / CONFIG_PATH)
    if file_sha256(target) != CONFIG_FILE_SHA256:
        raise BENRealDevelopmentPreflightV2Error("V2 config hash changed")
    config = read_json(target)
    validate_config(config)
    validate_bound_files(config)
    return config


def validate_authorization(
    authorization: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    require_authorized: bool,
) -> None:
    strict_keys(authorization, EXPECTED_AUTHORIZATION_KEYS, "authorization")
    if (
        authorization["schema_version"] != AUTHORIZATION_SCHEMA
        or authorization["config_file_sha256"] != CONFIG_FILE_SHA256
        or authorization["incident_commit"] != config["lineage"]["incident_commit"]
        or authorization["candidate_registry_content_sha256"]
        != config["candidate_registry"]["content_sha256"]
        or authorization["sparc_role"] != "development_only"
        or authorization["sparc_objects"] not in (0, 139)
        or authorization["sparc_rows"] not in (0, 2720)
        or authorization["xcop_role"] != "development_only"
        or authorization["xcop_objects"] not in ([], EXPECTED_XCOP_OBJECTS)
        or authorization["claim_acknowledgements"] != EXPECTED_ACKNOWLEDGEMENTS
    ):
        raise BENRealDevelopmentPreflightV2Error("authorization boundary changed")
    if authorization["authorized"] is False:
        if require_authorized:
            raise BENRealDevelopmentPreflightV2Error("UNAUTHORIZED_BEFORE_PAYLOAD_LOAD")
        expected_empty = {
            "authorization_id": None,
            "approved_by": None,
            "approved_at": None,
            "preflight_receipt_file_sha256": None,
            "preflight_receipt_content_sha256": None,
            "sparc_objects": 0,
            "sparc_rows": 0,
            "xcop_objects": [],
            "compute_ceiling": {
                "cpu_formula_domain_batches": 0,
                "gpu_formula_domain_batches": 0,
                "cpu_gpu_parity_comparisons": 0,
                "payload_file_opens": 0,
                "paid_calls": 0,
                "maximum_api_spend_usd": 0.0,
            },
            "mapping_resolution_contract": None,
        }
        if any(authorization[key] != value for key, value in expected_empty.items()):
            raise BENRealDevelopmentPreflightV2Error("unauthorized manifest grants work")
        return
    if authorization["authorized"] is not True:
        raise BENRealDevelopmentPreflightV2Error("authorization flag is not boolean")
    receipt = read_json(ROOT / RECEIPT_PATH)
    if (
        not authorization["authorization_id"]
        or not authorization["approved_by"]
        or not authorization["approved_at"]
        or authorization["preflight_receipt_file_sha256"] != file_sha256(ROOT / RECEIPT_PATH)
        or authorization["preflight_receipt_content_sha256"] != receipt["content_sha256"]
        or authorization["sparc_objects"] != 139
        or authorization["sparc_rows"] != 2720
        or authorization["xcop_objects"] != EXPECTED_XCOP_OBJECTS
        or authorization["compute_ceiling"] != config["compute_ceiling"]
        or not isinstance(authorization["mapping_resolution_contract"], dict)
    ):
        raise BENRealDevelopmentPreflightV2Error("authorized artifact is incomplete")


def artifact_binding(path: Path, *, content: bool = False) -> dict[str, Any]:
    target = confined(path)
    binding: dict[str, Any] = {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }
    if content:
        binding["content_sha256"] = read_json(target)["content_sha256"]
    return binding


def build_receipt(config: Mapping[str, Any], authorization: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "v2_pre_score_contract_blocked_no_payload_access",
        "decision": DECISION,
        "evidence": {
            "config": artifact_binding(ROOT / CONFIG_PATH),
            "source": artifact_binding(Path(__file__)),
            "tests": artifact_binding(ROOT / TEST_PATH),
            "current_unauthorized_manifest": artifact_binding(ROOT / AUTHORIZATION_PATH),
            **{label: dict(binding) for label, binding in config["source_bindings"].items()},
        },
        "lineage": config["lineage"],
        "sparc_reclassification": config["sparc_reclassification"],
        "planned_populations": config["planned_populations"],
        "item61_metric_reuse": config["item61_metric_reuse"],
        "predictor_only_mapping": config["predictor_only_mapping"],
        "mapping_decision": {
            "sparc_mapping_ready": True,
            "xcop_input_mapping_ready": True,
            "xcop_output_mapping_ready": False,
            "blocked_before_payload_load": True,
            "reason": config["predictor_only_mapping"]["xcop"]["blocker"],
        },
        "candidate_registry": config["candidate_registry"],
        "selection_and_ablation": config["selection_and_ablation"],
        "future_compute_ceiling": config["compute_ceiling"],
        "zero_access_chronology": config["zero_access_chronology"],
        "current_authorization": {
            "path": AUTHORIZATION_PATH.as_posix(),
            "authorized": authorization["authorized"],
            "authorized_payload_file_opens": authorization["compute_ceiling"]["payload_file_opens"],
            "authorized_cpu_formula_domain_batches": authorization["compute_ceiling"][
                "cpu_formula_domain_batches"
            ],
            "authorized_gpu_formula_domain_batches": authorization["compute_ceiling"][
                "gpu_formula_domain_batches"
            ],
            "authorized_paid_calls": authorization["compute_ceiling"]["paid_calls"],
        },
        "production_gate": config["production_gate"],
        "approval_schema": config["approval_schema"],
        "claims": config["claim_boundary"],
        "limitations": [
            "Every locally accessible SPARC row is development-only for this descendant; no local confirmation claim survives.",
            "A future SPARC score is restricted to the historical 139-object/2720-row subset despite the broader development-only reclassification.",
            "The X-COP input adapter is predictor-only, but a physical pressure/temperature output requires a measured boundary and response scaling; V2 therefore blocks before payload load.",
            "The current authorization manifest grants zero access and zero compute, and authorization alone cannot override the mapping blocker.",
            "No real metric, score, selection, ablation, CPU/GPU production call, or scientific claim exists here.",
        ],
    }
    return {**body, "content_sha256": content_sha256(body)}


def write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    target = confined(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as error:
        raise BENRealDevelopmentPreflightV2Error(
            f"atomic no-clobber publication refused existing artifact: {target}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write() -> dict[str, Any]:
    config = load_config()
    authorization = read_json(ROOT / AUTHORIZATION_PATH)
    validate_authorization(authorization, config, require_authorized=False)
    receipt = build_receipt(config, authorization)
    write_json_no_clobber(ROOT / RECEIPT_PATH, receipt)
    return receipt


def check() -> dict[str, Any]:
    config = load_config()
    authorization = read_json(ROOT / AUTHORIZATION_PATH)
    validate_authorization(authorization, config, require_authorized=False)
    receipt = read_json(ROOT / RECEIPT_PATH)
    if receipt != build_receipt(config, authorization):
        raise BENRealDevelopmentPreflightV2Error("V2 receipt does not reconstruct")
    return {
        "valid": True,
        "decision": receipt["decision"],
        "authorized": False,
        "mapping_ready": False,
        "real_candidates_scored": 0,
        "payload_rows_read": 0,
        "receipt_sha256": file_sha256(ROOT / RECEIPT_PATH),
    }


def production_preflight(authorization_path: Path) -> None:
    config = load_config()
    stored = read_json(ROOT / RECEIPT_PATH)
    current = read_json(ROOT / AUTHORIZATION_PATH)
    if stored != build_receipt(config, current):
        raise BENRealDevelopmentPreflightV2Error("frozen preflight receipt changed")
    authorization = read_json(authorization_path)
    validate_authorization(authorization, config, require_authorized=True)
    raise BENRealDevelopmentPreflightV2Error(
        "BLOCKED_XCOP_RESPONSE_DERIVED_OUTPUT_MAPPING_BEFORE_PAYLOAD_LOAD"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write")
    commands.add_parser("check")
    production = commands.add_parser("production-preflight")
    production.add_argument("--authorization", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "write":
        result: Any = write()
    elif args.command == "check":
        result = check()
    else:
        production_preflight(args.authorization)
        result = None
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
