"""Source-and-paper preflight for a 12-galaxy outer-disk 3-D replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(
    "configs/open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_preflight_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_preflight_v1.py"
)
TEST_PATH = Path(
    "tests/test_open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_preflight_v1.py"
)
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-refracted-gravity-things-heracles-sparc-3d-expansion-preflight-v1/receipt.json"
)
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_RAW_SHA256 = "32c4dbb7de0d19f2ce2688c8f2764d46ad2ef97464d922e1f9816013e199ed5a"
_CONFIG_CONTENT_SHA256 = "9367ae2e64b0e8dc195ce02bdcf2dc2b05d192d3923a993e695d7a5c80f7e015"
_MODULE_SEMANTIC_SHA256 = "3be6b10ae299d8eeb81fb3d1188b8e27cfbe5c01263e13e74a0a9a60e60a61e9"
_TEST_RAW_SHA256 = "0b98edacd71f9470ad10df396d70f294cc0e3593aa113bd87cfb2b24813fedfe"
_SCHEMA = (
    "invariant-open-gravity-refracted-gravity-things-heracles-sparc-3d-expansion-preflight-1.0"
)
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-refracted-gravity-things-heracles-sparc-3d-expansion-"
    "preflight-receipt-1.0"
)
_STATUS = (
    "PASS_12_OBJECT_SOURCE_AND_PAPER_PREFLIGHT_SEVEN_REQUIRE_FIXED_STELLAR_"
    "CONVERSION_NO_RESPONSE_OPENED"
)
_OBJECTS = (
    "UGC04305",
    "NGC2841",
    "NGC2903",
    "NGC2976",
    "NGC3198",
    "IC2574",
    "NGC3521",
    "NGC4214",
    "DDO154",
    "NGC5055",
    "NGC6946",
    "NGC7331",
)
_PAPER_IDS = {
    "SINGS_KENNICUTT_2003",
    "S4G_QUEREJETA_2015",
    "MEIDT_2014_ML36",
    "THINGS_WALTER_2008",
    "HERACLES_LEROY_2009",
    "SPARC_LELLI_2016",
    "REFRACTED_GRAVITY_MATSAS_2016",
    "REFRACTED_GRAVITY_DISKMASS_2020",
}
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class ExpansionPreflightError(RuntimeError):
    """Raised when the response-free expansion contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpansionPreflightError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    raw = path.read_bytes()
    normalized, replacements = _MODULE_PIN_PATTERN.subn(rb"\g<1>" + b"0" * 64 + rb"\g<2>", raw)
    _require(replacements == 1, "module semantic pin structure changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path) -> Path:
    path = (_ROOT / relative).resolve()
    _require(path.is_relative_to(_ROOT), "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpansionPreflightError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _stream_root(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical(row))
        digest.update(b"\n")
    return digest.hexdigest()


def flatten_source_files(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for object_row in config["object_source_contracts"]:
        object_id = object_row["object_id"]
        stellar = object_row["stellar_branch"]
        for source in stellar["files"]:
            rows.append(
                {
                    "object_id": object_id,
                    "survey": stellar["survey"],
                    "disposition": stellar["disposition"],
                    **source,
                }
            )
        for source in object_row["hi_files"]:
            rows.append(
                {
                    "object_id": object_id,
                    "survey": "THINGS",
                    "disposition": "DATA_AND_PAPER_ADMITTED",
                    **source,
                }
            )
        for source in object_row["molecular_files"]:
            rows.append(
                {
                    "object_id": object_id,
                    "survey": "HERACLES",
                    "disposition": "DATA_AND_PAPER_ADMITTED",
                    **source,
                }
            )
    return rows


def validate_config(config: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "observed_at_utc",
        "admission_policy",
        "predecessor_bindings",
        "primary_papers",
        "selection_contract",
        "object_source_contracts",
        "transform_contract",
        "benchmark_contract",
        "future_acquisition",
        "response_boundary",
        "claims",
        "access_state",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected_keys, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"]
        == "open-gravity-refracted-gravity-things-heracles-sparc-3d-expansion-preflight-v1",
        "package ID changed",
    )
    _require(config["status"] == _STATUS, "status changed")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")

    policy = config["admission_policy"]
    _require(
        policy
        == {
            "path": "docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md",
            "sha256": "174e68e85872566d86d5007f152e75113ea3fbf1b0a3d375c61d65db71be672e",
            "rule": policy["rule"],
        },
        "admission policy binding changed",
    )
    _require("SOURCE_BLOCKED" in policy["rule"], "source-blocked rule missing")
    _require("3-D claim" in policy["rule"], "dimensional rule missing")

    selection = config["selection_contract"]
    _require(selection["admitted_objects"] == list(_OBJECTS), "admitted objects changed")
    _require(selection["candidate_count"] == 18, "parent count changed")
    _require(selection["admitted_count"] == 12, "admitted count changed")
    _require(len(selection["excluded_not_in_local_sparc"]) == 6, "excluded count changed")
    _require(not selection["selection_used_response_values"], "selection used response")
    _require(selection["response_rows_opened"] == 0, "response opened")

    papers = config["primary_papers"]
    _require(len(papers) == 8, "paper count changed")
    _require({row["paper_id"] for row in papers} == _PAPER_IDS, "paper inventory changed")
    _require(
        all(row["url"].startswith("https://arxiv.org/") for row in papers), "paper URL changed"
    )

    objects = config["object_source_contracts"]
    _require([row["object_id"] for row in objects] == list(_OBJECTS), "object order changed")
    _require(len(objects) == 12, "object count changed")
    builder_ready = 0
    conversion_required = 0
    for row in objects:
        stellar = row["stellar_branch"]
        if stellar["survey"] == "S4G_P5":
            builder_ready += 1
            _require(
                stellar["disposition"] == "DATA_AND_PAPER_ADMITTED_BUILDER_READY",
                "S4G disposition changed",
            )
            _require(
                {item["role"] for item in stellar["files"]}
                == {"STELLAR_MASS_MAP", "STELLAR_ICA_MASK", "STELLAR_COLOR_MAP"},
                "S4G roles changed",
            )
        else:
            conversion_required += 1
            _require(stellar["survey"] == "SINGS_IRAC1", "stellar survey changed")
            _require(
                stellar["disposition"] == "DATA_AND_PAPER_ADMITTED_CONVERSION_REQUIRED",
                "SINGS disposition changed",
            )
            _require(
                {item["role"] for item in stellar["files"]}
                == {"STELLAR_IRAC1_FLUX", "STELLAR_IRAC1_WEIGHT"},
                "SINGS roles changed",
            )
        _require(
            {item["role"] for item in row["hi_files"]} == {"HI_MOM0_NATURAL", "HI_MOM0_ROBUST"},
            "H I roles changed",
        )
        _require(
            {item["role"] for item in row["molecular_files"]} == {"CO21_MOM0", "CO21_EMOM0"},
            "CO roles changed",
        )
    _require(builder_ready == 5, "builder-ready count changed")
    _require(conversion_required == 7, "conversion-required count changed")

    sources = flatten_source_files(config)
    _require(len(sources) == 77, "source file count changed")
    _require(len({row["url"] for row in sources}) == 77, "duplicate source URL")
    _require(sum(row["bytes"] for row in sources) == 351_461_418, "source bytes changed")
    forbidden = ("MOM1", "MOM2", "VROT", "VELOCITY", ".HANS.")
    for row in sources:
        _require(type(row["bytes"]) is int and row["bytes"] > 0, "invalid source bytes")
        _require(row["url"].startswith("https://"), "non-HTTPS source")
        upper = row["url"].upper()
        _require(not any(token in upper for token in forbidden), "response/cube URL admitted")
        if row["survey"] == "THINGS":
            _require(
                row["url"].startswith("https://things.www3.mpia.de/Data_files/"),
                "THINGS direct endpoint changed",
            )
        if row["survey"] == "HERACLES":
            _require(
                row["url"].startswith("https://www.iram.fr/ILPA/LP001/"),
                "HERACLES official endpoint changed",
            )

    future = config["future_acquisition"]
    _require(future["exact_file_count"] == 77, "future file count changed")
    _require(future["exact_network_byte_ceiling"] == 351_461_418, "byte ceiling changed")
    _require(not future["payload_downloads_authorized_by_this_packet"], "download authorized")
    _require(future["selected_endpoint_head_observations"] == 77, "HEAD evidence changed")

    benchmark = config["benchmark_contract"]
    _require(benchmark["paper_only_is_not_data_validation"], "paper/data boundary lost")
    _require(
        benchmark["one_dimensional_response_cannot_validate_three_dimensional_solver"],
        "dimensional boundary lost",
    )
    _require(len(benchmark["required_before_real_scoring"]) == 7, "benchmark gates changed")

    response = config["response_boundary"]
    _require(response["sparc_membership_names_checked"] == 175, "SPARC name count changed")
    _require(response["sparc_velocity_values_opened"] == 0, "SPARC response opened")
    _require(response["scientific_image_pixels_read"] == 0, "image pixels opened")
    _require(response["rotation_response_rows_read"] == 0, "response rows opened")
    _require(response["scores_computed"] == 0, "scores computed")

    claims = config["claims"]
    _require(claims["twelve_object_real_source_and_primary_paper_path_exists"], "source claim lost")
    _require(
        claims["five_objects_ready_for_existing_s4g_mass_map_builder_after_download"],
        "ready claim lost",
    )
    _require(
        claims["seven_objects_require_new_frozen_sings_flux_to_mass_builder"],
        "conversion claim lost",
    )
    _require(
        not any(
            claims[key]
            for key in (
                "source_payloads_downloaded",
                "three_dimensional_fields_built",
                "rotation_curves_scored",
                "outer_disk_replication_passed",
                "refracted_gravity_supported",
                "discovery_or_publication_claim",
            )
        ),
        "claim ceiling changed",
    )


def load_config() -> dict[str, Any]:
    config_path = _repo_path(CONFIG_PATH)
    module_path = _repo_path(MODULE_PATH)
    test_path = _repo_path(TEST_PATH)
    _require(file_sha256(config_path) == _CONFIG_RAW_SHA256, "config bytes changed")
    _require(module_semantic_sha256(module_path) == _MODULE_SEMANTIC_SHA256, "module changed")
    _require(file_sha256(test_path) == _TEST_RAW_SHA256, "tests changed")
    config = _read_json(config_path, "expansion preflight config")
    validate_config(config)
    policy = config["admission_policy"]
    _require(
        file_sha256(_repo_path(Path(policy["path"]))) == policy["sha256"],
        "admission policy bytes changed",
    )
    for binding in config["predecessor_bindings"]:
        _require(
            file_sha256(_repo_path(Path(binding["path"]))) == binding["sha256"],
            f"predecessor changed: {binding['path']}",
        )
    return config


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    source_rows = []
    for source in flatten_source_files(config):
        row = {
            **source,
            "head_status": 200,
            "content_length_matched": True,
            "payload_downloaded": False,
            "scientific_pixels_read": 0,
            "sha256_status": "LEARN_AND_FREEZE_AFTER_EXACT_BYTE_DOWNLOAD",
        }
        row["row_sha256"] = content_sha256(row)
        source_rows.append(row)

    object_rows = []
    for item in config["object_source_contracts"]:
        stellar = item["stellar_branch"]
        row = {
            "object_id": item["object_id"],
            "stellar_survey": stellar["survey"],
            "stellar_disposition": stellar["disposition"],
            "stellar_file_count": len(stellar["files"]),
            "hi_file_count": len(item["hi_files"]),
            "molecular_file_count": len(item["molecular_files"]),
            "response_rows_opened": 0,
            "score_status": "SEALED_UNOPENED",
        }
        row["row_sha256"] = content_sha256(row)
        object_rows.append(row)

    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": (
            "PASS_SOURCE_AND_PAPER_PREFLIGHT_ADVANCE_5_DIRECT_S4G_AND_7_SINGS_"
            "CONVERSION_BUILDER_CASES_NO_RESPONSE"
        ),
        "config": {
            "path": CONFIG_PATH.as_posix(),
            "raw_sha256": _CONFIG_RAW_SHA256,
            "content_sha256": _CONFIG_CONTENT_SHA256,
        },
        "implementation": {
            "path": MODULE_PATH.as_posix(),
            "semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_path": TEST_PATH.as_posix(),
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "admission": {
            "object_count": 12,
            "s4g_builder_ready_after_download": 5,
            "sings_conversion_builder_required": 7,
            "source_blocked": 0,
            "paper_count": 8,
            "object_root_sha256": _stream_root(object_rows),
            "objects": object_rows,
        },
        "sources": {
            "selected_endpoint_count": 77,
            "network_byte_ceiling": 351_461_418,
            "source_root_sha256": _stream_root(source_rows),
            "records": source_rows,
        },
        "benchmarks": {
            "bound_reference_package": config["benchmark_contract"]["bound_reference_package"],
            "required_gate_count": 7,
            "paper_only_is_not_data_validation": True,
            "one_dimensional_response_cannot_validate_three_dimensional_solver": True,
        },
        "checks": {
            "exact_12_object_intersection": True,
            "exact_77_unique_source_endpoints": True,
            "exact_351461418_byte_ceiling": True,
            "all_endpoints_observed_http_200_with_matching_content_length": True,
            "eight_primary_papers_bound": True,
            "analytic_manufactured_and_reference_benchmarks_bound": True,
            "sings_conversion_not_pretended_complete": True,
            "no_velocity_map_or_cube_admitted": True,
            "no_scientific_payload_or_response_opened": True,
            "no_scoring_or_selection_by_performance": True,
        },
        "access_state": dict(config["access_state"]),
        "claims": dict(config["claims"]),
        "next_action": (
            "freeze and test the SINGS IRAC1 flux-to-stellar-mass conversion builder, then acquire "
            "and SHA-seal exactly 77 admitted files before building model-lifted 3-D sources"
        ),
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _require(type(receipt) is dict, "receipt is not an object")
    supplied = dict(receipt)
    claimed = supplied.pop("content_sha256", None)
    _require(type(claimed) is str and len(claimed) == 64, "receipt hash missing")
    _require(content_sha256(supplied) == claimed, "receipt self-hash mismatch")
    _require(dict(receipt) == build_receipt(config), "receipt does not exactly rebuild")


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    output = _repo_path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output.exists():
        validate_receipt(_read_json(output, "stored receipt"), config)
        _require(output.read_text(encoding="utf-8") == payload, "stored receipt bytes changed")
        return "EXISTING_IDENTICAL"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False, newline="\n"
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        if os.name != "nt":
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except FileExistsError as exc:
        raise ExpansionPreflightError("receipt appeared concurrently") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "CREATED"


def check_receipt() -> str:
    config = load_config()
    output = _repo_path(OUTPUT_PATH)
    _require(output.exists(), "receipt missing")
    validate_receipt(_read_json(output, "stored receipt"), config)
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        print(check_receipt())
    else:
        receipt = build_receipt(load_config())
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "decision": receipt["decision"],
                    "objects": receipt["admission"]["object_count"],
                    "s4g_builder_ready": receipt["admission"]["s4g_builder_ready_after_download"],
                    "sings_conversion_required": receipt["admission"][
                        "sings_conversion_builder_required"
                    ],
                    "source_files": receipt["sources"]["selected_endpoint_count"],
                    "source_bytes": receipt["sources"]["network_byte_ceiling"],
                    "response_rows_read": receipt["access_state"]["response_rows_read"],
                    "scores_computed": receipt["access_state"]["scores_computed"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
