"""Response-free source preflight for a three-galaxy model-lifted 3-D pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_phangs_things_model_lifted_3d_pilot_preflight_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_phangs_things_model_lifted_3d_pilot_preflight_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_phangs_things_model_lifted_3d_pilot_preflight_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-phangs-things-model-lifted-3d-pilot-preflight-v1/receipt.json"
)
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_RAW_SHA256 = "667358940eef51c8c0f9c2bea956eaeab73a435f424dab3c82f8dc803df36bda"
_CONFIG_CONTENT_SHA256 = "7c30778abf1578e31ca230732d3d6567bb5bae0d9bf70241aa8101ba376eb6a5"
_MODULE_SEMANTIC_SHA256 = "2016465c3334ecaea3d48d3fbf74486701f7ab5d287cc6a37872936c5a2e3a58"
_TEST_RAW_SHA256 = "419c3e12d28d9cc698e0722c5f99b1cf053e565f780d1c03c9e4d391c756af0b"
_SCHEMA = "invariant-open-gravity-phangs-things-model-lifted-3d-pilot-preflight-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-phangs-things-model-lifted-3d-pilot-preflight-receipt-1.0"
_EXPECTED_OBJECTS = ("NGC2903", "NGC3351", "NGC3627")
_EXPECTED_ROLES = {
    "STELLAR_FLUX",
    "STELLAR_ICA_MASK",
    "STELLAR_COLOR",
    "HI_MOM0_NATURAL_SENSITIVITY",
    "HI_MOM0_ROBUST_PRIMARY",
    "CO21_BROAD_MOM0",
    "CO21_BROAD_EMOM0",
}
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class PilotPreflightError(RuntimeError):
    """Raised when the frozen source-only contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotPreflightError(message)


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
    normalized, replacements = _MODULE_PIN_PATTERN.subn(rb"\g<1>" + (b"0" * 64) + rb"\g<2>", raw)
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
        raise PilotPreflightError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _git_show(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotPreflightError("committed predecessor binding unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "repository_head_at_freeze",
        "predecessor_bindings",
        "selection_contract",
        "object_metadata",
        "metadata_provenance",
        "source_files",
        "future_source_acquisition",
        "source_transform",
        "theory_comparison",
        "response_boundary",
        "construction_incident",
        "forbidden_inputs",
        "claims",
        "access_state",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected_keys, "config keys changed")
    _require(config["schema"] == _SCHEMA, "config schema changed")
    _require(
        config["package_id"] == "open-gravity-phangs-things-model-lifted-3d-pilot-preflight-v1",
        "package ID changed",
    )
    _require(
        config["status"] == "READY_SOURCE_ONLY_MODEL_LIFTED_3D_DEVELOPMENT_PILOT",
        "status changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")

    selection = config["selection_contract"]
    _require(selection["eligible_objects"] == list(_EXPECTED_OBJECTS), "objects changed")
    _require(selection["eligible_count"] == 3, "object count changed")
    _require(selection["reserved_confirmation_objects_opened"] == 0, "confirmation opened")
    _require(not selection["selection_used_rotation_values"], "selection used response")

    metadata = config["object_metadata"]
    _require(len(metadata) == 3, "metadata count changed")
    _require([row["object_id"] for row in metadata] == list(_EXPECTED_OBJECTS), "order changed")
    for row in metadata:
        expected_i = photometric_inclination_deg(row["s4g_outer_ellipticity"], 0.2)
        _require(
            math.isclose(
                expected_i,
                row["expected_photometric_inclination_deg"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"inclination mismatch for {row['object_id']}",
        )

    sources = config["source_files"]
    _require(len(sources) == 21, "source file count changed")
    _require(len({row["url"] for row in sources}) == 21, "duplicate source URL")
    _require(sum(row["bytes"] for row in sources) == 74_030_400, "byte ceiling changed")
    for object_id in _EXPECTED_OBJECTS:
        roles = {row["role"] for row in sources if row["object_id"] == object_id}
        _require(roles == _EXPECTED_ROLES, f"source roles changed for {object_id}")
    forbidden_url_tokens = ("MOM1", "MOM2", "CUBE", "VROT", "VELOCITY")
    for row in sources:
        upper = row["url"].upper()
        _require(
            not any(token in upper for token in forbidden_url_tokens),
            f"response-bearing URL admitted for {row['object_id']}",
        )
        _require(type(row["bytes"]) is int and row["bytes"] > 0, "invalid byte count")
        if row["survey"] == "THINGS":
            _require(
                row["url"].startswith("https://things.www3.mpia.de/Data_files/"),
                "THINGS source must use the direct no-redirect endpoint",
            )

    future = config["future_source_acquisition"]
    _require(future["exact_get_ceiling"] == 21, "GET ceiling changed")
    _require(future["exact_network_byte_ceiling"] == 74_030_400, "network bytes changed")
    _require(future["metadata_preflight_payload_files_downloaded"] == 0, "payload opened")
    _require(future["metadata_preflight_scientific_image_pixels_read"] == 0, "pixels read")
    _require(not config["source_transform"]["geometry_is_observed_3d"], "3-D overclaim")
    _require(config["response_boundary"]["fresh_confirmation_claim_allowed"] is False, "overclaim")
    _require(config["construction_incident"]["occurred"] is True, "incident hidden")
    _require(config["access_state"] == _zero_access_state(), "access state changed")
    claims = config["claims"]
    _require(claims["model_lifted_3d_source_ready_after_download_and_validation"], "scope lost")
    _require(
        not any(
            value
            for key, value in claims.items()
            if key != "model_lifted_3d_source_ready_after_download_and_validation"
        ),
        "claim ceiling changed",
    )


def _zero_access_state() -> dict[str, Any]:
    return {
        "source_payload_files_downloaded": 0,
        "source_payload_bytes_downloaded": 0,
        "scientific_image_pixels_read": 0,
        "response_rows_read_by_this_packet": 0,
        "scores_computed": 0,
        "models_fit": 0,
        "network_cost_usd": 0.0,
        "model_calls": 0,
        "paid_calls": 0,
    }


def photometric_inclination_deg(ellipticity: float, q0: float) -> float:
    _require(0.0 <= ellipticity < 1.0, "ellipticity out of range")
    _require(0.0 <= q0 < 1.0, "q0 out of range")
    q = 1.0 - ellipticity
    cos2 = max(0.0, min(1.0, (q * q - q0 * q0) / (1.0 - q0 * q0)))
    return math.degrees(math.acos(math.sqrt(cos2)))


def _stream_root(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _same_acceleration_fixture() -> dict[str, Any]:
    g = 1.0e-10
    c = 299_792_458.0
    au = 149_597_870_700.0
    pc = 3.085_677_581_491_367e16
    systems = [
        ("OUTER_SOLAR", 7000.0 * au),
        ("GALAXY_EDGE", 20_000.0 * pc),
        ("CLUSTER", 1_000_000.0 * pc),
    ]
    rows = []
    for label, radius_m in systems:
        row = {
            "system": label,
            "local_acceleration_m_s2": g,
            "radius_m": radius_m,
            "potential_proxy_abs_phi_over_c2": g * radius_m / (c * c),
            "curvature_proxy_s2": g / radius_m,
        }
        row["row_sha256"] = content_sha256(row)
        rows.append(row)
    _require(len({row["local_acceleration_m_s2"] for row in rows}) == 1, "g not matched")
    _require(
        rows[0]["potential_proxy_abs_phi_over_c2"]
        < rows[1]["potential_proxy_abs_phi_over_c2"]
        < rows[2]["potential_proxy_abs_phi_over_c2"],
        "potential discriminator collapsed",
    )
    _require(
        rows[0]["curvature_proxy_s2"]
        > rows[1]["curvature_proxy_s2"]
        > rows[2]["curvature_proxy_s2"],
        "curvature discriminator collapsed",
    )
    return {"rows": rows, "root_sha256": _stream_root(rows)}


def load_config() -> dict[str, Any]:
    config_path = _repo_path(CONFIG_PATH)
    module_path = _repo_path(MODULE_PATH)
    test_path = _repo_path(TEST_PATH)
    _require(file_sha256(config_path) == _CONFIG_RAW_SHA256, "config bytes changed")
    _require(module_semantic_sha256(module_path) == _MODULE_SEMANTIC_SHA256, "module changed")
    _require(file_sha256(test_path) == _TEST_RAW_SHA256, "tests changed")
    config = _read_json(config_path, "pilot config")
    validate_config(config)
    commit = config["repository_head_at_freeze"]
    for binding in config["predecessor_bindings"]:
        path = binding["path"]
        expected = binding["sha256"]
        _require(
            file_sha256(_repo_path(Path(path))) == expected, f"working predecessor changed: {path}"
        )
        _require(
            hashlib.sha256(_git_show(commit, path)).hexdigest() == expected,
            f"committed predecessor changed: {path}",
        )
    return config


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    object_rows = []
    for row in config["object_metadata"]:
        object_row = {
            "object_id": row["object_id"],
            "distance_mpc": row["distance_mpc"],
            "photometric_pa_deg": row["s4g_outer_pa_deg"],
            "photometric_inclination_deg": photometric_inclination_deg(
                row["s4g_outer_ellipticity"],
                config["source_transform"]["intrinsic_disk_axis_ratio_q0"],
            ),
            "source_file_count": sum(
                source["object_id"] == row["object_id"] for source in config["source_files"]
            ),
            "role": "DEVELOPMENT_EXPLORATION",
            "geometry": "MODEL_LIFTED_2P5D_TO_3D",
        }
        object_row["row_sha256"] = content_sha256(object_row)
        object_rows.append(object_row)

    source_rows = []
    for source in config["source_files"]:
        source_row = dict(source)
        source_row["payload_opened"] = False
        source_row["pixels_read"] = 0
        source_row["sha256_status"] = "LEARN_AFTER_EXACT_BYTE_DOWNLOAD"
        source_row["row_sha256"] = content_sha256(source_row)
        source_rows.append(source_row)

    fixture = _same_acceleration_fixture()
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": "PASS_SOURCE_ONLY_PREFLIGHT_ADVANCE_EXACT_21_FILE_ACQUISITION",
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
        "selection": {
            "eligible_objects": list(_EXPECTED_OBJECTS),
            "object_count": 3,
            "reserved_confirmation_opened": 0,
            "performance_selected": False,
            "object_root_sha256": _stream_root(object_rows),
            "objects": object_rows,
        },
        "sources": {
            "file_count": len(source_rows),
            "network_byte_ceiling": sum(row["bytes"] for row in source_rows),
            "source_root_sha256": _stream_root(source_rows),
            "records": source_rows,
        },
        "geometry": {
            "observed_3d": False,
            "label": "MODEL_LIFTED_2P5D_TO_3D",
            "stellar_height_cells": config["source_transform"][
                "stellar_height_over_radial_scale_cells"
            ],
            "gas_height_pc_cells": config["source_transform"]["gas_height_pc_cells"],
            "thickness_cell_count": 9,
            "response_tuned": False,
        },
        "discriminator": {
            "statement": config["theory_comparison"]["primary_discriminator"],
            "same_acceleration_fixture": fixture,
            "variables": config["theory_comparison"]["matched_acceleration_variables"],
        },
        "checks": {
            "exact_exhaustive_three_object_intersection": True,
            "exact_21_source_files": True,
            "exact_74030400_source_bytes": True,
            "no_velocity_or_cube_url": True,
            "photometric_not_kinematic_orientation": True,
            "model_lift_explicit": True,
            "matched_acceleration_separates_potential_and_curvature": True,
            "construction_response_exposure_disclosed": True,
            "fresh_confirmation_claim_refused": True,
            "zero_payload_access": True,
        },
        "access_state": dict(config["access_state"]),
        "claims": dict(config["claims"]),
        "next_action": "download exactly the 21 source-only files, verify exact bytes and FITS schemas, learn SHA256, then freeze the source-volume builder before development-response scoring",
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _require(type(receipt) is dict, "receipt is not an object")
    supplied = dict(receipt)
    claimed = supplied.pop("content_sha256", None)
    _require(type(claimed) is str and len(claimed) == 64, "receipt hash missing")
    _require(content_sha256(supplied) == claimed, "receipt self-hash mismatch")
    expected = build_receipt(config)
    _require(dict(receipt) == expected, "receipt does not exactly rebuild")


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    output = _repo_path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output.exists():
        existing = _read_json(output, "stored receipt")
        validate_receipt(existing, config)
        _require(output.read_text(encoding="utf-8") == payload, "stored bytes changed")
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
        raise PilotPreflightError("receipt appeared concurrently") from exc
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
        config = load_config()
        receipt = build_receipt(config)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "decision": receipt["decision"],
                    "objects": receipt["selection"]["object_count"],
                    "source_files": receipt["sources"]["file_count"],
                    "source_bytes": receipt["sources"]["network_byte_ceiling"],
                    "payload_files_opened": receipt["access_state"][
                        "source_payload_files_downloaded"
                    ],
                    "response_rows_read": receipt["access_state"][
                        "response_rows_read_by_this_packet"
                    ],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
