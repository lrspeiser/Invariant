"""Seal six-galaxy THINGS response maps without decoding response pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_rg_things_six_external_2d_replication_preflight_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_things_six_external_2d_replication_preflight_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_things_six_external_2d_replication_preflight_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-things-six-external-2d-replication-preflight-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-things-six-external-2d-replication-preflight-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-rg-things-six-external-2d-replication-preflight-receipt-1.0"
)
_OBJECTS = ("NGC2841", "IC2574", "DDO154", "NGC5055", "NGC6946", "NGC7331")
_RESOLUTIONS = ("NATURAL", "ROBUST")
_OBSERVABLES = ("MOM1", "MOM2")
_INCLINATIONS = {
    "NGC2841": 73.7,
    "IC2574": 53.4,
    "DDO154": 66.0,
    "NGC5055": 59.0,
    "NGC6946": 32.6,
    "NGC7331": 75.8,
}
_CONFIG_RAW_SHA256 = "5c2ddee868ea417148e424068103f92ddc1f4e019df93d759baf5e42546f6e7f"
_CONFIG_CONTENT_SHA256 = "a8fb5a952f2fce329d08c2517d521ae1f72d966e917bb1cb4c3b516ff55928bf"
_MODULE_SEMANTIC_SHA256 = "869def090dd4d337a666b76803e0090589a2a7ce653522aef706ff9cfe6a58ed"
_TEST_RAW_SHA256 = "f4c8ff4d2294671b6b4dedf7ad9a239c38a2df61867aa63e011de19f8e14d0af"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')
_BEAM_PATTERN = re.compile(
    r"AIPS\s+CLEAN\s+BMAJ=\s*([-+0-9.Ee]+)\s+BMIN=\s*([-+0-9.Ee]+)\s+BPA=\s*([-+0-9.Ee]+)"
)


class ReplicationPreflightError(RuntimeError):
    """Raised when response-blind preflight evidence changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplicationPreflightError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    normalized, count = _MODULE_PIN_PATTERN.subn(
        rb"\g<1>" + b"0" * 64 + rb"\g<2>", path.read_bytes()
    )
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    path = (_ROOT / relative).resolve()
    _require(path == _ROOT or _ROOT in path.parents, "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplicationPreflightError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def _fits_value(text: str) -> Any:
    token = text.split("/", 1)[0].strip()
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1].strip()
    if token in {"T", "F"}:
        return token == "T"
    try:
        return int(token)
    except ValueError:
        try:
            return float(token.replace("D", "E"))
        except ValueError:
            return token


def read_primary_header_only(path: Path) -> tuple[dict[str, Any], int]:
    cards: dict[str, Any] = {}
    history: list[str] = []
    blocks = 0
    with path.open("rb") as handle:
        while blocks < 64:
            block = handle.read(2880)
            _require(len(block) == 2880, "truncated FITS header")
            blocks += 1
            for offset in range(0, 2880, 80):
                raw = block[offset : offset + 80]
                try:
                    card = raw.decode("ascii")
                except UnicodeDecodeError as exc:
                    raise ReplicationPreflightError("non-ASCII FITS header") from exc
                key = card[:8].strip()
                if key == "END":
                    cards["__HISTORY__"] = history
                    return cards, blocks * 2880
                if key == "HISTORY":
                    history.append(card[8:].strip())
                    continue
                if key and card[8:10] == "= ":
                    _require(key not in cards, "duplicate FITS header key")
                    cards[key] = _fits_value(card[10:])
    raise ReplicationPreflightError("FITS END card not found")


def _beam_from_header(header: Mapping[str, Any]) -> list[float]:
    matches: list[list[float]] = []
    for line in header["__HISTORY__"]:
        match = _BEAM_PATTERN.search(line)
        if match:
            matches.append([float(match.group(index)) for index in (1, 2, 3)])
    _require(len(matches) >= 1, "beam history missing")
    return matches[-1]


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "OPAQUE_RESPONSE_ACQUISITION_COMPLETE_PIXELS_UNOPENED",
        "status changed",
    )
    objects = config["objects"]
    _require(tuple(row["object_id"] for row in objects) == _OBJECTS, "objects changed")
    _require(
        {row["object_id"]: row["inclination_deg"] for row in objects} == _INCLINATIONS,
        "inclinations changed",
    )
    _require(
        Counter(row["inclination_stratum"] for row in objects)
        == Counter({"HIGH": 3, "INTERMEDIATE": 2, "LOW": 1}),
        "inclination strata changed",
    )
    assets = config["response_assets"]
    _require(len(assets) == 24, "asset count changed")
    keys = [(row["object_id"], row["resolution"], row["observable"]) for row in assets]
    expected = [(o, r, m) for o in _OBJECTS for r in _RESOLUTIONS for m in _OBSERVABLES]
    _require(sorted(keys) == sorted(expected), "asset cross-product changed")
    _require(len(set(keys)) == 24, "duplicate response asset")
    accounting = config["acquisition_accounting"]
    _require(accounting["object_count"] == 6, "object accounting changed")
    _require(accounting["response_asset_count"] == 24, "asset accounting changed")
    _require(accounting["response_bytes"] == 102101760, "byte accounting changed")
    _require(accounting["head_calls"] == 27, "HEAD accounting changed")
    _require(accounting["get_calls"] == 24, "GET accounting changed")
    _require(accounting["response_headers_opened"] == 24, "header accounting changed")
    for key in (
        "response_pixels_decoded",
        "scientific_scores_computed",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        _require(accounting[key] == 0, f"forbidden access changed: {key}")
    prediction = config["prediction_contract"]
    _require(
        prediction["all_predictions_must_be_sealed_before_response_pixels_open"] is True,
        "response order changed",
    )
    _require(
        prediction["same_fixed_parameters_all_objects"] is True, "parameter universality changed"
    )
    _require(prediction["response_parameter_fitting"] is False, "response fitting enabled")
    _require(prediction["geometry_selection_after_response"] is False, "geometry tuning enabled")
    claims = config["claim_boundary"]
    _require(claims["public_response_assets_sealed"] is True, "asset seal suppressed")
    _require(claims["inclination_stratified_sample_frozen"] is True, "sample seal suppressed")
    for key in (
        "response_blind_predictions_complete",
        "response_scoring_complete",
        "refracted_gravity_replication_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim overpromoted: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output changed")


def _validate_package() -> None:
    if _MODULE_SEMANTIC_SHA256 != "0" * 64:
        _require(
            module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
            "module changed",
        )
    if _TEST_RAW_SHA256 != "0" * 64:
        _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    if _CONFIG_RAW_SHA256 != "0" * 64:
        _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _validate_package()
    return config


def validate_predecessors(config: Mapping[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for binding in config["predecessor_bindings"]:
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), f"missing predecessor: {binding['role']}")
            _require(
                file_sha256(path) == artifact["sha256"], f"changed predecessor: {binding['role']}"
            )
        receipt = _read_json(_repo_path(binding["artifacts"][-1]["path"]), binding["role"])
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            f"changed receipt: {binding['role']}",
        )
        output[binding["role"]] = receipt["content_sha256"]
    _require(len(output) == 4, "predecessor roles changed")
    return output


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    predecessor_receipts = validate_predecessors(config)
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    total_header_bytes = 0
    for asset in config["response_assets"]:
        path = _repo_path(asset["relative_path"])
        _require(path.is_file(), "response asset missing")
        _require(path.stat().st_size == asset["bytes"], "response asset size changed")
        _require(file_sha256(path) == asset["sha256"], "response asset bytes changed")
        header, header_bytes = read_primary_header_only(path)
        _require(header.get("SIMPLE") is True, "not a primary FITS image")
        _require(header.get("NAXIS") in {2, 4}, "unsupported response image rank")
        if header["NAXIS"] == 4:
            _require(
                header.get("NAXIS3") == 1 and header.get("NAXIS4") == 1,
                "higher response axes must be singleton",
            )
        _require(type(header.get("NAXIS1")) is int and header["NAXIS1"] > 0, "invalid NAXIS1")
        _require(type(header.get("NAXIS2")) is int and header["NAXIS2"] > 0, "invalid NAXIS2")
        _require(str(header.get("CTYPE1", "")).startswith("RA---"), "invalid CTYPE1")
        _require(str(header.get("CTYPE2", "")).startswith("DEC--"), "invalid CTYPE2")
        beam = _beam_from_header(header)
        rows.append(
            {
                "object_id": asset["object_id"],
                "resolution": asset["resolution"],
                "observable": asset["observable"],
                "relative_path": asset["relative_path"],
                "bytes": asset["bytes"],
                "sha256": asset["sha256"],
                "header_bytes_read": header_bytes,
                "shape": [header["NAXIS2"], header["NAXIS1"]],
                "wcs_ctype": [header["CTYPE1"], header["CTYPE2"]],
                "beam_deg": beam,
                "pixel_values_decoded": 0,
            }
        )
        total_bytes += asset["bytes"]
        total_header_bytes += header_bytes
    _require(
        total_bytes == config["acquisition_accounting"]["response_bytes"], "byte total changed"
    )
    rows.sort(key=lambda row: (row["object_id"], row["resolution"], row["observable"]))
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_OPAQUE_ACQUISITION_AND_HEADER_PREFLIGHT",
        "decision": "READY_TO_BUILD_ALL_SIX_GALAXY_RESPONSE_BLIND_PREDICTIONS",
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "predecessor_receipt_content_sha256": predecessor_receipts,
        "objects": config["objects"],
        "response_assets": rows,
        "response_asset_root_sha256": content_sha256(rows),
        "inventory_counts": {
            "objects": len(config["objects"]),
            "assets": len(rows),
            "opaque_response_bytes": total_bytes,
            "header_bytes_read": total_header_bytes,
            "response_pixels_decoded": 0,
        },
        "prediction_contract": config["prediction_contract"],
        "acquisition_accounting": config["acquisition_accounting"],
        "claim_boundary": config["claim_boundary"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    _require(dict(receipt) == build_receipt(config), "receipt differs from deterministic rebuild")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "refusing nonidentical overwrite")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent nonidentical output")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_receipt() -> str:
    config = load_config()
    return _atomic_no_clobber(
        _repo_path(OUTPUT_PATH), canonical_bytes(build_receipt(config)) + b"\n"
    )


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    validate_receipt(config, _read_json(path, "receipt"))
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    subparsers.add_parser("check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(write_receipt() if arguments.command == "build" else check_receipt())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
