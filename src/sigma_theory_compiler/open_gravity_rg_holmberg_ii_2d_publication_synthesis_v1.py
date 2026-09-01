"""Position the sealed Holmberg II 2D RG result against prior work and earlier objects."""

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

CONFIG_PATH = Path("configs/open_gravity_rg_holmberg_ii_2d_publication_synthesis_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_holmberg_ii_2d_publication_synthesis_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_holmberg_ii_2d_publication_synthesis_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-holmberg-ii-2d-publication-synthesis-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-holmberg-ii-2d-publication-synthesis-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-holmberg-ii-2d-publication-synthesis-receipt-1.0"
_RG = "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"
_CONFIG_RAW_SHA256 = "1af2981446e8ad66bb3b7dcdd404d7ab3077e174339bc4075f939b4b02b634b8"
_CONFIG_CONTENT_SHA256 = "0743553ffbdcfabc4220676d5f0428c00250efd4c9f951a95579461e649c9d51"
_MODULE_SEMANTIC_SHA256 = "a6c7673d4a179544eb7b8118e27233f49df285ca71b898e4f664f50123f2e8a0"
_TEST_RAW_SHA256 = "da549888f6280067773a4e3d7bc2e01cb145392bc4ba7459294366b159d1efa2"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class PublicationSynthesisError(RuntimeError):
    """Raised when a sealed synthesis input or claim changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationSynthesisError(message)


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
        raise PublicationSynthesisError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def _package_bindings() -> dict[str, str]:
    return {
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "POST_RESPONSE_RETROSPECTIVE_PUBLICATION_NOVELTY_SYNTHESIS",
        "status changed",
    )
    program = config["fixed_program_contract"]
    _require(program["refracted_gravity_candidate_id"] == _RG, "candidate changed")
    _require(program["epsilon_0"] == 0.661, "epsilon changed")
    _require(program["Q"] == 1.79, "Q changed")
    _require(program["log10_rho_c_g_cm3"] == -24.54, "density threshold changed")
    _require(
        program["holmberg_external_cell_id"] == "UGC04305__IRAC1_FIXED_ML0P6__I27P0__ROBUST",
        "external cell changed",
    )
    _require(program["response_parameter_fitting"] is False, "response fitting enabled")
    _require(program["post_response_candidate_repair"] is False, "candidate repair enabled")
    evidence = config["external_inclination_evidence"]
    _require(
        [row["id"] for row in evidence] == ["GENTILE_ET_AL_2012", "SANCHEZ_SALCEDO_ET_AL_2014"],
        "inclination evidence changed",
    )
    _require(
        all(row["independent_of_current_score"] is True for row in evidence),
        "evidence dependence changed",
    )
    corpus = config["bounded_primary_rg_corpus"]
    novelty = config["novelty_contract"]
    _require(len(corpus) == novelty["bounded_corpus_size"] == 6, "corpus count changed")
    _require(
        all(row["two_dimensional_velocity_map_score"] is False for row in corpus),
        "prior 2D test omitted",
    )
    _require(all(row["holmberg_ii"] is False for row in corpus), "prior Holmberg test omitted")
    _require(
        novelty["global_priority_or_exhaustive_literature_claim"] is False, "priority overclaim"
    )
    synthesis = config["retrospective_synthesis_contract"]
    _require(synthesis["base_object_count"] == 5, "base count changed")
    _require(synthesis["extended_object_count"] == 6, "extended count changed")
    _require(synthesis["minimum_total_signal_objects"] == 3, "signal threshold changed")
    _require(synthesis["minimum_new_response_blind_signal_objects"] == 2, "blind threshold changed")
    _require(
        synthesis["threshold_carried_forward_after_response"] is True, "retrospective status hidden"
    )
    _require(synthesis["confirmation"] is False, "confirmation overclaim")
    access = config["access_accounting"]
    _require(access["sealed_aggregate_receipts_opened"] == 3, "receipt accounting changed")
    for key in (
        "response_files_opened",
        "response_pixels_decoded",
        "predictions_recomputed",
        "scores_recomputed",
        "network_calls_during_deterministic_rebuild",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        _require(access[key] == 0, f"forbidden synthesis access: {key}")
    claims = config["claim_boundary"]
    _require(claims["bounded_primary_corpus_novelty_candidate"] is True, "novelty suppressed")
    _require(
        claims["retrospective_six_object_development_threshold_met"] is True, "signal suppressed"
    )
    for key in (
        "holmberg_external_inclination_resolved",
        "independent_confirmation",
        "refracted_gravity_generally_preferred",
        "unique_gravity_theory_established",
        "global_literature_priority_established",
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


def _load_predecessors(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
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
        loaded[binding["role"]] = receipt
    _require(
        set(loaded)
        == {
            "FIVE_OBJECT_2D_EXPANSION",
            "HOLMBERG_II_FIXED_2D_SCORE",
            "HOLMBERG_II_POSTSCORE_ROBUSTNESS",
        },
        "predecessor roles changed",
    )
    return loaded


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    inputs = _load_predecessors(config)
    five = inputs["FIVE_OBJECT_2D_EXPANSION"]
    score = inputs["HOLMBERG_II_FIXED_2D_SCORE"]
    robustness = inputs["HOLMBERG_II_POSTSCORE_ROBUSTNESS"]
    _require(
        five["decision"]
        == "NGC2976_RG_SIGNAL_DID_NOT_GENERALIZE_TO_PREREGISTERED_FIVE_OBJECT_GATE",
        "five-object decision changed",
    )
    _require(
        robustness["decision"]
        == "FOLLOW_UP_WORTHY_INCLINATION_CONDITIONAL_RG_PATTERN_NOT_GENERAL_PREFERENCE",
        "Holmberg robustness decision changed",
    )
    cell_id = config["fixed_program_contract"]["holmberg_external_cell_id"]
    selected = [row for row in score["scores"] if row["cell_score_id"] == cell_id]
    _require(len(selected) == 1, "external Holmberg cell missing or duplicated")
    holmberg = selected[0]
    _require(holmberg["winner"] == _RG, "external Holmberg cell no longer favors RG")
    _require(holmberg["rg_beats_all_three_comparators"] is True, "Holmberg comparator gate changed")
    synthesis = config["retrospective_synthesis_contract"]
    base_signals = list(five["expansion_gate"]["signal_objects"])
    _require(base_signals == synthesis["base_signal_objects"], "base signal objects changed")
    signals = sorted([*base_signals, synthesis["holmberg_signal_object"]])
    new_signals = sorted(
        [*five["expansion_gate"]["new_blind_signal_objects"], synthesis["holmberg_signal_object"]]
    )
    _require(
        signals == sorted(synthesis["expected_extended_signal_objects"]), "signal synthesis changed"
    )
    _require(
        new_signals == sorted(synthesis["expected_extended_new_response_blind_signal_objects"]),
        "new signal synthesis changed",
    )
    gate = (
        len(signals) >= synthesis["minimum_total_signal_objects"]
        and len(new_signals) >= synthesis["minimum_new_response_blind_signal_objects"]
    )
    _require(gate, "retrospective development threshold not met")
    corpus = config["bounded_primary_rg_corpus"]
    bounded_novelty = all(
        row["two_dimensional_velocity_map_score"] is False for row in corpus
    ) and all(row["holmberg_ii"] is False for row in corpus)
    _require(bounded_novelty, "bounded novelty condition changed")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_BOUNDED_NOVELTY_AND_RETROSPECTIVE_SIGNAL_SYNTHESIS",
        "decision": "PUBLISHABLE_METHOD_AND_CONDITIONAL_RG_SIGNAL_CANDIDATE_REQUIRES_EXTERNAL_REPLICATION",
        "package_bindings": _package_bindings(),
        "predecessor_receipt_content_sha256": {
            role: receipt["content_sha256"] for role, receipt in sorted(inputs.items())
        },
        "fixed_program": config["fixed_program_contract"],
        "holmberg_external_cell": {
            "cell_score_id": holmberg["cell_score_id"],
            "winner": holmberg["winner"],
            "common_pixel_count": holmberg["common_pixel_count"],
            "rmse_m_s": {
                candidate: holmberg["models"][candidate]["rmse_m_s"]
                for candidate in (
                    "NEWTON_3D_DST",
                    "MOND_STANDARD_MU_ON_NEWTON_3D",
                    "RAR_2016_ON_NEWTON_3D",
                    _RG,
                )
            },
            "all_three_comparators_beaten": holmberg["rg_beats_all_three_comparators"],
            "external_inclination_is_plausible_not_resolved": True,
        },
        "retrospective_six_object_gate": {
            "object_count": synthesis["extended_object_count"],
            "signal_objects": signals,
            "signal_object_count": len(signals),
            "new_response_blind_signal_objects": new_signals,
            "new_response_blind_signal_object_count": len(new_signals),
            "minimum_signal_objects": synthesis["minimum_total_signal_objects"],
            "minimum_new_response_blind_signal_objects": synthesis[
                "minimum_new_response_blind_signal_objects"
            ],
            "threshold_met": gate,
            "retrospective_not_confirmation": True,
        },
        "bounded_literature_audit": {
            "primary_rg_papers": len(corpus),
            "paper_ids": [row["id"] for row in corpus],
            "prior_2d_velocity_map_rg_test_identified": False,
            "prior_holmberg_ii_rg_test_identified": False,
            "global_priority_established": False,
            "novelty_scope": "NO_PRIOR_IDENTIFIED_IN_FROZEN_BOUNDED_PRIMARY_CORPUS",
        },
        "external_inclination_evidence": config["external_inclination_evidence"],
        "next_test_contract": config["next_test_contract"],
        "access_accounting": config["access_accounting"],
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
    receipt = build_receipt(config)
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt) + b"\n")


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    validate_receipt(config, _read_json(path, "receipt"))
    return "VALID"


def status() -> dict[str, Any]:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    if not path.is_file():
        return {"package_id": config["package_id"], "status": "FROZEN_UNRUN"}
    receipt = _read_json(path, "receipt")
    return {
        "package_id": config["package_id"],
        "status": receipt["status"],
        "decision": receipt["decision"],
        "retrospective_six_object_gate": receipt["retrospective_six_object_gate"],
        "bounded_literature_audit": receipt["bounded_literature_audit"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    subparsers.add_parser("check")
    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        print(write_receipt())
    elif arguments.command == "check":
        print(check_receipt())
    else:
        print(json.dumps(status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
