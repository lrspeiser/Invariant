"""Assess seven-galaxy RG two-dimensional robustness and publication scope."""

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

import numpy as np

CONFIG_PATH = Path("configs/open_gravity_rg_seven_galaxy_2d_external_robustness_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_seven_galaxy_2d_external_robustness_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_seven_galaxy_2d_external_robustness_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-seven-galaxy-2d-external-robustness-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-seven-galaxy-2d-external-robustness-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-seven-galaxy-2d-external-robustness-receipt-1.0"
_CANDIDATES = (
    "NEWTON_3D_DST",
    "RAR_2016_ON_NEWTON_3D",
    "MOND_STANDARD_MU_ON_NEWTON_3D",
    "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG",
)
_RG = _CANDIDATES[-1]
_CONFIG_RAW_SHA256 = "c3edfd02884c626cb673c2dce0f9472ff97a74551b30910a5413b2ba5e2d2a4b"
_CONFIG_CONTENT_SHA256 = "d92c77224d3c2a0d14a94ac16e00dc5fce30ab2a5bbe049d5c3acc8863325a6e"
_MODULE_SEMANTIC_SHA256 = "98197ec7fe6fe16be65dc9b95b5becb49339cc5d9a42aa93e0d76b9192d7dbb6"
_TEST_RAW_SHA256 = "8b489441bc915c9c5e4f7404e09a5fd9b236a2d516a82565af618c91fe028e0e"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class SevenGalaxyRobustnessError(RuntimeError):
    """Raised when an evidence, robustness, or package gate fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SevenGalaxyRobustnessError(message)


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
        raise SevenGalaxyRobustnessError(f"invalid {label}") from exc
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
        config["status"]
        == "FROZEN_POST_SCORE_SEVEN_GALAXY_2D_RG_ROBUSTNESS_AND_PUBLICATION_ASSESSMENT",
        "status changed",
    )
    _require(
        [row["role"] for row in config["input_bindings"]]
        == [
            "SIX_EXTERNAL_FIXED_2D_SCORE",
            "HOLMBERG_II_FIXED_2D_SCORE",
            "BOUNDED_PRIMARY_LITERATURE_AND_PUBLICATION_SYNTHESIS",
        ],
        "input inventory changed",
    )
    contract = config["robustness_contract"]
    _require(contract["external_primary_objects"] == 6, "external objects changed")
    _require(contract["combined_objects_including_holmberg_ii"] == 7, "combined objects changed")
    _require(
        contract["minimum_material_fractional_aggregate_improvement"] == 0.02,
        "materiality gate changed",
    )
    _require(contract["leave_one_object_out_reaggregations"] == 6, "jackknife count changed")
    for key in (
        "no_postscore_parameter_tuning",
        "no_object_or_cell_removal",
        "retain_every_failure_and_counterexample",
    ):
        _require(contract[key] is True, f"robustness protection removed: {key}")
    _require(contract["p_values_computed"] is False, "unregistered inference enabled")
    publication = config["publication_contract"]
    _require(publication["bounded_primary_rg_corpus_size"] == 6, "literature corpus changed")
    _require(
        publication["prior_2d_velocity_map_rg_test_found_in_bounded_corpus"] is False,
        "prior art changed",
    )
    _require(
        publication["prior_holmberg_ii_rg_test_found_in_bounded_corpus"] is False,
        "prior art changed",
    )
    claims = config["claim_boundary"]
    _require(claims["bounded_corpus_2d_method_novelty_supported"] is True, "method novelty removed")
    for key in (
        "postscore_robustness_completed",
        "universal_rg_replication",
        "ic2574_robust_object_replication",
        "inclination_crossover_generalized",
        "negative_or_constraint_result_publication_candidate",
        "unique_theory_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim promoted before analysis: {key}")
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


def _load_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for binding in config["input_bindings"]:
        receipt: dict[str, Any] | None = None
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), "bound artifact missing")
            _require(file_sha256(path) == artifact["sha256"], "bound artifact changed")
            if artifact["path"].endswith("receipt.json"):
                receipt = _read_json(path, "input receipt")
        _require(receipt is not None, "input receipt missing")
        _require(receipt["content_sha256"] == binding["receipt_content_sha256"], "receipt changed")
        values[binding["role"]] = receipt
    return values


def _leave_one_out(primary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for omitted in [row["object_id"] for row in primary]:
        retained = [row for row in primary if row["object_id"] != omitted]
        means = {
            candidate: float(np.mean([row["models"][candidate]["rmse_m_s"] for row in retained]))
            for candidate in _CANDIDATES
        }
        ranking = sorted(_CANDIDATES, key=lambda candidate: (means[candidate], candidate))
        output.append(
            {
                "omitted_object_id": omitted,
                "retained_object_count": 5,
                "equal_object_mean_rmse_m_s": means,
                "ranking": ranking,
                "rg_rank": ranking.index(_RG) + 1,
                "rg_fractional_improvement_over_newton": float(
                    (means["NEWTON_3D_DST"] - means[_RG]) / means["NEWTON_3D_DST"]
                ),
            }
        )
    return output


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = _load_inputs(config)
    external = inputs["SIX_EXTERNAL_FIXED_2D_SCORE"]
    holmberg = inputs["HOLMBERG_II_FIXED_2D_SCORE"]
    literature = inputs["BOUNDED_PRIMARY_LITERATURE_AND_PUBLICATION_SYNTHESIS"]
    primary = external["primary_adjudication"]["primary_cells"]
    _require(len(primary) == 6, "external primary inventory changed")
    loo = _leave_one_out(primary)
    aggregates = external["primary_adjudication"]["equal_object_mean_primary_rmse_m_s"]
    rg_newton_fraction = float(
        (aggregates["NEWTON_3D_DST"] - aggregates[_RG]) / aggregates["NEWTON_3D_DST"]
    )
    material = rg_newton_fraction >= float(
        config["robustness_contract"]["minimum_material_fractional_aggregate_improvement"]
    )
    object_rows: list[dict[str, Any]] = []
    for object_id in external["primary_adjudication"]["primary_object_winners"]:
        cells = [row for row in external["scores"] if row["object_id"] == object_id]
        primary_row = next(
            row
            for row in cells
            if row["cell_score_id"]
            == external["score_contract"]["primary_cell_by_object"][object_id]
        )
        rg_wins = [row for row in cells if row["winner"] == _RG]
        resolutions = sorted({row["resolution"] for row in rg_wins})
        conversions = sorted({row["conversion_cell_id"] for row in rg_wins})
        robust_object_signal = (
            primary_row["winner"] == _RG
            and resolutions == ["NATURAL", "ROBUST"]
            and len(conversions) == len({row["conversion_cell_id"] for row in cells})
        )
        object_rows.append(
            {
                "object_id": object_id,
                "inclination_deg": primary_row["inclination_deg"],
                "inclination_stratum": primary_row["inclination_stratum"],
                "primary_winner": primary_row["winner"],
                "primary_rg_beats_all_comparators": primary_row["rg_beats_all_three_comparators"],
                "sensitivity_cell_count": len(cells),
                "rg_winning_sensitivity_cells": len(rg_wins),
                "rg_winning_cell_ids": [row["cell_score_id"] for row in rg_wins],
                "rg_winning_resolutions": resolutions,
                "rg_winning_conversions": conversions,
                "robust_object_signal": robust_object_signal,
            }
        )
    ic2574 = next(row for row in object_rows if row["object_id"] == "IC2574")
    ngc6946 = next(row for row in object_rows if row["object_id"] == "NGC6946")
    holmberg_low = [row for row in holmberg["scores"] if float(row["inclination_deg"]) == 27.0]
    holmberg_low_rg_wins = sum(row["winner"] == _RG for row in holmberg_low)
    inclination_generalized = (
        holmberg_low_rg_wins > 0 and ngc6946["rg_winning_sensitivity_cells"] > 0
    )
    external_strong = bool(external["primary_adjudication"]["strong_external_replication"])
    publication_candidate = (
        literature["bounded_literature_audit"]["prior_2d_velocity_map_rg_test_identified"] is False
        and literature["bounded_literature_audit"]["prior_holmberg_ii_rg_test_identified"] is False
        and external["aggregate"]["all_cells_reported"] is True
        and len(primary) == 6
    )
    decision = (
        "PUBLICATION_CANDIDATE_BOUNDED_CORPUS_FIRST_2D_RG_STRESS_TEST_"
        "UNIVERSAL_REPLICATION_FAILS_IC2574_CONDITIONAL_SIGNAL_RETAINED"
    )
    claims = dict(config["claim_boundary"])
    claims["postscore_robustness_completed"] = True
    claims["universal_rg_replication"] = external_strong
    claims["ic2574_robust_object_replication"] = bool(ic2574["robust_object_signal"])
    claims["inclination_crossover_generalized"] = inclination_generalized
    claims["negative_or_constraint_result_publication_candidate"] = publication_candidate
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_SEVEN_GALAXY_POSTSCORE_ROBUSTNESS_AND_PUBLICATION_SCOPE",
        "decision": decision,
        "package_bindings": _package_bindings(),
        "input_receipt_content_sha256": {
            binding["role"]: binding["receipt_content_sha256"]
            for binding in config["input_bindings"]
        },
        "robustness_contract": config["robustness_contract"],
        "publication_contract": config["publication_contract"],
        "external_primary_result": {
            "strong_external_replication": external_strong,
            "rg_primary_object_wins": external["primary_adjudication"]["rg_primary_object_wins"],
            "equal_object_aggregate_ranking": external["primary_adjudication"][
                "equal_object_aggregate_ranking"
            ],
            "rg_fractional_improvement_over_newton": rg_newton_fraction,
            "material_two_percent_improvement_over_newton": material,
            "rg_wins_all_three_comparators_in_full_cells": external["aggregate"][
                "rg_all_comparator_win_cells"
            ],
            "full_cell_count": len(external["scores"]),
        },
        "object_robustness": object_rows,
        "leave_one_object_out": loo,
        "leave_one_out_summary": {
            "rg_rank_one_count": sum(row["rg_rank"] == 1 for row in loo),
            "rg_beats_newton_count": sum(
                row["rg_fractional_improvement_over_newton"] > 0.0 for row in loo
            ),
            "reaggregations": len(loo),
        },
        "inclination_crosscheck": {
            "holmberg_ii_i27_cells": len(holmberg_low),
            "holmberg_ii_i27_rg_wins": holmberg_low_rg_wins,
            "ngc6946_i32p6_rg_wins": ngc6946["rg_winning_sensitivity_cells"],
            "simple_low_inclination_crossover_generalized": inclination_generalized,
        },
        "ic2574_follow_up": {
            "primary_cell_winner": ic2574["primary_winner"],
            "rg_winning_sensitivity_cells": ic2574["rg_winning_sensitivity_cells"],
            "sensitivity_cell_count": ic2574["sensitivity_cell_count"],
            "winning_cell_ids": ic2574["rg_winning_cell_ids"],
            "robust_object_replication": ic2574["robust_object_signal"],
            "interpretation": "Targeted conditional anomaly only: RG wins three of four sensitivity cells but loses the preregistered natural fixed-mass primary cell.",
        },
        "counterexamples": [
            {
                "id": "NO_PRIMARY_EXTERNAL_RG_WIN",
                "fact": "RG wins zero of six preregistered external primary cells.",
            },
            {
                "id": "AGGREGATE_ADVANTAGE_NOT_MATERIAL",
                "fact": "RG's equal-object aggregate RMSE is only marginally below Newton and below the two-percent diagnostic materiality threshold.",
            },
            {
                "id": "LOW_INCLINATION_CROSSOVER_NOT_EXTERNAL",
                "fact": "NGC6946 at 32.6 degrees has zero RG-winning sensitivity cells despite the Holmberg II 27-degree pattern.",
            },
            {
                "id": "IC2574_SIGNAL_NOT_PRIMARY_OR_SOURCE_COMPLETE",
                "fact": "IC2574 contains the only external RG-winning cells, but its fixed-mass natural primary and unavailable FastICA source cell prevent a robust object-level claim.",
            },
        ],
        "claim_boundary": claims,
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
    validate_receipt(config, receipt)
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt) + b"\n")


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    receipt = _read_json(path, "receipt")
    validate_receipt(config, receipt)
    return "VALID"


def status() -> dict[str, Any]:
    config = load_config()
    if not _repo_path(OUTPUT_PATH).is_file():
        return {"package_id": config["package_id"], "status": "FROZEN_UNRUN"}
    receipt = _read_json(_repo_path(OUTPUT_PATH), "receipt")
    return {
        "package_id": config["package_id"],
        "status": receipt["status"],
        "decision": receipt["decision"],
        "universal_rg_replication": receipt["claim_boundary"]["universal_rg_replication"],
        "publication_candidate": receipt["claim_boundary"][
            "negative_or_constraint_result_publication_candidate"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    sub.add_parser("status")
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
