"""Explore the v6 spherical action and v5 mechanisms on CLASH lensing profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_g1_pilot import _file_sha256, _load_json, _metric
from .gravity_g4_auxiliary_action_derivation import (
    action_prediction2,
)
from .gravity_g4_auxiliary_action_derivation import (
    validate_receipt as validate_action_receipt,
)
from .gravity_g4_first_principles_mechanism_search import (
    _component_for_spec,
    _packet_context,
    mechanism_specs,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-g4-cluster-lensing-exploration-receipt-7.0"
CONFIG_SCHEMA = "invariant-gravity-g4-cluster-lensing-exploration-config-7.0"
CONFIG_PATH = "configs/gravity_g4_cluster_lensing_exploration.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g4_cluster_lensing_exploration.py"
TEST_PATH = "tests/test_gravity_g4_cluster_lensing_exploration.py"
OUTPUT_PATH = "runs/gravity/g4/cluster-lensing-exploration-v7.json"

KPC_M = 3.085677581491367e19
VELOCITY2_TO_ACCELERATION = 1.0e6 / KPC_M
G_DAGGER = 1.2e-10


class GravityG4ClusterLensingError(ValueError):
    """The exploratory CLASH contract, data, or calculation is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load the exploratory contract and verify all frozen source bindings."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG4ClusterLensingError("cluster-lensing config changed")
    if config.get("status") != "exploratory_post_source_discovery_not_confirmation":
        raise GravityG4ClusterLensingError("cluster-lensing status changed")
    predecessor_binding = config.get("predecessor_binding", {})
    predecessor_path = root / str(predecessor_binding.get("path"))
    if _file_sha256(predecessor_path) != predecessor_binding.get("file_sha256"):
        raise GravityG4ClusterLensingError("cluster-lensing predecessor file changed")
    predecessor = _load_json(predecessor_path)
    validate_action_receipt(predecessor, root=root)
    if predecessor.get("content_sha256") != predecessor_binding.get(
        "content_sha256"
    ) or predecessor.get("decision") != predecessor_binding.get("required_decision"):
        raise GravityG4ClusterLensingError("cluster-lensing predecessor content changed")
    mechanism_binding = config.get("mechanism_grammar_binding", {})
    if _file_sha256(root / str(mechanism_binding.get("path"))) != mechanism_binding.get(
        "file_sha256"
    ):
        raise GravityG4ClusterLensingError("cluster-lensing mechanism grammar changed")
    if int(mechanism_binding.get("declared_total_specs", 0)) != len(mechanism_specs()):
        raise GravityG4ClusterLensingError("cluster-lensing mechanism count changed")
    source = config.get("source", {})
    for binding in source.get("files", ()):
        if _file_sha256(root / str(binding.get("path"))) != binding.get("file_sha256"):
            raise GravityG4ClusterLensingError("cluster-lensing source file changed")
    action = config.get("fixed_action_transfer", {})
    if (
        int(action.get("support_dimension", 0)) != 3
        or not np.isclose(float(action.get("beta")), 1.0 / 3.0, rtol=0.0, atol=1e-15)
        or not np.isclose(float(action.get("log_radius_scale")), 1.0 / 6.0, rtol=0.0, atol=1e-15)
        or float(action.get("transition_acceleration_m_s2")) != G_DAGGER
        or float(action.get("transition_y")) != 0.1
        or int(action.get("parameters_fit_to_cluster_data", -1)) != 0
    ):
        raise GravityG4ClusterLensingError("cluster-lensing fixed action changed")
    parent = config.get("fixed_galaxy_parent_transfer", {})
    if (
        parent.get("candidate_id") != "cross-scale:y:q0p1:ell0p25:permittivity_plus_auxiliary"
        or float(parent.get("beta")) != 0.5
        or float(parent.get("log_radius_scale")) != 0.25
        or int(parent.get("parameters_fit_to_cluster_data", -1)) != 0
    ):
        raise GravityG4ClusterLensingError("cluster-lensing fixed galaxy parent changed")
    evaluation = config.get("whole_cluster_evaluation", {})
    if (
        int(evaluation.get("outer_folds", 0)) != 5
        or evaluation.get("object_identity_available_to_formula") is not False
        or evaluation.get("heldout_cluster_target_available_to_selection") is not False
        or evaluation.get("published_profile_covariance_available") is not False
        or evaluation.get("gbar_uncertainty_in_primary_score") is not False
    ):
        raise GravityG4ClusterLensingError("cluster-lensing evaluation changed")
    interpretation = config.get("interpretation", {})
    required_false = (
        "direct_lensing_falsification",
        "direct_cluster_thermodynamic_forward_test",
        "sequential_G6_G7_G8_authorized",
        "historical_novelty_claimed",
        "covariant_lensing_equation_derived",
        "diagnostic_can_confirm_a_gravity_theory",
    )
    if any(interpretation.get(key) is not False for key in required_false):
        raise GravityG4ClusterLensingError("cluster-lensing claim boundary changed")
    return config


def _parse_source(path: Path) -> list[dict[str, Any]]:
    """Parse the hash-bound VizieR TSV without accepting service metadata as rows."""

    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header_index = next(index for index, line in enumerate(lines) if line.startswith("recno\t"))
    except StopIteration as error:
        raise GravityG4ClusterLensingError("CLASH TSV header is missing") from error
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])), delimiter="\t")
    rows: list[dict[str, Any]] = []
    for raw in reader:
        record = str(raw.get("recno", "")).strip()
        if not record.isdigit():
            continue
        try:
            row = {
                "record": int(record),
                "cluster": str(raw["AName"]).strip(),
                "radius_kpc": float(str(raw["Rad"]).strip()),
                "log_gbar": float(str(raw["log(gbar)"]).strip()),
                "log_gtot": float(str(raw["log(gtot)"]).strip()),
                "sigma_log_gbar": float(str(raw["e_log(gbar)"]).strip()),
                "sigma_log_gtot": float(str(raw["e_log(gtot)"]).strip()),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise GravityG4ClusterLensingError("invalid CLASH source row") from error
        if (
            not row["cluster"]
            or row["radius_kpc"] <= 0
            or row["sigma_log_gbar"] <= 0
            or row["sigma_log_gtot"] <= 0
            or not all(np.isfinite(float(row[key])) for key in row if key not in {"cluster"})
        ):
            raise GravityG4ClusterLensingError("nonphysical CLASH source row")
        rows.append(row)
    if len({row["record"] for row in rows}) != len(rows):
        raise GravityG4ClusterLensingError("duplicate CLASH record number")
    return rows


def _galaxy_rar(gbar: np.ndarray) -> np.ndarray:
    root_y = np.sqrt(np.maximum(gbar / G_DAGGER, np.finfo(np.float64).tiny))
    return gbar / (-np.expm1(-root_y))


def prepare_packets(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create spherical packets using only radius and the published baryonic profile."""

    source_path = root / str(config["source"]["files"][0]["path"])
    rows = _parse_source(source_path)
    if len(rows) != int(config["source"]["expected_radial_points"]):
        raise GravityG4ClusterLensingError("CLASH radial point count changed")
    clusters = sorted({str(row["cluster"]) for row in rows})
    if len(clusters) != int(config["source"]["expected_clusters"]):
        raise GravityG4ClusterLensingError("CLASH cluster count changed")
    packets: list[dict[str, Any]] = []
    for cluster in clusters:
        selected = sorted(
            (row for row in rows if row["cluster"] == cluster),
            key=lambda row: (float(row["radius_kpc"]), int(row["record"])),
        )
        radius = np.asarray([row["radius_kpc"] for row in selected], dtype=np.float64)
        if np.any(np.diff(radius) <= 0):
            raise GravityG4ClusterLensingError(f"non-increasing radius for {cluster}")
        log_gbar = np.asarray([row["log_gbar"] for row in selected], dtype=np.float64)
        gbar = np.power(10.0, log_gbar)
        vbar2 = gbar * radius / VELOCITY2_TO_ACCELERATION
        rar = _galaxy_rar(gbar)
        packets.append(
            {
                "a0": G_DAGGER / VELOCITY2_TO_ACCELERATION,
                "arrays": {
                    "radius": radius,
                    "vbar2": vbar2,
                },
                "cluster": cluster,
                "features": {
                    "log1p_sb_total": np.zeros_like(radius),
                    "log_y": np.log(gbar / G_DAGGER),
                },
                "gbar": gbar,
                "log_gbar": log_gbar,
                "log_gtot": np.asarray([row["log_gtot"] for row in selected], dtype=np.float64),
                "rar2": rar * radius / VELOCITY2_TO_ACCELERATION,
                "records": [int(row["record"]) for row in selected],
                "sigma_log_gbar": np.asarray(
                    [row["sigma_log_gbar"] for row in selected], dtype=np.float64
                ),
                "sigma_log_gtot": np.asarray(
                    [row["sigma_log_gtot"] for row in selected], dtype=np.float64
                ),
            }
        )
    return packets


def _eligible_spec(spec: Mapping[str, Any]) -> bool:
    """Retain only mechanisms that do not require a stellar surface-density field."""

    return (
        spec.get("source") != "stellar_surface_density"
        and spec.get("lane") != "geometry_directed_gravity"
    )


def _materialize(
    packets: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    specs = [dict(spec) for spec in mechanism_specs() if _eligible_spec(spec)]
    parts: dict[str, list[np.ndarray]] = {str(spec["candidate_id"]): [] for spec in specs}
    for packet in packets:
        context = _packet_context(packet)
        for spec in specs:
            component = np.asarray(_component_for_spec(spec, packet, context), dtype=np.float64)
            if component.shape != np.asarray(packet["gbar"]).shape or np.any(
                ~np.isfinite(component)
            ):
                raise GravityG4ClusterLensingError("invalid transferred mechanism component")
            parts[str(spec["candidate_id"])].append(component)
    return specs, {candidate: np.concatenate(values) for candidate, values in parts.items()}


def _fold(cluster: str, salt: str, count: int) -> int:
    digest = hashlib.sha256(f"{salt}:{cluster}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % count


def _score(
    log_prediction: np.ndarray, target: np.ndarray, sigma: np.ndarray, mask: np.ndarray
) -> float:
    residual = (log_prediction[mask] - target[mask]) / sigma[mask]
    return float(np.sum(residual**2))


def _summary(
    prediction: np.ndarray,
    target_log: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(prediction), dtype=bool)
    if np.any(prediction[mask] <= 0) or np.any(~np.isfinite(prediction[mask])):
        raise GravityG4ClusterLensingError("cannot score nonpositive acceleration")
    log_prediction = np.log10(prediction)
    residual_dex = log_prediction[mask] - target_log[mask]
    normalized = residual_dex / sigma[mask]
    return {
        "chi_square": _metric(float(np.sum(normalized**2))),
        "mean_residual_dex": _metric(float(np.mean(residual_dex))),
        "median_absolute_residual_dex": _metric(float(np.median(np.abs(residual_dex)))),
        "points": int(np.sum(mask)),
        "rmse_dex": _metric(float(np.sqrt(np.mean(residual_dex**2)))),
        "within_1sigma": int(np.sum(np.abs(normalized) <= 1.0)),
        "within_2sigma": int(np.sum(np.abs(normalized) <= 2.0)),
    }


def _select_cell(
    specs: Sequence[Mapping[str, Any]],
    components: Mapping[str, np.ndarray],
    vbar2: np.ndarray,
    radius: np.ndarray,
    target_log: np.ndarray,
    sigma: np.ndarray,
    coefficient_grid: Sequence[float],
    training: np.ndarray,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for spec in specs:
        if spec["role"] != "mechanism":
            continue
        component = components[str(spec["candidate_id"])]
        for beta in coefficient_grid:
            prediction2 = vbar2 + beta * component
            if np.any(prediction2 <= 0) or np.any(~np.isfinite(prediction2)):
                continue
            acceleration = prediction2 / radius * VELOCITY2_TO_ACCELERATION
            chi_square = _score(np.log10(acceleration), target_log, sigma, training)
            cells.append(
                {
                    "beta": float(beta),
                    "candidate_id": str(spec["candidate_id"]),
                    "chi_square": chi_square,
                    "lane": str(spec["lane"]),
                    "origin_label": str(spec["origin_label"]),
                }
            )
    if not cells:
        raise GravityG4ClusterLensingError("no valid transferred mechanism cell")
    best = min(
        cells,
        key=lambda row: (
            float(row["chi_square"]),
            abs(float(row["beta"])),
            str(row["candidate_id"]),
        ),
    )
    return {
        "beta": _metric(float(best["beta"])),
        "candidate_id": best["candidate_id"],
        "chi_square": _metric(float(best["chi_square"])),
        "lane": best["lane"],
        "origin_label": best["origin_label"],
    }


def _cell_prediction(
    cell: Mapping[str, Any],
    components: Mapping[str, np.ndarray],
    vbar2: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    prediction2 = vbar2 + float(cell["beta"]) * components[str(cell["candidate_id"])]
    if np.any(prediction2 <= 0) or np.any(~np.isfinite(prediction2)):
        raise GravityG4ClusterLensingError("selected mechanism became nonpositive")
    return prediction2 / radius * VELOCITY2_TO_ACCELERATION


def build_receipt(root: Path) -> dict[str, Any]:
    """Run the fixed-action diagnostic and nested whole-cluster transfer search."""

    root = root.resolve()
    config = load_config(root)
    packets = prepare_packets(root, config)
    specs, components = _materialize(packets)
    cluster = np.concatenate(
        [np.repeat(str(packet["cluster"]), len(packet["gbar"])) for packet in packets]
    )
    radius = np.concatenate([np.asarray(packet["arrays"]["radius"]) for packet in packets])
    gbar = np.concatenate([np.asarray(packet["gbar"]) for packet in packets])
    vbar2 = np.concatenate([np.asarray(packet["arrays"]["vbar2"]) for packet in packets])
    target_log = np.concatenate([np.asarray(packet["log_gtot"]) for packet in packets])
    sigma = np.concatenate([np.asarray(packet["sigma_log_gtot"]) for packet in packets])
    records = [record for packet in packets for record in packet["records"]]
    if len(set(records)) != len(records):
        raise GravityG4ClusterLensingError("CLASH record assembly changed")

    action_prediction = np.concatenate(
        [
            np.asarray(action_prediction2(packet, support_dimension=3)["prediction2"])
            / np.asarray(packet["arrays"]["radius"])
            * VELOCITY2_TO_ACCELERATION
            for packet in packets
        ]
    )
    galaxy_rar = _galaxy_rar(gbar)
    cluster_relation = np.sqrt(2.02e-9 * gbar)
    controls = {
        "newtonian_baryons": _summary(gbar, target_log, sigma),
        "galaxy_rar": _summary(galaxy_rar, target_log, sigma),
        "published_cluster_low_acceleration_relation": _summary(
            cluster_relation, target_log, sigma
        ),
    }
    action_summary = _summary(action_prediction, target_log, sigma)
    action_summary.update(
        {
            "amplification_observed_over_prediction_median": _metric(
                float(np.median(np.power(10.0, target_log) / action_prediction))
            ),
            "parameters_fit_to_cluster_data": 0,
            "prediction_manifest_sha256": canonical_sha256(
                [format(float(value), ".15e") for value in action_prediction]
            ),
            "support_dimension": 3,
        }
    )
    parent_contract = config["fixed_galaxy_parent_transfer"]
    parent_cell = {
        "beta": str(parent_contract["beta"]),
        "candidate_id": str(parent_contract["candidate_id"]),
    }
    parent_prediction = _cell_prediction(parent_cell, components, vbar2, radius)
    parent_summary = _summary(parent_prediction, target_log, sigma)
    parent_summary.update(
        {
            "beta": str(parent_contract["beta"]),
            "candidate_id": str(parent_contract["candidate_id"]),
            "parameters_fit_to_cluster_data": 0,
            "prediction_manifest_sha256": canonical_sha256(
                [format(float(value), ".15e") for value in parent_prediction]
            ),
        }
    )

    evaluation = config["whole_cluster_evaluation"]
    folds = int(evaluation["outer_folds"])
    salt = str(evaluation["fold_salt"])
    fold_by_cluster = {name: _fold(name, salt, folds) for name in sorted(set(cluster.tolist()))}
    fold_counts = Counter(fold_by_cluster.values())
    if set(fold_counts) != set(range(folds)):
        raise GravityG4ClusterLensingError("empty CLASH outer fold")
    coefficient_grid = [float(value) for value in evaluation["coefficient_grid"]]
    oof_prediction = np.full(len(cluster), np.nan, dtype=np.float64)
    outer_ledger: list[dict[str, Any]] = []
    for fold_index in range(folds):
        heldout_clusters = sorted(
            name for name, assigned in fold_by_cluster.items() if assigned == fold_index
        )
        heldout = np.isin(cluster, heldout_clusters)
        training = ~heldout
        cell = _select_cell(
            specs,
            components,
            vbar2,
            radius,
            target_log,
            sigma,
            coefficient_grid,
            training,
        )
        prediction = _cell_prediction(cell, components, vbar2, radius)
        oof_prediction[heldout] = prediction[heldout]
        outer_ledger.append(
            {
                "fold": fold_index,
                "heldout_clusters": heldout_clusters,
                "heldout_score": _summary(prediction, target_log, sigma, heldout),
                "selected_cell": cell,
                "training_clusters": len(set(cluster[training].tolist())),
                "training_points": int(np.sum(training)),
            }
        )
    if np.any(~np.isfinite(oof_prediction)):
        raise GravityG4ClusterLensingError("incomplete whole-cluster predictions")
    full_mask = np.ones(len(cluster), dtype=bool)
    final_cell = _select_cell(
        specs,
        components,
        vbar2,
        radius,
        target_log,
        sigma,
        coefficient_grid,
        full_mask,
    )
    selected_pairs = {
        (str(row["selected_cell"]["candidate_id"]), str(row["selected_cell"]["beta"]))
        for row in outer_ledger
    }
    exact_control = next(
        spec
        for spec in specs
        if spec["candidate_id"] == "known-control:exact-empirical-rar-rewrite"
    )
    exact_prediction = (
        (vbar2 + components[str(exact_control["candidate_id"])])
        / radius
        * VELOCITY2_TO_ACCELERATION
    )
    exact_error = float(np.max(np.abs(exact_prediction - galaxy_rar)))
    per_cluster = []
    for name in sorted(set(cluster.tolist())):
        mask = cluster == name
        per_cluster.append(
            {
                "cluster": name,
                "fixed_action": _summary(action_prediction, target_log, sigma, mask),
                "galaxy_rar": _summary(galaxy_rar, target_log, sigma, mask),
                "oof_transferred_mechanism": _summary(oof_prediction, target_log, sigma, mask),
                "points": int(np.sum(mask)),
            }
        )
    lane_counts = Counter(str(spec["lane"]) for spec in specs if spec["role"] == "mechanism")
    creative_specs = sum(spec["role"] == "mechanism" for spec in specs)
    cell_count = creative_specs * len(coefficient_grid)
    direct_gate_open = bool(config["interpretation"]["direct_lensing_falsification"])
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G4_CLUSTER_LENSING_EXPLORATION",
        "decision": "BLOCK_CROSS_SCALE_ACTION_CLUSTER_LENSING_EXPLORATION",
        "claims": {
            "alternative_to_gr_confirmed": False,
            "covariant_lensing_equation_derived": False,
            "direct_cluster_thermodynamic_test_completed": False,
            "direct_lensing_test_completed": False,
            "fixed_D3_projection_diagnostically_tested": True,
            "historical_novelty_established": False,
            "sequential_G6_G7_G8_advanced": False,
            "whole_cluster_transfer_explored": True,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "controls": controls,
        "counts": {
            "clusters": len(set(cluster.tolist())),
            "coefficient_cells_per_selection": cell_count,
            "creative_mechanism_specs": creative_specs,
            "direct_image_or_shear_likelihood_evaluations": 0,
            "eligible_mechanism_lanes": len(lane_counts),
            "known_control_specs": sum(spec["role"] != "mechanism" for spec in specs),
            "outer_folds": folds,
            "radial_points": len(cluster),
            "scoring_point_evaluations": cell_count * len(cluster) * (folds + 1),
            "source_compatible_specs_total": len(specs),
        },
        "data_lineage": {
            **config["source"]["observable_lineage"],
            "interpretation": (
                "model-dependent acceleration diagnostic reconstructed from lensing; "
                "not a direct shear, image, arc, or time-delay likelihood"
            ),
        },
        "fixed_D3_action": action_summary,
        "gate_checks": {
            "covariant_same_field_lensing_equation": False,
            "direct_lensing_observable_likelihood": direct_gate_open,
            "fixed_action_beats_galaxy_rar_diagnostic": (
                float(action_summary["chi_square"]) < float(controls["galaxy_rar"]["chi_square"])
            ),
            "fixed_action_beats_newtonian_baryons_diagnostic": (
                float(action_summary["chi_square"])
                < float(controls["newtonian_baryons"]["chi_square"])
            ),
            "fixed_galaxy_parent_beats_galaxy_rar_diagnostic": (
                float(parent_summary["chi_square"]) < float(controls["galaxy_rar"]["chi_square"])
            ),
            "known_RAR_control_recovered_exactly": exact_error <= 1e-20,
            "same_coefficient_as_fixed_galaxy_parent": (
                all(float(row["selected_cell"]["beta"]) == 0.5 for row in outer_ledger)
            ),
            "same_structure_as_fixed_galaxy_parent_in_all_outer_folds": (
                all(
                    row["selected_cell"]["candidate_id"] == parent_contract["candidate_id"]
                    for row in outer_ledger
                )
            ),
            "same_transferred_mechanism_selected_in_all_outer_folds": len(selected_pairs) == 1,
            "whole_cluster_transfer_beats_galaxy_rar_diagnostic": (
                float(_summary(oof_prediction, target_log, sigma)["chi_square"])
                < float(controls["galaxy_rar"]["chi_square"])
            ),
        },
        "mechanism_transfer": {
            "cluster_selected_to_galaxy_parent_beta_ratio": _metric(
                float(final_cell["beta"]) / float(parent_contract["beta"])
            ),
            "final_all_data_selection_not_generalization_evidence": final_cell,
            "fixed_galaxy_parent_cluster_diagnostic": parent_summary,
            "fold_assignment": fold_by_cluster,
            "known_control_max_absolute_acceleration_error": _metric(exact_error),
            "lane_counts": dict(sorted(lane_counts.items())),
            "nested_oof_prediction_manifest_sha256": canonical_sha256(
                [format(float(value), ".15e") for value in oof_prediction]
            ),
            "nested_oof_score": _summary(oof_prediction, target_log, sigma),
            "outer_ledger": outer_ledger,
            "stable_outer_selection": len(selected_pairs) == 1,
        },
        "per_cluster_diagnostics": per_cluster,
        "source_bindings": config["source"],
        "limitations": [
            "The published gtot values are derived from spherical NFW posteriors fitted to CLASH lensing constraints, not direct lensing observables.",
            "The baryonic profiles include empirical stellar corrections and the source table does not publish full radial covariance.",
            "The fixed action has no covariant metric completion, so the no-slip lensing mapping is an explicit diagnostic assumption rather than a derivation.",
            "The mechanism search is exploratory and source-compatible only; its all-data winner is not confirmation evidence.",
            "X-COP direct thermodynamic profile forward modeling and CLASH image/shear likelihoods remain separate future tests.",
        ],
        "reproducibility": {
            "data_manifest_sha256": canonical_sha256(
                [
                    {
                        "cluster": packet["cluster"],
                        "records": packet["records"],
                        "radius_kpc": [
                            format(float(value), ".15e") for value in packet["arrays"]["radius"]
                        ],
                    }
                    for packet in packets
                ]
            ),
            "source_file": {
                "file_sha256": _file_sha256(root / SOURCE_PATH),
                "path": SOURCE_PATH,
            },
            "test_file": {
                "file_sha256": _file_sha256(root / TEST_PATH),
                "path": TEST_PATH,
            },
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Fail closed if the exploratory evidence or its claim boundary changes."""

    config = load_config(root)
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG4ClusterLensingError("cluster-lensing receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != canonical_sha256(body):
        raise GravityG4ClusterLensingError("cluster-lensing receipt seal changed")
    if receipt.get("decision") != "BLOCK_CROSS_SCALE_ACTION_CLUSTER_LENSING_EXPLORATION":
        raise GravityG4ClusterLensingError("cluster-lensing decision overstates evidence")
    if receipt.get("config") != {
        "content_sha256": canonical_sha256(config),
        "path": CONFIG_PATH,
    }:
        raise GravityG4ClusterLensingError("cluster-lensing config binding changed")
    claims = receipt.get("claims", {})
    forbidden_true = (
        "alternative_to_gr_confirmed",
        "covariant_lensing_equation_derived",
        "direct_cluster_thermodynamic_test_completed",
        "direct_lensing_test_completed",
        "historical_novelty_established",
        "sequential_G6_G7_G8_advanced",
    )
    if any(claims.get(key) is not False for key in forbidden_true):
        raise GravityG4ClusterLensingError("cluster-lensing claim boundary violated")
    counts = receipt.get("counts", {})
    if (
        counts.get("clusters") != 20
        or counts.get("radial_points") != 84
        or counts.get("creative_mechanism_specs") != 184
        or counts.get("known_control_specs") != 1
        or counts.get("source_compatible_specs_total") != 185
        or counts.get("eligible_mechanism_lanes") != 9
        or counts.get("direct_image_or_shear_likelihood_evaluations") != 0
    ):
        raise GravityG4ClusterLensingError("cluster-lensing accounting changed")
    if receipt.get("source_bindings") != config.get("source"):
        raise GravityG4ClusterLensingError("cluster-lensing source binding changed")
    reproducibility = receipt.get("reproducibility", {})
    for key, path in (("source_file", SOURCE_PATH), ("test_file", TEST_PATH)):
        if reproducibility.get(key) != {
            "file_sha256": _file_sha256(root / path),
            "path": path,
        }:
            raise GravityG4ClusterLensingError("cluster-lensing code binding changed")
    if receipt.get("gate_checks", {}).get("direct_lensing_observable_likelihood") is not False:
        raise GravityG4ClusterLensingError("cluster-lensing direct gate opened")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != encoded:
            raise GravityG4ClusterLensingError(
                f"refusing to overwrite immutable cluster-lensing receipt: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    receipt = build_receipt(root)
    validate_receipt(receipt, root=root)
    if args.write:
        output = args.output or (root / OUTPUT_PATH)
        if not output.is_absolute():
            output = root / output
        _write_immutable(output, receipt)
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "OUTPUT_PATH",
    "SCHEMA",
    "GravityG4ClusterLensingError",
    "build_receipt",
    "load_config",
    "main",
    "prepare_packets",
    "validate_receipt",
]
