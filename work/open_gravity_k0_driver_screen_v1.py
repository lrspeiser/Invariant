from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler import gravity_cluster_comparator_suite as cluster_suite
from sigma_theory_compiler import gravity_extended_source_clock_xcop_development as clock
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59
from sigma_theory_compiler import open_gravity_campaign_v1 as campaign
from sigma_theory_compiler import open_gravity_static_radial_adapter_v1 as adapter

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "runs/gravity/twell-400-v2-typed-compiler-final-v3/cards.jsonl"
OUTPUT = ROOT / "work/open-gravity-k0-driver-screen-v1.json"

DEVELOPMENT = ["A1644", "A1795", "A2142", "A2255", "A2319", "A3266", "A85", "ZW1215"]
HOLDOUT = ["A2029", "A3158", "A644", "RXC1825"]
K0 = {
    "A1644": 19.0,
    "A1795": 19.0,
    "A2029": 10.5,
    "A2142": 68.1,
    "A2255": 529.1,
    "A2319": 270.2,
    "A3158": 166.0,
    "A3266": 72.5,
    "A644": 132.4,
    "A85": 12.5,
    "RXC1825": 217.9,
    "ZW1215": 163.2,
}
SCENARIOS = [
    {
        "cell_id": "XCOP-SOURCE-LOW",
        "density_scale": 0.95,
        "published_stellar_mass_scale": 0.8,
        "missing_stellar_to_gas_mass_ratio": 0.1,
        "outer_nonthermal_fraction": 0.1,
        "xray_temperature_cross_calibration": 0.95,
    },
    {
        "cell_id": "XCOP-SOURCE-NOMINAL",
        "density_scale": 1.0,
        "published_stellar_mass_scale": 1.0,
        "missing_stellar_to_gas_mass_ratio": 0.1,
        "outer_nonthermal_fraction": 0.15,
        "xray_temperature_cross_calibration": 1.0,
    },
    {
        "cell_id": "XCOP-SOURCE-HIGH",
        "density_scale": 1.05,
        "published_stellar_mass_scale": 1.2,
        "missing_stellar_to_gas_mass_ratio": 0.1,
        "outer_nonthermal_fraction": 0.2,
        "xray_temperature_cross_calibration": 1.05,
    },
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cards() -> list[dict[str, object]]:
    rows = [json.loads(line) for line in CARDS.read_text(encoding="utf-8").splitlines()]
    return [
        row
        for row in rows
        if row["entry_kind"] == "ATOMIC"
        and row["architecture_id"]
        in {
            "A01_LAPSE",
            "A02_CLOCK",
            "A03_CONFORMAL",
            "A04_DISFORMAL",
            "A05_SLIP",
            "A06_SPATIAL_KERNEL",
            "A07_BOUNDARY",
            "A08_PERMITTIVITY",
            "A09_ENTROPIC",
            "A10_DENSITY_SCREEN",
            "A11_DERIV_SCREEN",
            "A12_MASSIVE",
            "A13_MIXED_MODE",
            "A14_PHASE",
            "A19_FEEDBACK",
        }
        and row["driver_ids"] == ["D15_COOL"]
    ]


def _proxy(cluster: str, orientation: str) -> float:
    cooling_strength = 1.0 / (1.0 + K0[cluster] / 30.0)
    if orientation == "COOLING_STRENGTH":
        return cooling_strength
    if orientation == "ENTROPY_STRENGTH_REVERSAL":
        return 1.0 - cooling_strength
    raise ValueError(orientation)


def _score(
    packet: dict[str, object],
    scenario: dict[str, object],
    architecture: str,
    parameters: dict[str, object],
    orientation: str,
    item59_config: dict[str, object],
) -> tuple[float, float]:
    scaled, state, bundle, gbar = campaign._cluster_state_bundle(packet, scenario, item59_config)
    nuisance = campaign._cluster_nuisance(scenario)
    local_grid = campaign._gp01_l_factor(bundle, 1)
    local_factor = campaign._factor_on_radii(local_grid, bundle, state["radius_m"])
    u = np.full_like(
        np.asarray(bundle["xi"], dtype=float), _proxy(str(packet["cluster"]), orientation)
    )
    compiled = adapter.compile_static_architecture(
        architecture,
        bundle["xi"],
        u,
        bundle["physical"]["D01_ACC"],
        parameters,
    )
    modifier = campaign._factor_on_radii(compiled["primary"]["factor"], bundle, state["radius_m"])
    local_predictions = cluster_suite._predictions_from_acceleration(
        scaled, state, local_factor * gbar, nuisance, item59_config
    )
    candidate_predictions = cluster_suite._predictions_from_acceleration(
        scaled, state, modifier * local_factor * gbar, nuisance, item59_config
    )
    local_loss = campaign._loss_rows(
        local_predictions, scaled["rows"], minimum_fractional_error=0.05
    )["loss"]
    candidate_loss = campaign._loss_rows(
        candidate_predictions, scaled["rows"], minimum_fractional_error=0.05
    )["loss"]
    return float(candidate_loss), float(local_loss)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to replace {OUTPUT}")
    script_sha256 = _sha256(Path(__file__))
    item59_config = item59.load_config(ROOT)
    packets: dict[str, dict[str, object]] = {}
    for cluster in [*DEVELOPMENT, *HOLDOUT]:
        packet = item59._parse_cluster(ROOT, item59_config, cluster)
        clock._add_rows(packet, item59_config)
        packets[cluster] = packet

    scores = []
    for card in _cards():
        for cell in card["cell_results"]:
            for orientation in ("COOLING_STRENGTH", "ENTROPY_STRENGTH_REVERSAL"):
                object_rows = []
                for scenario in SCENARIOS:
                    for cluster, packet in packets.items():
                        candidate_loss, baseline_loss = _score(
                            packet,
                            scenario,
                            str(card["architecture_id"]),
                            dict(cell["parameters"]),
                            orientation,
                            item59_config,
                        )
                        object_rows.append(
                            {
                                "cluster": cluster,
                                "sample": "DEVELOPMENT" if cluster in DEVELOPMENT else "HOLDOUT",
                                "scenario_id": scenario["cell_id"],
                                "candidate_loss": candidate_loss,
                                "gp01l_n1_loss": baseline_loss,
                                "loss_ratio": candidate_loss / baseline_loss,
                            }
                        )
                development_ratios = [
                    row["loss_ratio"] for row in object_rows if row["sample"] == "DEVELOPMENT"
                ]
                holdout_ratios = [
                    row["loss_ratio"] for row in object_rows if row["sample"] == "HOLDOUT"
                ]
                scores.append(
                    {
                        "candidate_id": f"{card['concept_id']}::{cell['cell_id']}::{orientation}",
                        "concept_id": card["concept_id"],
                        "architecture_id": card["architecture_id"],
                        "cell_id": cell["cell_id"],
                        "parameters": cell["parameters"],
                        "orientation": orientation,
                        "development_mean_loss_ratio": float(np.mean(development_ratios)),
                        "development_worst_loss_ratio": float(np.max(development_ratios)),
                        "holdout_mean_loss_ratio": float(np.mean(holdout_ratios)),
                        "holdout_worst_loss_ratio": float(np.max(holdout_ratios)),
                        "development_beats_count": sum(value < 1.0 for value in development_ratios),
                        "holdout_beats_count": sum(value < 1.0 for value in holdout_ratios),
                        "object_scenario_rows": object_rows,
                    }
                )
    scores.sort(key=lambda row: (row["development_mean_loss_ratio"], row["candidate_id"]))
    selected = scores[0]
    result = {
        "schema_version": "open-gravity-k0-driver-screen-work-1.0",
        "status": "DEVELOPMENT_DISCOVERY_ONLY_NOT_CONFIRMATION",
        "script_sha256_before_scoring": script_sha256,
        "card_stream_sha256": _sha256(CARDS),
        "published_k0_source": "Ghirardini et al. 2019 X-COP Table 1",
        "proxy_definition": {
            "COOLING_STRENGTH": "1/(1+K0/30)",
            "ENTROPY_STRENGTH_REVERSAL": "1-1/(1+K0/30)",
            "warning": "These are external K0 proxies, not the original D15=tanh(t_cool/t_dyn) driver.",
        },
        "composition_rule": "g_candidate = GP01L_n1_factor * registered_time_architecture_modifier * g_b",
        "architectures": 15,
        "registered_cells": sum(len(card["cell_results"]) for card in _cards()),
        "orientation_count": 2,
        "candidate_count": len(scores),
        "development_clusters": DEVELOPMENT,
        "holdout_clusters": HOLDOUT,
        "selected_by_development_only": selected["candidate_id"],
        "selected_holdout_summary": {
            key: selected[key]
            for key in (
                "holdout_mean_loss_ratio",
                "holdout_worst_loss_ratio",
                "holdout_beats_count",
            )
        },
        "all_candidate_scores": scores,
        "claim_ceiling": {
            "publication_ready": False,
            "independent_confirmation": False,
            "original_D15_tested": False,
            "finding": "Response-independent K0 proxy composed on GP01L; discovery screen only; every candidate and failure retained.",
        },
    }
    result["content_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
