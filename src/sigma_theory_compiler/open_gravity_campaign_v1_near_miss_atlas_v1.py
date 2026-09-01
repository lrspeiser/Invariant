"""Build a response-free near-miss atlas from the sealed open-gravity campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_campaign_v1_near_miss_atlas_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_campaign_v1_near_miss_atlas_v1.py")
TEST_PATH = Path("tests/test_open_gravity_campaign_v1_near_miss_atlas_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-campaign-v1-near-miss-atlas-v1/receipt.json")
ARTIFACT_DIRECTORY = Path("runs/gravity/open-gravity-campaign-v1-near-miss-atlas-v1/artifacts")

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-campaign-near-miss-atlas-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-campaign-near-miss-atlas-receipt-1.0"
_CONFIG_RAW_SHA256 = "9aae775e29afa05bb797b7d22de86e1586af56dfe8c9f65983964adafc8dd2d5"
_CONFIG_CONTENT_SHA256 = "20ef45a45f1611e4884c66f455ad70b6927c8dcaa7be79fe1f839d5bf33e9702"
_MODULE_SEMANTIC_SHA256 = "763cd27b16f4750e67ee0bc2e1fe117215e37eea9866d35a91cef7ada8e5a408"
_TEST_RAW_SHA256 = "9fde51d6dbbac2140ddc70faa8f3664176e8eb5a229ca308bb58f7db959a77be"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')

_COMPARATOR_FIELDS = {
    "BARYON": "BARYON_ONLY",
    "RAR": "EMPIRICAL_RAR",
    "MOND_N1": "ALGEBRAIC_MOND_GP01_L_n1",
    "CLOCK": "EXTENDED_SOURCE_CLOCK",
    "NFW": "GR_PLUS_NFW_CONTEXTUAL_CEILING",
    "EINASTO": "GR_PLUS_EINASTO_CONTEXTUAL_CEILING",
    "PREVIOUS_CROSS_SCALE": "PREVIOUS_CROSS_SCALE",
}


class NearMissAtlasError(RuntimeError):
    """Raised when an atlas source or deterministic evidence gate fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NearMissAtlasError(message)


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


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NearMissAtlasError(f"invalid {label}") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_POSTRUN_RESPONSE_FREE_NEAR_MISS_REAGGREGATION", "status changed"
    )
    expected_roles = [
        "SUCCESSOR_CONFIG",
        "SUCCESSOR_MODULE",
        "SUCCESSOR_TEST",
        "SUCCESSOR_PREFLIGHT",
        "SUCCESSOR_RESULT",
        "SUCCESSOR_ADJUDICATION",
        "POSTRUN_STABILITY_ADJUDICATION",
        "GLOBAL_CELL_LEDGER",
        "COMPARATOR_LEDGER",
        "BLOCKED_IDEA_LEDGER",
        "CLOSURE_MATRIX",
        "LAY_SUMMARY",
        "TWELL_LIVE_CARD_STREAM",
    ]
    _require([row["role"] for row in config["input_bindings"]] == expected_roles, "inputs changed")
    for row in config["input_bindings"]:
        _require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "input seal invalid")
        suffix = Path(row["path"]).suffix.lower()
        _require(suffix in {".json", ".jsonl", ".py"}, "non-metadata input admitted")
    analysis = config["analysis_contract"]
    expected_counts = {
        "registered_live_candidates": 407,
        "registered_parameter_cells": 2486,
        "galaxy_cells_planned": 179,
        "cluster_cells_planned": 1669,
        "galaxy_cells_valid": 171,
        "cluster_cells_valid": 1651,
        "galaxy_objects": 139,
        "cluster_objects": 8,
        "blocked_or_unscored_ideas": 279,
        "domain_concept_rows": 189,
    }
    for key, value in expected_counts.items():
        _require(analysis[key] == value, f"analysis count changed: {key}")
    _require(analysis["strict_threshold_fraction"] == 0.02, "threshold changed")
    for key in ("postrun_formula_retuning", "new_response_scoring", "raw_response_payloads_opened"):
        _require(analysis[key] == 0, f"postrun access enabled: {key}")
    artifacts = config["artifact_contract"]
    _require(artifacts["directory"] == ARTIFACT_DIRECTORY.as_posix(), "artifact directory changed")
    _require(
        [(row["path"], row["rows"]) for row in artifacts["artifacts"]]
        == [
            ("cell-comparator-atlas.csv", 1848),
            ("concept-domain-atlas.csv", 189),
            ("comparator-ladder.csv", 15),
            ("blocked-and-unscored-ideas.csv", 285),
            ("near-miss-summary.md", 1),
            ("figure-near-miss-ladder.svg", 15),
        ],
        "artifact inventory changed",
    )
    claims = config["claim_boundary"]
    for key, value in claims.items():
        _require(value is False, f"claim promoted in frozen config: {key}")
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
    _require(type(config) is dict, "config must be object")
    validate_config(config)
    if verify_package:
        _validate_package()
    return config


def _load_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    values = {}
    for binding in config["input_bindings"]:
        path = _repo_path(binding["path"])
        _require(path.is_file(), f"input missing: {binding['role']}")
        _require(file_sha256(path) == binding["sha256"], f"input changed: {binding['role']}")
        if path.suffix.lower() == ".json":
            values[binding["role"]] = _read_json(path, binding["role"])
        elif path.suffix.lower() == ".jsonl":
            rows = []
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        rows.append(json.loads(line))
            except (OSError, json.JSONDecodeError) as exc:
                raise NearMissAtlasError(f"invalid {binding['role']}") from exc
            values[binding["role"]] = rows
    result = values["SUCCESSOR_RESULT"]
    adjudication = values["SUCCESSOR_ADJUDICATION"]
    postrun = values["POSTRUN_STABILITY_ADJUDICATION"]
    _require(
        result["status"] == "DEVELOPMENT_CAMPAIGN_CONTINUATION_COMPLETE", "result status changed"
    )
    _require(adjudication["survivor_count"] == 0, "survivor count changed")
    _require(
        postrun["status"] == "PASS_STABLE_REAGGREGATION_ZERO_SURVIVORS", "postrun gate changed"
    )
    _require(result["counts"]["live_candidates"] == 407, "candidate count changed")
    _require(result["counts"]["parameter_cells"] == 2486, "cell count changed")
    return values


def _card_descriptions(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    descriptions = {}
    for row in rows:
        concept = row["concept_id"]
        descriptions[concept] = {
            "lay_mechanism": row["card"]["lay_mechanism"],
            "architecture_id": row["architecture_id"],
            "driver_ids": "+".join(row["driver_ids"]),
            "entry_kind": row["entry_kind"],
        }
    return descriptions


def _comparator_map(ledger: Mapping[str, Any], domain: str) -> dict[tuple[str, str], float]:
    output = {}
    for row in ledger[domain]["scenario_results"]:
        key = (row["comparator_id"], row["scenario_id"])
        _require(key not in output, "duplicate comparator scenario")
        output[key] = float(row["mean_loss"])
    return output


def _fractional_rows(
    scenario_results: Sequence[Mapping[str, Any]],
    comparator: str,
    comparator_map: Mapping[tuple[str, str], float],
) -> list[float]:
    fractions = []
    for scenario in scenario_results:
        key = (comparator, scenario["scenario_id"])
        if key not in comparator_map:
            return []
        baseline = comparator_map[key]
        _require(baseline > 0.0, "nonpositive comparator loss")
        fractions.append((baseline - float(scenario["mean_loss"])) / baseline)
    return fractions


def _outcome(fractions: Sequence[float], threshold: float) -> str:
    if not fractions:
        return "NOT_APPLICABLE"
    if min(fractions) > threshold:
        return "ROBUST_2PCT_WIN"
    if min(fractions) > 0.0:
        return "ROBUST_WIN_UNDER_2PCT"
    if max(fractions) > 0.0:
        return "NUISANCE_CONDITIONAL_WIN"
    if max(abs(value) for value in fractions) <= 1e-12:
        return "PREDICTION_EQUIVALENT_TIE"
    return "LOSS_ALL_SCENARIOS"


def _cell_rows(inputs: Mapping[str, Any], threshold: float) -> list[dict[str, Any]]:
    global_ledger = inputs["GLOBAL_CELL_LEDGER"]
    comparator_ledger = inputs["COMPARATOR_LEDGER"]
    descriptions = _card_descriptions(inputs["TWELL_LIVE_CARD_STREAM"])
    output = []
    for domain, cell_key, adjudication_key in (
        ("GALAXIES", "galaxies", "galaxy_adjudication"),
        ("CLUSTERS", "clusters", "cluster_adjudication"),
    ):
        comparators = _comparator_map(comparator_ledger, domain)
        adjudications = {row["cell_id"]: row for row in global_ledger[adjudication_key]}
        _require(len(adjudications) == len(global_ledger[cell_key]), "adjudication join changed")
        for cell in global_ledger[cell_key]:
            adjudication = adjudications[cell["cell_id"]]
            valid = not cell["gate_failures"] and len(cell["scenario_results"]) == 3
            comparisons = {
                label: _fractional_rows(cell["scenario_results"], comparator, comparators)
                if valid
                else []
                for label, comparator in _COMPARATOR_FIELDS.items()
            }
            strong = [
                float(row["fractional_improvement"]) for row in adjudication["scenario_evidence"]
            ]
            if not valid:
                category = "INVALID_SOURCE_GATE"
            elif adjudication["passes"]:
                category = "STRICT_DOMAIN_SURVIVOR"
            elif _outcome(comparisons["RAR"], threshold) in {
                "ROBUST_2PCT_WIN",
                "ROBUST_WIN_UNDER_2PCT",
            }:
                category = "ROBUST_OVER_RAR_ONLY"
            elif _outcome(comparisons["RAR"], threshold) == "NUISANCE_CONDITIONAL_WIN":
                category = "NUISANCE_CONDITIONAL_OVER_RAR"
            elif _outcome(comparisons["BARYON"], threshold) in {
                "ROBUST_2PCT_WIN",
                "ROBUST_WIN_UNDER_2PCT",
            }:
                category = "ROBUST_OVER_BARYON_ONLY"
            else:
                category = "NO_BASELINE_ADVANCE"
            description = descriptions.get(
                cell["concept_id"],
                {
                    "lay_mechanism": "GP01 local or elliptic gain branch; see the bound GP01 package.",
                    "architecture_id": "GP01",
                    "driver_ids": "GP01",
                    "entry_kind": "GP01_BRANCH",
                },
            )
            row = {
                "domain": domain,
                "anonymous_formula_id": cell["anonymous_formula_id"],
                "concept_id": cell["concept_id"],
                "cell_id": cell["cell_id"],
                "lane": cell["lane"],
                "architecture_id": description["architecture_id"],
                "driver_ids": description["driver_ids"],
                "entry_kind": description["entry_kind"],
                "lay_mechanism": description["lay_mechanism"],
                "valid": valid,
                "source_gate_failure_count": int(cell["gate_failure_count"]),
                "category": category,
                "strict_domain_pass": bool(adjudication["passes"]),
                "support_count": int(adjudication["support_count"]),
                "statistical_gate_pass_count": sum(
                    bool(value) for value in adjudication["gates"].values()
                ),
                "robust_loss": float(cell["robust_loss"]) if valid else None,
                "strongest_worst_fractional_improvement": min(strong) if strong else None,
                "strongest_best_fractional_improvement": max(strong) if strong else None,
            }
            for label, fractions in comparisons.items():
                key = label.lower()
                row[f"{key}_outcome"] = _outcome(fractions, threshold)
                row[f"{key}_worst_fractional_improvement"] = min(fractions) if fractions else None
                row[f"{key}_best_fractional_improvement"] = max(fractions) if fractions else None
                row[f"{key}_winning_scenarios"] = sum(value > 0.0 for value in fractions)
            output.append(row)
    _require(len(output) == 1848, "cell atlas row count changed")
    return output


def _concept_rows(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in cells:
        groups[(row["domain"], row["concept_id"])].append(row)
    output = []
    for (domain, concept), rows in sorted(groups.items()):
        valid = [row for row in rows if row["valid"]]
        best = max(
            valid,
            key=lambda row: row["strongest_worst_fractional_improvement"],
            default=None,
        )
        output.append(
            {
                "domain": domain,
                "concept_id": concept,
                "lane": rows[0]["lane"],
                "architecture_id": rows[0]["architecture_id"],
                "driver_ids": rows[0]["driver_ids"],
                "lay_mechanism": rows[0]["lay_mechanism"],
                "planned_cells": len(rows),
                "valid_cells": len(valid),
                "invalid_cells": len(rows) - len(valid),
                "best_cell_id": best["cell_id"] if best else "",
                "best_strongest_worst_fractional_improvement": (
                    best["strongest_worst_fractional_improvement"] if best else None
                ),
                "strict_domain_survivors": sum(row["strict_domain_pass"] for row in rows),
                "rar_conditional_cells": sum(
                    row["rar_outcome"] == "NUISANCE_CONDITIONAL_WIN" for row in valid
                ),
                "rar_robust_cells": sum(row["rar_outcome"].startswith("ROBUST_") for row in valid),
                "baryon_robust_cells": sum(
                    row["baryon_outcome"].startswith("ROBUST_") for row in valid
                ),
                "dominant_category": Counter(row["category"] for row in rows).most_common(1)[0][0],
            }
        )
    _require(len(output) == 189, "concept-domain row count changed")
    return output


def _comparator_ladder(
    inputs: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    ledger = inputs["COMPARATOR_LEDGER"]
    output = []
    for domain in ("GALAXIES", "CLUSTERS"):
        domain_cells = [row for row in cells if row["domain"] == domain and row["valid"]]
        comparators = _comparator_map(ledger, domain)
        ids = sorted({key[0] for key in comparators})
        source_rows = {row["cell_id"]: row for row in inputs["GLOBAL_CELL_LEDGER"][domain.lower()]}
        for comparator in ids:
            losses = [loss for (cid, _), loss in comparators.items() if cid == comparator]
            any_win = all_win = all_two_percent = 0
            best_worst = -math.inf
            best_cell = ""
            for atlas_row in domain_cells:
                source = source_rows[atlas_row["cell_id"]]
                fractions = _fractional_rows(source["scenario_results"], comparator, comparators)
                any_win += max(fractions) > 0.0
                all_win += min(fractions) > 0.0
                all_two_percent += min(fractions) > 0.02
                if min(fractions) > best_worst:
                    best_worst = min(fractions)
                    best_cell = atlas_row["cell_id"]
            output.append(
                {
                    "domain": domain,
                    "comparator_id": comparator,
                    "scenario_count": len(losses),
                    "mean_of_scenario_mean_losses": sum(losses) / len(losses),
                    "candidate_cells_with_any_scenario_win": any_win,
                    "candidate_cells_winning_all_scenarios": all_win,
                    "candidate_cells_winning_all_by_two_percent": all_two_percent,
                    "best_candidate_cell": best_cell,
                    "best_candidate_worst_fractional_improvement": best_worst,
                }
            )
    _require(len(output) == 15, "comparator ladder row count changed")
    return output


def _blocked_rows(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in inputs["BLOCKED_IDEA_LEDGER"]:
        output.append(
            {
                "source": "CAMPAIGN_BLOCKED_IDEA_LEDGER",
                "concept_id": row["concept_id"],
                "candidate_status": row["candidate_status"],
                "lane": row["lane"],
                "galaxies": row["domains"]["GALAXIES"],
                "clusters": row["domains"]["CLUSTERS"],
                "groups": row["domains"]["GROUPS"],
                "lensing": row["domains"]["LENSING"],
                "physical_time_dilation_derived": row["physical_time_dilation_derived"],
                "light_propagation_derived": row["light_propagation_derived"],
                "redshift_closure_derived": row["redshift_closure_derived"],
                "capture_or_dissipation_derived": row["capture_or_dissipation_derived"],
                "tensor_or_quantum_gravity_derived": row["tensor_or_quantum_gravity_derived"],
            }
        )
    for row in inputs["COMPARATOR_LEDGER"]["declared_source_or_solver_blocked"]:
        output.append(
            {
                "source": "REQUIRED_PUBLISHED_COMPARATOR",
                "concept_id": row["id"],
                "candidate_status": "SOURCE_OR_SOLVER_BLOCKED",
                "lane": "RIVALS_CONTROLS",
                "galaxies": row["status"],
                "clusters": row["status"],
                "groups": "NOT_TESTED",
                "lensing": "NOT_TESTED",
                "physical_time_dilation_derived": False,
                "light_propagation_derived": False,
                "redshift_closure_derived": False,
                "capture_or_dissipation_derived": False,
                "tensor_or_quantum_gravity_derived": False,
            }
        )
    output.sort(key=lambda row: (row["source"], row["concept_id"]))
    _require(len(output) == 285, "blocked/unscored row count changed")
    return output


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        rendered = {}
        for key in columns:
            value = row[key]
            if value is None:
                rendered[key] = ""
            elif type(value) is float:
                rendered[key] = format(value, ".12g")
            elif type(value) is bool:
                rendered[key] = "true" if value else "false"
            else:
                rendered[key] = value
        writer.writerow(rendered)
    return stream.getvalue().encode("utf-8")


def _summary_markdown(
    cells: Sequence[Mapping[str, Any]],
    concepts: Sequence[Mapping[str, Any]],
    ladder: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> bytes:
    valid = [row for row in cells if row["valid"]]
    galaxies = [row for row in valid if row["domain"] == "GALAXIES"]
    clusters = [row for row in valid if row["domain"] == "CLUSTERS"]
    gp01_gal = next(row for row in galaxies if row["cell_id"] == "GP01L-n1")
    gp01_cluster = next(row for row in clusters if row["cell_id"] == "GP01L-n1")
    status_counts = Counter(
        row["candidate_status"]
        for row in blocked
        if row["source"] == "CAMPAIGN_BLOCKED_IDEA_LEDGER"
    )
    text = f"""# Open-gravity campaign near-miss atlas

## What was actually tested

The registry contained 407 live candidate cards and 2,486 parameter cells, but these counts are not equivalent to 407 complete physical theories. The sealed real-data campaign admitted 179 galaxy cells and 1,669 cluster cells. After retained source/operator failures, **{len(galaxies)} galaxy cells** and **{len(clusters)} cluster cells** were validly scored on 139 SPARC galaxies and eight X-COP clusters.

The remaining registry included **{status_counts["REGISTERED_THEORY_ONLY"]} theory-only concepts**, **{status_counts["SOURCE_BLOCKED"]} source-blocked GP01 branches**, one known rewrite, and one quarantined action placeholder. These were not real-data falsifications.

## Strict result

No valid candidate cell beat the strongest executable comparator in even one frozen nuisance scenario. There were no galaxy-domain, cluster-domain, or cross-domain survivors. This remains the official preregistered result.

The strongest comparators were demanding contextual ceilings: an NFW halo for galaxies and the earlier cross-scale empirical formula for clusters. Failure against those ceilings is not the same statement as failure to improve on baryons alone.

## What the less binary comparison shows

- **{sum(row["baryon_outcome"].startswith("ROBUST_") for row in galaxies)} of {len(galaxies)}** valid galaxy cells beat baryons alone in every nuisance scenario; **{sum(row["baryon_outcome"] == "ROBUST_2PCT_WIN" for row in galaxies)}** did so by more than two percent.
- **{sum(row["baryon_outcome"].startswith("ROBUST_") for row in clusters)} of {len(clusters)}** valid cluster cells beat baryons alone in every nuisance scenario; **{sum(row["baryon_outcome"] == "ROBUST_2PCT_WIN" for row in clusters)}** did so by more than two percent.
- Only **{sum(row["rar_outcome"] == "NUISANCE_CONDITIONAL_WIN" for row in galaxies)}** galaxy cell conditionally beat RAR: GP01-L n=1. Its RAR improvement ranges from **{gp01_gal["rar_worst_fractional_improvement"]:.1%}** to **{gp01_gal["rar_best_fractional_improvement"]:.1%}** across stellar-mass nuisance choices, so it is not robust.
- No cluster cell beat RAR in any nuisance scenario. GP01-L n=1 ranges from **{gp01_cluster["rar_worst_fractional_improvement"]:.1%}** to **{gp01_cluster["rar_best_fractional_improvement"]:.1%}** relative to RAR and performs worse.
- No TWELL cell other than the already MOND-like GP01-L control even conditionally beats RAR.

## What was not tested

The blocked/unscored ledger has {len(blocked)} rows when the six required published comparator solvers are included. None of its mechanism cards derives physical time dilation, photon propagation, cumulative redshift, irreversible capture/dissipation, or tensor/quantum-gravity behavior. Groups and lensing were not response-scored. Dynamic memory, retarded history, and full three-dimensional source evolution remain outside this campaign.

Therefore the campaign strongly constrains the tested **static radial matter-response closures**. It does not eliminate the broader time-well, light-propagation, history, quantum, or full-3D ideas that lacked executable source/response closures.

## Publication assessment

The atlas supports a possible methods/negative-result paper about preregistered breadth versus executable depth and about the collapse of many static radial mechanisms against RAR and halo/cross-scale ceilings. It does not establish a unique new theory. The scientifically useful next step is to advance one genuinely orthogonal mechanism only when its required public source and response data and an independent solver benchmark are available.
"""
    _require(len(concepts) == 189 and len(ladder) == 15, "summary inputs changed")
    return text.encode("utf-8")


def _ladder_svg(rows: Sequence[Mapping[str, Any]]) -> bytes:
    width, height = 1160, 650
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333}.barG{fill:#4C78A8}.barC{fill:#F58518}</style>",
        '<text x="580" y="28" text-anchor="middle" font-size="19">Comparator ladder and candidate reach</text>',
        '<text x="580" y="50" text-anchor="middle" font-size="12">Bar length is log10(1 + mean loss); shorter comparators are harder to beat</text>',
    ]
    ordered = sorted(rows, key=lambda row: (row["domain"], row["mean_of_scenario_mean_losses"]))
    for index, row in enumerate(ordered):
        y = 82 + index * 35
        bar = 115 * math.log10(1 + row["mean_of_scenario_mean_losses"])
        color_class = "barG" if row["domain"] == "GALAXIES" else "barC"
        label = f"{row['domain']}: {row['comparator_id']}"
        parts.append(f'<text x="15" y="{y + 14}" font-size="11">{html.escape(label)}</text>')
        parts.append(f'<rect class="{color_class}" x="420" y="{y}" width="{bar:.3f}" height="18"/>')
        parts.append(
            f'<text x="{430 + bar:.3f}" y="{y + 14}" font-size="11">loss {row["mean_of_scenario_mean_losses"]:.2f}; all-scenario wins {row["candidate_cells_winning_all_scenarios"]}</text>'
        )
    parts.append("</svg>")
    return ("\n".join(parts) + "\n").encode("utf-8")


def build_artifacts(config: Mapping[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    inputs = _load_inputs(config)
    cells = _cell_rows(inputs, config["analysis_contract"]["strict_threshold_fraction"])
    concepts = _concept_rows(cells)
    ladder = _comparator_ladder(inputs, cells)
    blocked = _blocked_rows(inputs)
    cell_columns = list(cells[0])
    concept_columns = list(concepts[0])
    ladder_columns = list(ladder[0])
    blocked_columns = list(blocked[0])
    payloads = {
        "cell-comparator-atlas.csv": _csv_bytes(cells, cell_columns),
        "concept-domain-atlas.csv": _csv_bytes(concepts, concept_columns),
        "comparator-ladder.csv": _csv_bytes(ladder, ladder_columns),
        "blocked-and-unscored-ideas.csv": _csv_bytes(blocked, blocked_columns),
        "near-miss-summary.md": _summary_markdown(cells, concepts, ladder, blocked),
        "figure-near-miss-ladder.svg": _ladder_svg(ladder),
    }
    contract = {row["path"]: row for row in config["artifact_contract"]["artifacts"]}
    _require(set(payloads) == set(contract), "artifact set changed")
    index = [
        {
            "role": contract[name]["role"],
            "path": f"{ARTIFACT_DIRECTORY.as_posix()}/{name}",
            "rows": contract[name]["rows"],
            "bytes": len(payloads[name]),
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }
        for name in sorted(payloads)
    ]
    valid = [row for row in cells if row["valid"]]
    summary = {
        "registered_live_candidates": 407,
        "registered_parameter_cells": 2486,
        "planned_domain_cells": len(cells),
        "valid_domain_cells": len(valid),
        "invalid_source_gate_cells": len(cells) - len(valid),
        "strict_domain_survivors": sum(row["strict_domain_pass"] for row in cells),
        "galaxy_rar_conditional_cells": sum(
            row["domain"] == "GALAXIES" and row["rar_outcome"] == "NUISANCE_CONDITIONAL_WIN"
            for row in valid
        ),
        "cluster_rar_any_win_cells": sum(
            row["domain"] == "CLUSTERS" and row["rar_best_fractional_improvement"] > 0.0
            for row in valid
        ),
        "galaxy_baryon_robust_cells": sum(
            row["domain"] == "GALAXIES" and row["baryon_outcome"].startswith("ROBUST_")
            for row in valid
        ),
        "cluster_baryon_robust_cells": sum(
            row["domain"] == "CLUSTERS" and row["baryon_outcome"].startswith("ROBUST_")
            for row in valid
        ),
        "blocked_or_unscored_rows": len(blocked),
        "domain_concept_rows": len(concepts),
        "artifact_index": index,
        "artifact_index_sha256": content_sha256(index),
    }
    return payloads, summary


def _package_bindings() -> dict[str, str]:
    return {
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    _, summary = build_artifacts(config)
    claims = dict(config["claim_boundary"])
    claims["response_free_reaggregation_complete"] = True
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_RESPONSE_FREE_FULL_CAMPAIGN_NEAR_MISS_ATLAS_BUILT",
        "decision": "NO_STRICT_SURVIVOR_ONE_MOND_LIKE_GALAXY_RAR_CONDITIONAL_MOST_BROADER_TIME_LIGHT_HISTORY_IDEAS_UNSCORED",
        "package_bindings": _package_bindings(),
        "input_sha256": {row["role"]: row["sha256"] for row in config["input_bindings"]},
        "summary": summary,
        "access_accounting": {
            "sealed_metadata_and_aggregate_artifacts_read": len(config["input_bindings"]),
            "raw_scientific_response_payloads_opened": 0,
            "new_response_rows_scored": 0,
            "formula_or_parameter_retuning_events": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
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
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
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


def write_packet() -> str:
    config = load_config()
    payloads, _ = build_artifacts(config)
    states = [
        _atomic_no_clobber(_repo_path(ARTIFACT_DIRECTORY / name), payload)
        for name, payload in sorted(payloads.items())
    ]
    receipt = build_receipt(config)
    validate_receipt(config, receipt)
    states.append(_atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt) + b"\n"))
    return (
        "EXISTING_IDENTICAL"
        if all(state == "EXISTING_IDENTICAL" for state in states)
        else "CREATED"
    )


def check_packet() -> str:
    config = load_config()
    payloads, _ = build_artifacts(config)
    for name, payload in payloads.items():
        path = _repo_path(ARTIFACT_DIRECTORY / name)
        _require(path.is_file(), f"artifact missing: {name}")
        _require(path.read_bytes() == payload, f"artifact differs: {name}")
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    receipt = _read_json(path, "receipt")
    _require(type(receipt) is dict, "receipt must be object")
    validate_receipt(config, receipt)
    return "VALID"


def status() -> dict[str, Any]:
    config = load_config()
    if not _repo_path(OUTPUT_PATH).is_file():
        return {"package_id": config["package_id"], "status": "FROZEN_UNBUILT"}
    receipt = _read_json(_repo_path(OUTPUT_PATH), "receipt")
    validate_receipt(config, receipt)
    return {
        "package_id": config["package_id"],
        "status": receipt["status"],
        "decision": receipt["decision"],
        "strict_survivors": receipt["summary"]["strict_domain_survivors"],
        "galaxy_rar_conditional_cells": receipt["summary"]["galaxy_rar_conditional_cells"],
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
        print(write_packet())
    elif arguments.command == "check":
        print(check_packet())
    else:
        print(json.dumps(status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
