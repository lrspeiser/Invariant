from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_cluster_predictor_strata_preflight_v1.json")
TEST_PATH = Path("tests/test_gravity_cluster_predictor_strata_preflight.py")
ARTIFACT_DIR = Path("runs/gravity/publication-readiness/cluster-predictor-strata-preflight-v1")
SOURCE_RECEIPT_PATH = ARTIFACT_DIR / "public-predictor-source-receipt.json"
STRATA_PATH = ARTIFACT_DIR / "predictor-only-strata.json"
CAUSE_MATRIX_PATH = ARTIFACT_DIR / "alternative-cause-preflight-matrix.json"
RECEIPT_PATH = Path("runs/gravity/publication-readiness/cluster-predictor-strata-preflight-v1.json")

CONFIG_SCHEMA = "invariant-gravity-cluster-predictor-strata-preflight-config-1.0"
SOURCE_SCHEMA = "invariant-gravity-cluster-public-predictor-source-receipt-1.0"
STRATA_SCHEMA = "invariant-gravity-cluster-predictor-strata-1.0"
MATRIX_SCHEMA = "invariant-gravity-cluster-alternative-cause-preflight-matrix-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-predictor-strata-preflight-receipt-1.0"

CLUSTERS = ["A1644", "A1795", "A2142", "A2255", "A2319", "A3266", "A85", "ZW1215"]
CAUSES = [
    "nonthermal_pressure",
    "extra_member_baryons",
    "calibration_shift",
    "gas_clumping",
    "geometry",
    "merger_or_assembly_state",
    "boundary_error",
]
MORPHOLOGY_METRICS = ["concentration_csb", "centroid_shift_x1e3", "gini", "zernike_cz"]
EXPECTED_PUBLIC_SOURCES = [
    {
        "source_id": "dupourque_et_al_2023_xcop_morphology",
        "title": "Investigating the turbulent hot gas in X-COP galaxy clusters",
        "authors": "Dupourque et al.",
        "year": 2023,
        "journal": "Astronomy & Astrophysics 673 A91",
        "doi": "10.1051/0004-6361/202245779",
        "arxiv_id": "2303.15102",
        "url": "https://arxiv.org/abs/2303.15102",
        "payload_filename": "dupourque-2303.15102.tar",
        "payload_sha256": ("73423b343a9f3df9f0a6d95d47a5847f4140b1d3e737c39937248318555bf676"),
        "payload_bytes": 10_865_934,
        "extracted_member": "morpho_parameters.tex",
        "extracted_member_sha256": (
            "3aa84b89004f5cf8d1cdf29f42a6ee4729e682059acedf541206ac125f3241dd"
        ),
        "authoritative_fields": [
            "concentration_csb",
            "centroid_shift_x1e3",
            "gini",
            "zernike_cz",
            "positive_merger_or_sloshing_text_flags",
        ],
        "role": "published_predictor_metadata_only_not_target_response",
    },
    {
        "source_id": "ghirardini_et_al_2019_xcop_thermodynamics",
        "title": (
            "Universal thermodynamic properties of the intracluster medium over two "
            "decades in radius in the X-COP sample"
        ),
        "authors": "Ghirardini et al.",
        "year": 2019,
        "journal": "Astronomy & Astrophysics 621 A41",
        "doi": "10.1051/0004-6361/201833325",
        "arxiv_id": "1805.00042",
        "url": "https://arxiv.org/abs/1805.00042",
        "payload_filename": "ghirardini-1805.00042.tar",
        "payload_sha256": ("9edb7d25de34b769d7c8591db38bfbc7325c93f0d83d9569a774396c3d8bfa38"),
        "payload_bytes": 5_986_634,
        "extracted_member": "XCOP_thermo.tex",
        "extracted_member_sha256": (
            "ef37d3b5708c5e648da5e1c402ccd66441f979a42993dd2b4f9cc1eba1ca8838"
        ),
        "authoritative_fields": [
            "central_entropy_kev_cm2",
            "published_cool_core_class",
            "boundary_background_method",
        ],
        "role": "published_predictor_metadata_only_not_target_response",
    },
]
ASSEMBLY_LABELS = (
    "no_class_assigned_in_frozen_source",
    "sloshing_reported",
    "sub_or_post_merger_reported",
)
EXPECTED_ASSEMBLY_BY_CLUSTER = {
    "A1644": "no_class_assigned_in_frozen_source",
    "A1795": "no_class_assigned_in_frozen_source",
    "A2142": "sloshing_reported",
    "A2255": "no_class_assigned_in_frozen_source",
    "A2319": "sub_or_post_merger_reported",
    "A3266": "sub_or_post_merger_reported",
    "A85": "sloshing_reported",
    "ZW1215": "no_class_assigned_in_frozen_source",
}
BOUNDARY_LABELS = (
    "xmm_mosaic_r_gt_2r500_5pct_systematic",
    "rosat_background_30pct_systematic",
)
EXPECTED_BOUNDARY_BY_CLUSTER = {
    "A1644": "xmm_mosaic_r_gt_2r500_5pct_systematic",
    "A1795": "xmm_mosaic_r_gt_2r500_5pct_systematic",
    "A2142": "xmm_mosaic_r_gt_2r500_5pct_systematic",
    "A2255": "xmm_mosaic_r_gt_2r500_5pct_systematic",
    "A2319": "xmm_mosaic_r_gt_2r500_5pct_systematic",
    "A3266": "rosat_background_30pct_systematic",
    "A85": "xmm_mosaic_r_gt_2r500_5pct_systematic",
    "ZW1215": "xmm_mosaic_r_gt_2r500_5pct_systematic",
}
EXPECTED_CLUSTER_ROW_SHA256 = {
    "A1644": "bfb928b534480feb10d5d91559b0e44690e9cfbf1be4dbce9aa9037c4c2d8a60",
    "A1795": "f5dc770fb572420fdbc67974366828ae08f692961e2c761fa37e6d3fc99ec9c7",
    "A2142": "d358d8994a8dce098a67dd2d11c557c15a0f28e812db259cff0789cab16b5406",
    "A2255": "6b686f1e6485fe4e1b82beb0786e4f3eb599925c383d078712a6f4b337dac873",
    "A2319": "7dd1b62a3bc9b84fcdd4f54645b690f579993f3d512057aaad61335c93a3455c",
    "A3266": "90b74920c9ad7ac9055b84cd444f58702bcd4a929868e7aadb29baf12f2dee10",
    "A85": "e31ea1172403a34cb21db95c692380dc575c2071e8589d3770cdfca9be146d8d",
    "ZW1215": "ced6f203fbf1477a8ebc3c78b029c1215a66253294fe0640694934682cde2d95",
}
EXPECTED_MISSINGNESS_POLICY = {
    "predictor_values_are_never_imputed_from_targets_or_responses": True,
    "absent_public_predictor_remains_explicit_missing": True,
    "absent_assembly_statement_is_unclassified_not_negative": True,
    "incomplete_stellar_profile_coverage_is_reported_as_availability_stratum": True,
    "no_cluster_is_dropped_or_replaced_for_missingness": True,
    "all_missingness_counts_are_reported": True,
}
EXPECTED_CLAIM_BOUNDARY = {
    "CP5_11_predictor_definitions_frozen": True,
    "CP5_11_predictor_labels_ready": True,
    "CP5_11_scientific_stratum_scoring_complete": False,
    "alternative_cause_planning_matrix_present": True,
    "CP5_13_task_complete": False,
    "CP5_13_scientific_comparisons_complete": False,
    "ordinary_halo_comparison_reopened_or_rescored": False,
    "cause_identified": False,
    "candidate_supported_or_refuted": False,
    "publication_readiness_changed": False,
    "scientific_claim_allowed": False,
}
EXPECTED_CAUSE_STATUSES = {
    "nonthermal_pressure": {
        "predictor_only_preflight_status": "EXECUTABLE_NOW",
        "scientific_comparison_status": (
            "BLOCKED_MISSING_INDEPENDENT_NONTHERMAL_PRESSURE_PROFILES_AND_"
            "AUTHORIZED_SCORING_PROTOCOL"
        ),
    },
    "extra_member_baryons": {
        "predictor_only_preflight_status": "EXECUTABLE_NOW",
        "scientific_comparison_status": (
            "BLOCKED_INCOMPLETE_BARYON_POSTERIORS_AND_AUTHORIZED_SCORING_PROTOCOL"
        ),
    },
    "calibration_shift": {
        "predictor_only_preflight_status": "EXECUTABLE_NOW",
        "scientific_comparison_status": (
            "BLOCKED_MISSING_INDEPENDENT_CROSS_CALIBRATION_PREDICTORS_AND_"
            "AUTHORIZED_SCORING_PROTOCOL"
        ),
    },
    "gas_clumping": {
        "predictor_only_preflight_status": "EXECUTABLE_NOW",
        "scientific_comparison_status": (
            "BLOCKED_MISSING_TARGET_INDEPENDENT_CLUMPING_PREDICTORS_AND_AUTHORIZED_SCORING_PROTOCOL"
        ),
    },
    "geometry": {
        "predictor_only_preflight_status": "EXECUTABLE_NOW",
        "scientific_comparison_status": (
            "BLOCKED_MISSING_INDEPENDENT_3D_GEOMETRY_POSTERIORS_AND_AUTHORIZED_SCORING_PROTOCOL"
        ),
    },
    "merger_or_assembly_state": {
        "predictor_only_preflight_status": "EXECUTABLE_NOW",
        "scientific_comparison_status": (
            "BLOCKED_NO_CLEAN_NEGATIVE_ASSEMBLY_CLASS_SMALL_CELLS_AND_AUTHORIZED_SCORING_PROTOCOL"
        ),
    },
    "boundary_error": {
        "predictor_only_preflight_status": "EXECUTABLE_NOW",
        "scientific_comparison_status": (
            "BLOCKED_SINGLE_EXCEPTION_INSUFFICIENT_SUPPORT_AND_AUTHORIZED_SCORING_PROTOCOL"
        ),
    },
}
EXPECTED_CAUSE_ROW_SHA256 = {
    "nonthermal_pressure": ("c3bc6867770c9de4679661268e1c27d8d6afad73d38794deb72c3039f615342d"),
    "extra_member_baryons": ("d3de464894455f1756569653dea700a6cc9fcc3de6e47b919a95a905d93fc0df"),
    "calibration_shift": ("a4d2f307234e3c1b75e562ce2f2134e5dcba18fa1779d7e69a6217343fccc534"),
    "gas_clumping": "a35731d00c73ca0801c030b11b33d1fefcd6ba1165a395ecf69c21b451e9ed5e",
    "geometry": "f0b63424d2070dec571957615ff2e92b290dab5d66b5c0d5b9d92758fd73e198",
    "merger_or_assembly_state": (
        "c9f44a7433de065bcf4f1ed1215843440028031ae24eeac339efb41fee4ba6fb"
    ),
    "boundary_error": ("f3be118e79ae429348b788296fa25a87cc13c917e2186b1fc4649f7ee04f743c"),
}
FORBIDDEN_CLUSTER_KEYS = {
    "target",
    "response",
    "residual",
    "likelihood",
    "posterior",
    "pressure",
    "temperature",
    "holdout",
    "confirmation",
    "independent",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def content_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    ).hexdigest()


def confined(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"path escaped repository: {path}") from error
    return target


def strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise RuntimeError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def write_json(path: Path, value: dict[str, Any]) -> None:
    target = confined(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
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
        raise RuntimeError(f"refusing to overwrite existing artifact: {target}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def artifact_binding(path: Path) -> dict[str, str]:
    target = confined(path)
    if not target.is_file():
        raise RuntimeError(f"bound artifact is absent: {path}")
    return {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }


def validate_binding(binding: dict[str, Any], label: str) -> Path:
    strict_keys(binding, {"path", "file_sha256"}, label)
    target = confined(ROOT / str(binding["path"]))
    if not target.is_file() or file_sha256(target) != binding["file_sha256"]:
        raise RuntimeError(f"{label} missing or tampered")
    return target


def contains_forbidden_cluster_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_CLUSTER_KEYS):
                return True
            if contains_forbidden_cluster_key(child):
                return True
    elif isinstance(value, list):
        return any(contains_forbidden_cluster_key(child) for child in value)
    return False


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if target != ROOT / CONFIG_PATH or not target.is_file():
        raise RuntimeError("predictor-strata config path changed")
    if file_sha256(target) != expected_sha256:
        raise RuntimeError("predictor-strata config hash changed")
    config = json.loads(target.read_text(encoding="utf-8"))
    validate_config_contract(config)
    config["_config_sha256"] = expected_sha256
    return config


def validate_config_contract(config: dict[str, Any]) -> None:
    strict_keys(
        config,
        {
            "schema_version",
            "status",
            "purpose",
            "implementation_source",
            "implementation_source_normalized_sha256",
            "local_population_binding",
            "public_sources",
            "population",
            "cluster_metadata",
            "strata_definitions",
            "missingness_policy",
            "no_post_response_selection_rule",
            "alternative_cause_matrix",
            "data_boundary",
            "claim_boundary",
            "output_paths",
        },
        "predictor-strata config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_predictor_only_preflight_before_any_stratum_scoring"
        or config["population"]
        != {
            "role": "already_exposed_development_only",
            "cluster_ids": CLUSTERS,
            "count": 8,
            "replacement_allowed": False,
        }
        or [row["cluster_id"] for row in config["cluster_metadata"]] != CLUSTERS
        or [row["cause_id"] for row in config["alternative_cause_matrix"]] != CAUSES
    ):
        raise RuntimeError("predictor-strata frozen identity changed")
    source = confined(ROOT / config["implementation_source"])
    if (
        source != Path(__file__).resolve()
        or normalized_sha256(source) != config["implementation_source_normalized_sha256"]
    ):
        raise RuntimeError("predictor-strata implementation changed after freeze")
    validate_binding(config["local_population_binding"], "local_population_binding")
    local = json.loads(
        (ROOT / config["local_population_binding"]["path"]).read_text(encoding="utf-8")
    )
    if local.get("population", {}).get("development_clusters_already_exposed") != CLUSTERS:
        raise RuntimeError("local exposed development population changed")
    if config["public_sources"] != EXPECTED_PUBLIC_SOURCES:
        raise RuntimeError("authoritative public-source provenance changed")
    for row in config["cluster_metadata"]:
        validate_cluster_metadata(row)
    validate_definitions(config)
    validate_cause_matrix(config["alternative_cause_matrix"])
    if config["missingness_policy"] != EXPECTED_MISSINGNESS_POLICY:
        raise RuntimeError("missingness policy changed")
    if config["claim_boundary"] != EXPECTED_CLAIM_BOUNDARY:
        raise RuntimeError("claim boundary changed")
    if config["data_boundary"] != {
        "development_cluster_identities_used": 8,
        "public_predictor_rows_used": 8,
        "target_or_response_rows_loaded": 0,
        "holdout_rows_loaded": 0,
        "confirmation_rows_loaded": 0,
        "independent_rows_loaded": 0,
        "target_scoring_calls": 0,
        "model_or_paid_calls": 0,
        "public_source_http_payloads_acquired": 2,
    }:
        raise RuntimeError("predictor-strata data boundary changed")


def validate_cluster_metadata(row: dict[str, Any]) -> None:
    strict_keys(
        row,
        {
            "cluster_id",
            "aliases",
            "morphology",
            "central_entropy_kev_cm2",
            "published_cool_core_class",
            "published_stellar_profile_available",
            "assembly_literature_flag",
            "boundary_background_method",
            "provenance_ids",
        },
        "cluster metadata",
    )
    if contains_forbidden_cluster_key(row):
        raise RuntimeError("cluster metadata contains a forbidden response-derived key")
    if not row["aliases"] or row["cluster_id"] not in row["aliases"]:
        raise RuntimeError("canonical cluster id missing from aliases")
    cluster_id = row["cluster_id"]
    if (
        row["assembly_literature_flag"] not in ASSEMBLY_LABELS
        or row["assembly_literature_flag"] != EXPECTED_ASSEMBLY_BY_CLUSTER[cluster_id]
    ):
        raise RuntimeError("assembly literature label changed")
    if (
        row["boundary_background_method"] not in BOUNDARY_LABELS
        or row["boundary_background_method"] != EXPECTED_BOUNDARY_BY_CLUSTER[cluster_id]
    ):
        raise RuntimeError("boundary background label changed")
    if set(row["morphology"]) != set(MORPHOLOGY_METRICS):
        raise RuntimeError("morphology metric set changed")
    for metric in row["morphology"].values():
        strict_keys(metric, {"value", "minus", "plus"}, "morphology measurement")
        values = np.asarray([metric["value"], metric["minus"], metric["plus"]], dtype=float)
        if not np.all(np.isfinite(values)) or metric["minus"] < 0 or metric["plus"] < 0:
            raise RuntimeError("morphology measurement invalid")
    entropy = float(row["central_entropy_kev_cm2"])
    if not np.isfinite(entropy) or entropy < 0:
        raise RuntimeError("central entropy invalid")
    expected_core = "CC" if entropy < 30.0 else "NCC"
    if row["published_cool_core_class"] != expected_core:
        raise RuntimeError("cool-core class disagrees with frozen K0 threshold")
    if content_sha256(row) != EXPECTED_CLUSTER_ROW_SHA256.get(cluster_id):
        raise RuntimeError(f"cluster metadata row changed: {cluster_id}")


def validate_definitions(config: dict[str, Any]) -> None:
    definitions = config["strata_definitions"]
    strict_keys(
        definitions,
        {
            "cool_core",
            "morphology_metric_strata",
            "relaxation_proxy",
            "assembly_literature",
        },
        "strata definitions",
    )
    if definitions["cool_core"] != {
        "source_field": "central_entropy_kev_cm2",
        "cc_rule": "K0 < 30 keV cm^2",
        "ncc_rule": "K0 >= 30 keV cm^2",
        "threshold_kev_cm2": 30.0,
        "missing_label": "cool_core_missing",
    }:
        raise RuntimeError("cool-core definition changed")
    if definitions["morphology_metric_strata"] != {
        "rule": "strictly_below_vs_strictly_above_each_frozen_eight_cluster_sample_median",
        "tie_label": "at_frozen_median",
        "missing_label": "morphology_metric_missing",
        "physical_claim": "descriptive_predictor_partition_only",
    }:
        raise RuntimeError("morphology-metric strata definition changed")
    proxy = definitions["relaxation_proxy"]
    if (
        proxy.get("metrics") != MORPHOLOGY_METRICS
        or proxy.get("disturbance_directions")
        != {
            "concentration_csb": "lower_is_more_disturbed",
            "centroid_shift_x1e3": "higher_is_more_disturbed",
            "gini": "lower_is_more_disturbed",
            "zernike_cz": "higher_is_more_disturbed",
        }
        or proxy.get("score")
        != "mean_of_four_within_population_fractional_midranks_in_disturbance_direction"
        or proxy.get("split") != "strictly_below_vs_strictly_above_frozen_sample_median"
        or proxy.get("labels") != ["relaxed_proxy", "disturbed_proxy"]
        or proxy.get("tie_label") != "relaxation_proxy_median_tie"
        or proxy.get("physical_claim")
        != "response_blind_morphology_proxy_not_a_definitive_dynamical_classification"
    ):
        raise RuntimeError("relaxation-proxy definition changed")
    if definitions["assembly_literature"] != {
        "rule": (
            "retain_only_explicit_positive_sub_or_post_merger_or_sloshing_statements_"
            "from_frozen_source"
        ),
        "absent_statement_label": "no_class_assigned_in_frozen_source",
        "absent_statement_meaning": ("unclassified_not_a_negative_nonmerger_or_nonsloshing_class"),
    }:
        raise RuntimeError("assembly literature definition changed")
    if config["no_post_response_selection_rule"] != {
        "population_fixed_before_stratum_scoring": True,
        "drop_or_replace_cluster_after_response_access": False,
        "change_threshold_or_direction_after_response_access": False,
        "merge_small_cells_after_response_access": False,
        "impute_predictor_from_any_response": False,
        "missing_values_remain_explicit_missing_strata": True,
        "all_strata_results_must_be_reported_even_if_unfavorable": True,
        "scientific_scoring_requires_a_separate_frozen_authorized_protocol": True,
    }:
        raise RuntimeError("no-post-response-selection rule changed")


def validate_cause_matrix(rows: list[dict[str, Any]]) -> None:
    expected_keys = {
        "cause_id",
        "ordinary_explanation",
        "available_predictors",
        "predictor_only_test",
        "predictor_only_preflight_status",
        "future_scientific_test",
        "scientific_comparison_status",
        "blocking_evidence",
        "minimum_future_requirements",
        "interpretation_ceiling",
    }
    if [row.get("cause_id") for row in rows] != CAUSES:
        raise RuntimeError("alternative-cause order changed")
    for row in rows:
        strict_keys(row, expected_keys, "alternative-cause row")
        cause_id = row["cause_id"]
        actual_status = {
            "predictor_only_preflight_status": row["predictor_only_preflight_status"],
            "scientific_comparison_status": row["scientific_comparison_status"],
        }
        if actual_status != EXPECTED_CAUSE_STATUSES[cause_id]:
            raise RuntimeError(f"alternative-cause status changed: {cause_id}")
        if (
            not row["ordinary_explanation"]
            or not isinstance(row["available_predictors"], list)
            or not row["predictor_only_test"]
            or not row["future_scientific_test"]
            or not row["blocking_evidence"]
            or not row["minimum_future_requirements"]
            or not row["interpretation_ceiling"]
        ):
            raise RuntimeError(f"alternative-cause contract incomplete: {cause_id}")
        if content_sha256(row) != EXPECTED_CAUSE_ROW_SHA256[cause_id]:
            raise RuntimeError(f"alternative-cause row changed: {cause_id}")


def expected_source_receipt(config: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": SOURCE_SCHEMA,
        "status": "authoritative_public_predictors_extracted_no_target_responses",
        "local_population_binding": config["local_population_binding"],
        "public_sources": config["public_sources"],
        "extraction": {
            "cluster_ids": CLUSTERS,
            "morphology_fields": MORPHOLOGY_METRICS,
            "cool_core_field": "central_entropy_kev_cm2_and_published_CC_NCC",
            "assembly_text_role": "positive_literature_flags_only_absence_is_unclassified",
            "boundary_method_role": "public_observing_and_background_method_metadata",
            "aliases_frozen": True,
        },
        "counts": {
            "public_source_payloads": 2,
            "cluster_rows": 8,
            "morphology_measurements": 32,
            "central_entropy_measurements": 8,
            "target_or_response_rows": 0,
        },
        "data_boundary": config["data_boundary"],
    }
    body["content_sha256"] = content_sha256(body)
    return body


def fractional_midranks(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise RuntimeError("fractional midrank input invalid")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / (len(values) - 1)


def build_strata(config: dict[str, Any]) -> dict[str, Any]:
    metadata = config["cluster_metadata"]
    measurements = {
        name: np.asarray([row["morphology"][name]["value"] for row in metadata], dtype=float)
        for name in MORPHOLOGY_METRICS
    }
    medians = {name: float(np.median(values)) for name, values in measurements.items()}
    direction_sign = {
        "concentration_csb": -1.0,
        "centroid_shift_x1e3": 1.0,
        "gini": -1.0,
        "zernike_cz": 1.0,
    }
    rank_columns = [
        fractional_midranks(direction_sign[name] * measurements[name])
        for name in MORPHOLOGY_METRICS
    ]
    disturbance = np.mean(np.stack(rank_columns, axis=1), axis=1)
    disturbance_median = float(np.median(disturbance))
    rows = []
    for index, source in enumerate(metadata):
        metric_strata = {
            name: (
                "below_frozen_median"
                if measurements[name][index] < medians[name]
                else "above_frozen_median"
                if measurements[name][index] > medians[name]
                else "at_frozen_median"
            )
            for name in MORPHOLOGY_METRICS
        }
        relaxation = (
            "relaxed_proxy"
            if disturbance[index] < disturbance_median
            else "disturbed_proxy"
            if disturbance[index] > disturbance_median
            else "relaxation_proxy_median_tie"
        )
        rows.append(
            {
                "cluster_id": source["cluster_id"],
                "aliases": source["aliases"],
                "morphology": source["morphology"],
                "morphology_metric_strata": metric_strata,
                "relaxation_proxy_score": float(disturbance[index]),
                "relaxation_proxy_stratum": relaxation,
                "central_entropy_kev_cm2": source["central_entropy_kev_cm2"],
                "cool_core_stratum": source["published_cool_core_class"],
                "published_stellar_profile_available": source[
                    "published_stellar_profile_available"
                ],
                "assembly_literature_stratum": source["assembly_literature_flag"],
                "boundary_background_method": source["boundary_background_method"],
                "provenance_ids": source["provenance_ids"],
            }
        )
    if any(contains_forbidden_cluster_key(row) for row in rows):
        raise RuntimeError("derived strata contain a forbidden response-derived key")
    relaxation_counts = {
        label: sum(row["relaxation_proxy_stratum"] == label for row in rows)
        for label in ("relaxed_proxy", "disturbed_proxy", "relaxation_proxy_median_tie")
    }
    cool_core_counts = {
        label: sum(row["cool_core_stratum"] == label for row in rows)
        for label in ("CC", "NCC", "cool_core_missing")
    }
    assembly_counts: dict[str, int] = {}
    for row in rows:
        label = row["assembly_literature_stratum"]
        assembly_counts[label] = assembly_counts.get(label, 0) + 1
    body = {
        "schema_version": STRATA_SCHEMA,
        "status": "predictor_strata_ready_scientific_scoring_not_run",
        "definitions": config["strata_definitions"],
        "frozen_morphology_medians": medians,
        "frozen_relaxation_proxy_median": disturbance_median,
        "cluster_rows": rows,
        "strata_counts": {
            "relaxation_proxy": relaxation_counts,
            "cool_core": cool_core_counts,
            "assembly_literature": assembly_counts,
            "published_stellar_profile": {
                "available": sum(row["published_stellar_profile_available"] for row in rows),
                "unavailable": sum(not row["published_stellar_profile_available"] for row in rows),
            },
            "boundary_background_method": {
                "standard_xmm_outer_background": sum(
                    row["boundary_background_method"] == "xmm_mosaic_r_gt_2r500_5pct_systematic"
                    for row in rows
                ),
                "rosat_exception": sum(
                    row["boundary_background_method"] == "rosat_background_30pct_systematic"
                    for row in rows
                ),
            },
        },
        "missingness": {
            "morphology_rows_missing": 0,
            "central_entropy_rows_missing": 0,
            "aliases_missing": 0,
            "assembly_negative_class_available": False,
            "assembly_unclassified_rows": assembly_counts.get(
                "no_class_assigned_in_frozen_source", 0
            ),
            "direct_nonthermal_pressure_predictor_rows_missing": 8,
            "direct_calibration_shift_predictor_rows_missing": 8,
            "independent_clumping_predictor_rows_missing": 8,
            "independent_3d_geometry_predictor_rows_missing": 8,
            "missingness_imputed": False,
        },
        "no_post_response_selection_rule": config["no_post_response_selection_rule"],
        "readiness": {
            "CP5_11_predictor_strata_frozen": True,
            "scientific_stratum_scoring_complete": False,
            "alternative_cause_planning_matrix_present": True,
            "CP5_13_task_complete": False,
            "CP5_13_causal_comparison_complete": False,
        },
        "data_boundary": config["data_boundary"],
        "claim_boundary": config["claim_boundary"],
    }
    body["content_sha256"] = content_sha256(body)
    return body


def build_cause_matrix(config: dict[str, Any], strata: dict[str, Any]) -> dict[str, Any]:
    rows = config["alternative_cause_matrix"]
    executable = sum(row["predictor_only_preflight_status"] == "EXECUTABLE_NOW" for row in rows)
    blocked = sum(row["scientific_comparison_status"].startswith("BLOCKED_") for row in rows)
    body = {
        "schema_version": MATRIX_SCHEMA,
        "status": "predictor_only_comparison_map_ready_scientific_comparisons_blocked",
        "cause_rows": rows,
        "summary": {
            "causes": len(rows),
            "predictor_only_preflights_executable_now": executable,
            "scientific_comparisons_blocked": blocked,
            "scientific_comparisons_complete": 0,
            "ordinary_halo_comparison_role": (
                "already_frozen_elsewhere_in_CP4_not_reopened_or_rescored_here"
            ),
        },
        "strata_binding": {
            "content_sha256": strata["content_sha256"],
            "cluster_count": len(strata["cluster_rows"]),
        },
        "no_post_response_selection_rule": config["no_post_response_selection_rule"],
        "data_boundary": config["data_boundary"],
        "claim_boundary": config["claim_boundary"],
    }
    body["content_sha256"] = content_sha256(body)
    return body


def expected_receipt(
    config: dict[str, Any], source: dict[str, Any], strata: dict[str, Any], matrix: dict[str, Any]
) -> dict[str, Any]:
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "predictor_preflight_pass_scientific_scoring_not_run",
        "decision": "CP5_11_STRATA_FROZEN_CP5_13_REMAINS_OPEN_PLANNING_MATRIX_ONLY",
        "evidence": {
            "source": artifact_binding(Path(__file__)),
            "config": artifact_binding(ROOT / CONFIG_PATH),
            "tests": artifact_binding(ROOT / TEST_PATH),
            "public_source_receipt": artifact_binding(ROOT / SOURCE_RECEIPT_PATH),
            "predictor_strata": artifact_binding(ROOT / STRATA_PATH),
            "alternative_cause_matrix": artifact_binding(ROOT / CAUSE_MATRIX_PATH),
        },
        "content_bindings": {
            "public_source_receipt": source["content_sha256"],
            "predictor_strata": strata["content_sha256"],
            "alternative_cause_matrix": matrix["content_sha256"],
        },
        "counts": {
            "development_clusters": 8,
            "cool_core": strata["strata_counts"]["cool_core"]["CC"],
            "non_cool_core": strata["strata_counts"]["cool_core"]["NCC"],
            "relaxed_proxy": strata["strata_counts"]["relaxation_proxy"]["relaxed_proxy"],
            "disturbed_proxy": strata["strata_counts"]["relaxation_proxy"]["disturbed_proxy"],
            "alternative_causes_mapped": len(matrix["cause_rows"]),
            "target_or_response_rows_loaded": 0,
            "target_scoring_calls": 0,
        },
        "readiness": {
            "CP5_11_predictor_definition_and_labels_ready": True,
            "CP5_11_scientific_stratum_scoring_complete": False,
            "alternative_cause_planning_matrix_present": True,
            "CP5_13_task_complete": False,
            "CP5_13_scientific_alternative_cause_comparison_complete": False,
        },
        "data_boundary": config["data_boundary"],
        "claim_boundary": config["claim_boundary"],
    }
    body["content_sha256"] = content_sha256(body)
    return body


def build(config_path: Path, expected_config_sha256: str) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    source = expected_source_receipt(config)
    strata = build_strata(config)
    matrix = build_cause_matrix(config, strata)
    write_json(ROOT / SOURCE_RECEIPT_PATH, source)
    write_json(ROOT / STRATA_PATH, strata)
    write_json(ROOT / CAUSE_MATRIX_PATH, matrix)
    receipt = expected_receipt(config, source, strata, matrix)
    write_json(ROOT / RECEIPT_PATH, receipt)
    return receipt


def load_exact_json(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file():
        raise RuntimeError(f"{label} missing")
    actual = json.loads(target.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError(f"{label} changed")
    return actual


def check(config_path: Path, expected_config_sha256: str, receipt_path: Path) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    source = load_exact_json(
        ROOT / SOURCE_RECEIPT_PATH,
        expected_source_receipt(config),
        "public predictor source receipt",
    )
    strata = load_exact_json(ROOT / STRATA_PATH, build_strata(config), "predictor-only strata")
    matrix = load_exact_json(
        ROOT / CAUSE_MATRIX_PATH,
        build_cause_matrix(config, strata),
        "alternative-cause matrix",
    )
    receipt = load_exact_json(
        receipt_path,
        expected_receipt(config, source, strata, matrix),
        "predictor-strata implementation receipt",
    )
    for name, binding in receipt["evidence"].items():
        validate_binding(binding, f"receipt.evidence.{name}")
    return {
        "valid": True,
        "status": receipt["status"],
        "decision": receipt["decision"],
        "CP5_11_predictor_definition_and_labels_ready": True,
        "CP5_11_scientific_stratum_scoring_complete": False,
        "alternative_cause_planning_matrix_present": True,
        "CP5_13_task_complete": False,
        "CP5_13_scientific_alternative_cause_comparison_complete": False,
        "target_or_response_rows_loaded": 0,
        "target_scoring_calls": 0,
        "scientific_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "check"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--expected-config-sha256", required=True)
        if name == "check":
            command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.config, args.expected_config_sha256)
    else:
        result = check(args.config, args.expected_config_sha256, args.receipt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
