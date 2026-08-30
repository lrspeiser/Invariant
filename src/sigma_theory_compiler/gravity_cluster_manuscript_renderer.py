"""Render the frozen primary cluster-development tables and figures in one command."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_cluster_manuscript_renderer_v1.json")
CONFIG_SCHEMA = "invariant-gravity-cluster-manuscript-renderer-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-manuscript-artifact-manifest-1.0"
TABLE_IDS = (
    "TABLE_1_CANDIDATE_AND_CLAIMS",
    "TABLE_2_SPLIT_PERFORMANCE",
    "TABLE_3_COMPARATORS_AND_ABLATIONS",
    "TABLE_4_OBJECT_PERFORMANCE",
    "TABLE_5_ROBUSTNESS_AND_CONTROLS",
    "TABLE_6_ACCESS_CLAIMS_LIMITATIONS",
    "TABLE_7_PRIOR_ART_BOUNDARY",
)
FIGURE_IDS = (
    "FIGURE_1_OBSERVED_VS_PREDICTED",
    "FIGURE_2_COMPARATOR_SCORES",
    "FIGURE_3_OBJECT_SCORE_HEATMAP",
    "FIGURE_4_COVARIANCE_STRESS",
    "FIGURE_5_RADIAL_RESIDUALS",
    "FIGURE_6_FALSE_SELECTION_AND_POWER",
)
SOURCE_IDS = ("manuscript_evidence_package", "candidate_specification")
SPLIT_COLORS = {
    "development_train": "#7c8da6",
    "development_holdout": "#087e8b",
    "confirmation": "#ff5a5f",
}


class GravityClusterManuscriptRendererError(RuntimeError):
    """Raised when the frozen renderer, evidence, or artifacts change."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value.replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterManuscriptRendererError(f"expected JSON object: {path}")
    return value


def _content_sha(value: Mapping[str, Any]) -> str:
    body = dict(value)
    expected = body.pop("content_sha256", None)
    actual = _sha(body)
    if expected != actual:
        raise GravityClusterManuscriptRendererError("bound source content hash changed")
    return actual


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterManuscriptRendererError(f"{label} keys changed")


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise GravityClusterManuscriptRendererError(f"{label} escaped root") from error
    return path


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root.resolve() / CONFIG_PATH)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "renderer_id",
            "purpose",
            "source_bindings",
            "primary_tables",
            "primary_figures",
            "rendering_contract",
            "output_directory",
            "receipt_path",
        },
        "manuscript renderer config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_development_artifacts_only"
        or config["renderer_id"] != "gravity-cluster-manuscript-renderer-v1"
        or config["output_directory"]
        != "runs/gravity/publication-readiness/manuscript-artifacts-v1"
        or config["receipt_path"]
        != "runs/gravity/publication-readiness/manuscript-artifact-manifest-v1.json"
    ):
        raise GravityClusterManuscriptRendererError("renderer identity changed")
    bindings = config["source_bindings"]
    if tuple(row.get("source_id") for row in bindings) != SOURCE_IDS:
        raise GravityClusterManuscriptRendererError("renderer source order changed")
    for row in bindings:
        _strict(
            row,
            {"source_id", "path", "file_sha256", "content_sha256"},
            "renderer source binding",
        )
        if len(str(row["file_sha256"])) != 64 or (
            row["content_sha256"] is not None and len(str(row["content_sha256"])) != 64
        ):
            raise GravityClusterManuscriptRendererError("renderer source hash changed")
    tables = config["primary_tables"]
    figures = config["primary_figures"]
    if (
        tuple(row.get("artifact_id") for row in tables) != TABLE_IDS
        or tuple(row.get("artifact_id") for row in figures) != FIGURE_IDS
    ):
        raise GravityClusterManuscriptRendererError("primary artifact inventory changed")
    filenames: list[str] = []
    for role, rows, suffix in (("table", tables, ".csv"), ("figure", figures, ".svg")):
        for row in rows:
            _strict(row, {"artifact_id", "filename", "title"}, f"primary {role}")
            filename = str(row["filename"])
            if Path(filename).name != filename or not filename.endswith(suffix):
                raise GravityClusterManuscriptRendererError(f"primary {role} filename changed")
            filenames.append(filename)
    if len(filenames) != len(set(filenames)):
        raise GravityClusterManuscriptRendererError("duplicate primary artifact filename")
    if config["rendering_contract"] != {
        "table_format": "RFC4180-compatible UTF-8 CSV with LF line endings",
        "figure_format": "standalone deterministic SVG 1.1",
        "figure_width_px": 960,
        "figure_height_px": 640,
        "numeric_significant_digits": 12,
        "font_family": "Arial, sans-serif",
        "external_fonts_or_assets": False,
        "timestamps_in_artifacts": False,
        "independent_target_access_allowed": False,
    }:
        raise GravityClusterManuscriptRendererError("rendering contract changed")


def _load_sources(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for binding in config["source_bindings"]:
        path = _under(root, str(binding["path"]), "renderer source")
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise GravityClusterManuscriptRendererError(
                f"renderer source file changed: {binding['path']}"
            )
        value = _read_json(path)
        expected_content = binding["content_sha256"]
        if expected_content is not None and _content_sha(value) != expected_content:
            raise GravityClusterManuscriptRendererError(
                f"renderer source content changed: {binding['path']}"
            )
        result[str(binding["source_id"])] = value
    evidence = result["manuscript_evidence_package"]
    if (
        evidence["counts"]["per_row_candidate_predictions"] != 233
        or evidence["access_ledger"]["independent_target_rows_opened"] != 0
        or evidence["access_ledger"]["independent_observational_authorization"] is not False
        or evidence["claims"]["bounded_paper_ready"] is not False
    ):
        raise GravityClusterManuscriptRendererError("development evidence boundary changed")
    return result


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isfinite(value):
            return format(value, ".12g")
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return str(value)


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([_format(value) for value in row])
    return stream.getvalue().encode("utf-8")


def _table_1(evidence: Mapping[str, Any], spec: Mapping[str, Any]) -> bytes:
    candidate = spec["candidate"]
    selected = evidence["comparators_and_ablations"]["candidate"]["selection"]
    rows: list[Sequence[Any]] = []
    for index, equation in enumerate(candidate["equations"], start=1):
        rows.append(("candidate", f"equation_{index}", equation))
    rows.extend(
        [
            ("candidate", "candidate_id", candidate["candidate_id"]),
            ("candidate", "kernel_scope", candidate["kernel_scope"]),
            ("candidate", "declared_origin", candidate["declared_origin"]),
            ("candidate", "historical_novelty_claim", candidate["historical_novelty_claim"]),
            ("selection", "parameters", selected["parameters"]),
            ("selection", "nuisances", selected["nuisances"]),
            ("selection", "evaluated_variants", selected["evaluated_variants"]),
            ("selection", "refit", selected["refit"]),
        ]
    )
    for key, value in sorted(evidence["claims"].items()):
        rows.append(("claim_boundary", key, value))
    return _csv_bytes(("section", "field", "value"), rows)


def _table_2(evidence: Mapping[str, Any]) -> bytes:
    rows = []
    for split in ("development_train", "development_holdout", "confirmation"):
        candidate = evidence["split_summaries"][split]["candidate"]
        rows.append(
            (
                split,
                candidate["rows"],
                candidate["cluster_observable_groups"],
                candidate["score"],
                candidate["median_absolute_log_residual"],
                candidate["root_mean_square_log_residual"],
                candidate["by_observable"]["pressure"],
                candidate["by_observable"]["temperature"],
            )
        )
    return _csv_bytes(
        (
            "split",
            "rows",
            "cluster_observable_groups",
            "aggregate_score",
            "median_absolute_log_residual",
            "root_mean_square_log_residual",
            "pressure_score",
            "temperature_score",
        ),
        rows,
    )


def _table_3(evidence: Mapping[str, Any]) -> bytes:
    section = evidence["comparators_and_ablations"]
    ranking = {
        row["model_id"]: index
        for index, row in enumerate(section["ranking"]["holdout_score_ascending"], start=1)
    }
    models: list[tuple[str, Mapping[str, Any]]] = [("candidate", section["candidate"])]
    models.extend(("comparator", value) for _, value in sorted(section["comparators"].items()))
    models.extend(("ablation", value) for _, value in sorted(section["ablations"].items()))
    rows = []
    for role, model in models:
        complexity = model["complexity"]
        rows.append(
            (
                role,
                model["model_id"],
                ranking.get(model["model_id"]),
                model["holdout"]["score"],
                model["holdout"]["by_observable"]["pressure"],
                model["holdout"]["by_observable"]["temperature"],
                complexity["information_criterion_k"],
                complexity["discrete_variants_screened"],
                complexity["conditional_boundary_observations"],
            )
        )
    return _csv_bytes(
        (
            "role",
            "model_id",
            "comparator_ranking_if_applicable",
            "holdout_score",
            "pressure_score",
            "temperature_score",
            "information_criterion_k",
            "discrete_variants_screened",
            "conditional_boundary_observations",
        ),
        rows,
    )


def _table_4(evidence: Mapping[str, Any]) -> bytes:
    ledger = evidence["object_and_counterexample_ledger"]
    wins = set(ledger["confirmation_cluster_wins"])
    counterexamples = set(ledger["confirmation_counterexamples"])
    rows = []
    for split in ("development_train", "development_holdout", "confirmation"):
        candidate = evidence["split_summaries"][split]["candidate"]
        for cluster in sorted(candidate["by_cluster"]):
            disposition = ""
            if split == "confirmation":
                disposition = (
                    "candidate_win"
                    if cluster in wins
                    else "counterexample"
                    if cluster in counterexamples
                    else "unclassified"
                )
            rows.append(
                (
                    split,
                    cluster,
                    candidate["by_cluster"][cluster],
                    candidate["by_cluster_observable"].get(f"{cluster}:pressure"),
                    candidate["by_cluster_observable"].get(f"{cluster}:temperature"),
                    disposition,
                )
            )
    return _csv_bytes(
        (
            "split",
            "cluster",
            "aggregate_score",
            "pressure_score",
            "temperature_score",
            "disposition",
        ),
        rows,
    )


def _table_5(evidence: Mapping[str, Any]) -> bytes:
    uncertainty = evidence["uncertainty_and_alternative_cause_boundary"]
    controls = evidence["negative_and_numerical_controls"]
    calibration = evidence["quotient_sampler_calibration_and_newtonian_boundary"]
    pressure = evidence["development_pressure_covariance_boundary"]
    ben = evidence["shared_ben_synthetic_and_real_boundary"]
    ben_v4 = evidence["shared_ben_development_executor_v4_boundary"]
    strata = evidence["cluster_strata_boundary"]
    shape_missing = evidence["predictor_shape_and_missing_variable_boundary"]
    acquisition = evidence["group_and_act_acquisition_boundary"]
    theory = evidence["matter_lensing_theory_boundary"]
    covariance = uncertainty["covariance_sensitivity"]
    missingness = uncertainty["missingness_sensitivity"]
    sampler = uncertainty["marginalization"]["candidate"]["posterior_sampler"]
    false_selection = controls["false_selection"]
    implementation = controls["implementation_agreement"]
    power = controls["prospective_power_and_stopping"]
    rows = [
        (
            "covariance",
            "scenarios",
            len(covariance["scenarios"]),
            "stress models, not source covariance",
        ),
        (
            "covariance",
            "candidate_beats_nfw_scenarios",
            covariance["candidate_beats_nfw_scenarios"],
            "",
        ),
        ("covariance", "candidate_score_range", covariance["candidate_score_range"], ""),
        ("missingness", "scenarios", len(missingness["scenarios"]), missingness["status"]),
        ("missingness", "score_range", missingness["score_range"], ""),
        ("marginalization", "candidate_sampler_converged", sampler["converged"], ""),
        ("marginalization", "candidate_maximum_rhat", sampler["maximum_rhat"], ""),
        (
            "marginalization",
            "candidate_minimum_effective_samples",
            sampler["minimum_effective_samples"],
            "",
        ),
        ("false_selection", "trials", false_selection["trials"], ""),
        ("false_selection", "search_variants", false_selection["search_variants"], ""),
        ("false_selection", "fraction", false_selection["false_selection_fraction"], ""),
        ("false_selection", "wilson_95_percent", false_selection["wilson_95_percent"], ""),
        ("false_selection", "threshold", false_selection["threshold"], ""),
        ("implementation", "gpu_device", implementation["gpu_device"], ""),
        (
            "implementation",
            "cpu_gpu_maximum_absolute_score_difference",
            implementation["cpu_gpu_maximum_absolute_score_difference"],
            "",
        ),
        (
            "implementation",
            "direct_scorer_maximum_absolute_difference",
            implementation["separate_direct_scorer_maximum_absolute_difference"],
            "",
        ),
        (
            "power",
            "planned_independent_clusters",
            power["planned_independent_clusters"],
            "exploratory and underpowered",
        ),
        ("power", "planned_approximate_power", power["planned_approximate_power"], ""),
        (
            "power",
            "calculated_required_clusters",
            power["calculated_required_clusters"],
            "frozen confirmatory target",
        ),
        ("power", "target_power", power["target_power"], ""),
        ("quotient_sbc", "v1_passed", calibration["v1_passed"], "retained failure"),
        (
            "quotient_sbc",
            "v2_passed",
            calibration["v2_passed"],
            "independent reference fixed; candidate mixing failed",
        ),
        (
            "quotient_sbc",
            "v3_synthetic_sbc_passed",
            calibration["v3_synthetic_sbc_passed"],
            "synthetic calibration only",
        ),
        (
            "quotient_sbc",
            "candidate_production_unlock",
            calibration["candidate_production_unlock"],
            "",
        ),
        (
            "newtonian_control",
            "external_approval_present",
            calibration["newtonian_external_approval_present"],
            "",
        ),
        (
            "newtonian_control",
            "production_runs",
            calibration["newtonian_production_runs"],
            "",
        ),
        (
            "newtonian_control",
            "requested_likelihood_evaluations",
            calibration["newtonian_requested_likelihood_evaluations"],
            "$0 external cost; not executed",
        ),
        (
            "pressure_covariance",
            "reconstructed_matrices",
            pressure["reconstructed_matrices"],
            "development only",
        ),
        (
            "pressure_covariance",
            "scored_pressure_rows",
            pressure["scored_pressure_rows"],
            "",
        ),
        (
            "pressure_covariance",
            "scoring_decision",
            pressure["scoring_decision"],
            "4/8 full-covariance cluster wins versus 6/8 required",
        ),
        (
            "covariance_sources",
            "a1795_complete_source_packet",
            pressure["a1795_complete_source_packet"],
            "CP5.2-CP5.6 remain blocked",
        ),
        (
            "shared_ben_synthetic",
            "raw_candidates",
            ben["synthetic_raw_candidates"],
            "synthetic plumbing only",
        ),
        (
            "shared_ben_synthetic",
            "equivalence_classes",
            ben["synthetic_equivalence_classes"],
            "AST-deduplicated before synthetic responses",
        ),
        (
            "shared_ben_real",
            "real_scoring_executed",
            ben["v2_real_scoring_executed"],
            "blocked before payload load",
        ),
        (
            "shared_ben_v4",
            "canonical_full_classes",
            ben_v4["canonical_full_classes"],
            f"registered ablations={ben_v4['registered_ablations']}; unique ASTs={ben_v4['unique_asts_across_full_and_ablations']}",
        ),
        (
            "shared_ben_v4",
            "production_executed",
            ben_v4["production_executed"],
            f"target rows={ben_v4['target_rows_read']}; scores={ben_v4['scores_computed']}",
        ),
        (
            "shared_ben_v4",
            "comparison_operator",
            ben_v4["comparison_operator"],
            "validated reference runtime; indifference band; not a fully frozen runtime",
        ),
        (
            "cluster_strata",
            "candidate_full_covariance_score",
            strata["candidate_full_covariance_score"],
            "frozen absolute maximum=1",
        ),
        (
            "cluster_strata",
            "candidate_absolute_gate_passed",
            strata["candidate_absolute_gate_passed"],
            "exploratory development only",
        ),
        (
            "cluster_strata",
            "candidate_cluster_wins",
            strata["candidate_cluster_wins"],
            f"minimum={strata['minimum_cluster_wins']}",
        ),
        (
            "cluster_strata",
            "candidate_object_win_gate_passed",
            strata["candidate_object_win_gate_passed"],
            "4 of 8 clusters",
        ),
        (
            "cluster_strata",
            "frozen_stratum_explains_covariance_flips",
            strata["frozen_stratum_explains_covariance_flips"],
            "no Holm-significant explanation",
        ),
        (
            "xcop_shape_bridge",
            "real_scoring_executed",
            shape_missing["shape_real_scoring_executed"],
            "predictor-only, unauthorized",
        ),
        (
            "missing_variables",
            "defined_proxy_contracts",
            shape_missing["defined_proxy_contracts"],
            "categorical/projected proxies only",
        ),
        (
            "missing_variables",
            "continuous_measurement_ready_rows",
            shape_missing["continuous_measurement_ready_rows"],
            f"source-blocked applicable rows={shape_missing['source_blocked_applicable_rows']}",
        ),
        (
            "group_acquisition",
            "scientific_payload_rows_opened",
            acquisition["group_scientific_payload_rows_opened"],
            "metadata manifest only",
        ),
        (
            "act_erass_overlap",
            "population_gate_evaluated",
            acquisition["act_population_gate_evaluated"],
            "catalog rows unauthorized",
        ),
        (
            "act_erass_executor",
            "authorized",
            acquisition["act_executor_authorized"],
            "guarded executor frozen; no run",
        ),
        (
            "act_erass_executor",
            "catalog_rows_opened",
            acquisition["act_executor_catalog_rows_opened"],
            f"network calls={acquisition['act_executor_network_calls']}",
        ),
        (
            "group_source_v3",
            "ready_science_lanes",
            evidence["group_scale_source_boundary"]["v3_ready_lanes"],
            f"of {evidence['group_scale_source_boundary']['v3_candidate_lanes']}; X-CLASS preferred, eFEDS backup",
        ),
        (
            "xclass_identity_executor",
            "authorized",
            acquisition["xclass_executor_authorized"],
            "one 16,895-byte GET prepared; no run",
        ),
        (
            "xclass_identity_executor",
            "identity_rows_opened",
            acquisition["xclass_executor_identity_rows"],
            f"ObsID mapping={acquisition['xclass_executor_obsid_mapping_available']}; X-COP overlap={acquisition['xclass_executor_xcop_overlap_known']}",
        ),
        (
            "matter_lensing_theory",
            "template_level_gates_passed",
            theory["template_level_gates_passed"],
            f"of {theory['health_gates_total']}; blocked={theory['health_gates_blocked']}",
        ),
        (
            "matter_lensing_symbolic",
            "full_H2_passed",
            theory["full_H2_passed"],
            "bounded scalar identities only",
        ),
        (
            "matter_lensing_external_symbol",
            "H3_scalar_external_metric",
            theory["H3_scalar_external_metric"],
            "partial constant-local-jet result",
        ),
        (
            "matter_lensing_external_symbol",
            "H4_constant_coefficient",
            theory["H4_constant_coefficient"],
            "partial; metric constraints absent",
        ),
        (
            "matter_lensing_external_symbol",
            "designed_u_above_one_third_failure_preserved",
            theory["designed_u_above_one_third_failure_preserved"],
            "negative determinant contribution retained",
        ),
        (
            "matter_lensing_kinetic_gate",
            "conditional_timelike_mixing_no_go",
            theory["conditional_timelike_mixing_no_go"],
            "scope-restricted theorem; not a full-action no-go",
        ),
        (
            "matter_lensing_kinetic_gate",
            "bounded_domain_nonnegative_examples_exist",
            theory["bounded_domain_nonnegative_examples_exist"],
            f"observational files opened={theory['kinetic_gate_observational_files_opened']}",
        ),
        (
            "matter_lensing_source_bound",
            "restricted_static_source_bound_established",
            theory["restricted_static_source_bound_established"],
            f"physical source law={theory['physical_source_law_established']}; on-shell={theory['physical_on_shell_solution_established']}",
        ),
        (
            "matter_lensing_conformal_source",
            "universal_conformal_source_identity_established",
            theory["universal_conformal_source_identity_established"],
            f"physical profile={theory['physical_source_profile_established']}; metric backreaction={theory['metric_backreaction_established']}",
        ),
        (
            "matter_lensing_solar_gw",
            "necessary_conditions_established",
            theory["solar_necessary_conditions_established"],
            f"Solar gate={theory['solar_gate_passed']}; GW gate={theory['gw_gate_passed']}",
        ),
        (
            "matter_lensing_flrw",
            "restricted_flat_flrw_equations_established",
            theory["restricted_flat_flrw_equations_established"],
            f"healthy history={theory['healthy_late_time_history_exists']}; fit={theory['cosmological_fit_performed']}",
        ),
        (
            "matter_lensing_covariant",
            "scalar_stress_and_exchange_established",
            theory["covariant_scalar_stress_and_exchange_established"],
            f"full H2={theory['covariant_full_H2']}; ADM constraints={theory['covariant_ADM_constraints_derived']}",
        ),
        (
            "matter_lensing_adm_constraints",
            "CP11_3_complete",
            theory["CP11_3_complete"],
            f"standard ADM representative only={theory['standard_adm_representative_only']}; full-system hyperbolicity={theory['full_metric_scalar_matter_system_strongly_hyperbolic']}",
        ),
        (
            "matter_lensing_scalar_hamiltonian",
            "restricted_scalar_canonical_hamiltonian_derived",
            theory["restricted_scalar_canonical_hamiltonian_derived"],
            f"CP11.4={theory['scalar_hamiltonian_CP11_4_complete']}; positive Hamiltonian={theory['scalar_physical_hamiltonian_positive']}",
        ),
        (
            "matter_lensing_deep_aqual_transition",
            "conditional_exact_transition_no_go_established",
            theory["conditional_exact_transition_no_go_established"],
            f"exact C2={theory['exact_deep_aqual_transition_is_C2']}; regulated subluminal={theory['regulated_example_is_subluminal_relative_to_conformal_matter_cone']}",
        ),
        (
            "shared_formula_scalar_kinetic_reconstruction",
            "formula_to_minimal_kinetic_map_derived",
            theory["formula_to_minimal_kinetic_map_derived"],
            f"source-only={theory['formula_source_only_classes']}/60; quadrature causal={theory['quadrature_minimal_map_causal_relative_to_conformal_matter_cone']}; full covariant bridge={theory['formula_full_covariant_bridge_derived']}",
        ),
        (
            "shared_quadrature_covariant_action",
            "restricted_quadrature_action_embedding_established",
            theory["restricted_quadrature_action_defined"],
            f"exact motion={theory['quadrature_motion_law_recovered_exactly']}; direct conformal cancellation={theory['quadrature_direct_conformal_lensing_shift_cancels']}; quantitative lensing={theory['quadrature_quantitative_lensing_solution_derived']}; causal={theory['quadrature_scalar_cone_causal']}",
        ),
        (
            "shared_quadrature_lensing_backreaction",
            "restricted_exterior_lensing_backreaction_derived",
            theory["quadrature_restricted_lensing_backreaction_derived"],
            f"stress source={theory['quadrature_scalar_stress_lensing_source_nonzero']}; compactness suppressed={theory['quadrature_lensing_backreaction_compactness_suppressed']}; asymptotic match={theory['quadrature_asymptotic_motion_lensing_match']}; global lensing={theory['quadrature_global_quantitative_lensing_success']}",
        ),
        (
            "shared_quadrature_universal_vector_metric",
            "restricted_same_metric_motion_lensing_architecture",
            theory["quadrature_vector_metric_same_action_architecture"],
            f"universal metric={theory['quadrature_vector_metric_universal_matter_photon_metric']}; photon adjustment={theory['quadrature_vector_metric_separate_photon_adjustment']}; leading relation={theory['quadrature_vector_metric_leading_motion_lensing_relation']}; fixed-aether causal={theory['quadrature_vector_metric_fixed_aether_scalar_causal']}; full causal={theory['quadrature_vector_metric_full_causality']}; quantitative lensing={theory['quadrature_vector_metric_quantitative_lensing']}",
        ),
        (
            "shared_quadrature_aether_mode_conditions",
            "finite_positive_all_luminal_pure_aether_locus",
            theory["quadrature_aether_finite_luminal_locus_exists"],
            f"five-mode formulas={theory['quadrature_aether_five_mode_formulas_rechecked']}; exact GW+PPN zero regular={theory['quadrature_aether_exact_gw_ppn_zero_regular']}; uniform kinetic margin={theory['quadrature_aether_uniform_kinetic_margin']}; full coupled health={theory['quadrature_aether_full_coupled_health']}",
        ),
        (
            "shared_quadrature_reduced_principal_factorization",
            "static_branch_six_mode_reduced_factorization",
            theory["quadrature_reduced_six_mode_factorization"],
            f"local causal={theory['quadrature_reduced_six_mode_local_causality']}; principal scalar mixing={theory['quadrature_reduced_principal_scalar_mixing_present']}; nonzero-W factorization={theory['quadrature_reduced_nonzero_W_factorization']}; unreduced hyperbolicity={theory['quadrature_reduced_unreduced_constraint_hyperbolicity']}; healthy action={theory['quadrature_reduced_healthy_action']}",
        ),
        (
            "shared_quadrature_combined_tetrad_hyperbolicity",
            "restricted_W_zero_combined_symmetric_hyperbolicity",
            theory["quadrature_combined_symmetric_hyperbolicity"],
            f"common Cauchy time={theory['quadrature_combined_common_Cauchy_time']}; aether necessary bounds={theory['quadrature_combined_aether_necessary_bounds']}; all-mode Cherenkov safety={theory['quadrature_combined_all_mode_cherenkov_safety']}; full health={theory['quadrature_combined_full_health']}",
        ),
    ]
    for injection in controls["synthetic_recovery"]["injections"]:
        rows.append(
            (
                "synthetic_recovery",
                injection["injection_id"],
                injection["class_recovered"],
                f"selected={injection['selected_class']}",
            )
        )
    return _csv_bytes(("section", "metric", "value", "note"), rows)


def _table_6(evidence: Mapping[str, Any]) -> bytes:
    rows = []
    for key, value in sorted(evidence["access_ledger"].items()):
        rows.append(("access_ledger", key, value))
    for track, value in sorted(evidence["claim_tracks"].items()):
        rows.append(("claim_track", track, value))
    for key, value in sorted(evidence["claims"].items()):
        rows.append(("claim_boundary", key, value))
    for key, value in sorted(
        evidence["quotient_sampler_calibration_and_newtonian_boundary"].items()
    ):
        rows.append(("sampler_calibration_boundary", key, value))
    for key, value in sorted(evidence["development_pressure_covariance_boundary"].items()):
        rows.append(("pressure_covariance_boundary", key, value))
    for section, source in (
        ("shared_ben_boundary", evidence["shared_ben_synthetic_and_real_boundary"]),
        (
            "shared_ben_v4_boundary",
            evidence["shared_ben_development_executor_v4_boundary"],
        ),
        (
            "shape_missing_variable_boundary",
            evidence["predictor_shape_and_missing_variable_boundary"],
        ),
        ("group_scale_source_boundary", evidence["group_scale_source_boundary"]),
        ("group_act_acquisition_boundary", evidence["group_and_act_acquisition_boundary"]),
        ("cluster_strata_boundary", evidence["cluster_strata_boundary"]),
        ("matter_lensing_theory_boundary", evidence["matter_lensing_theory_boundary"]),
    ):
        for key, value in sorted(source.items()):
            rows.append((section, key, value))
    for index, limitation in enumerate(evidence["limitations"], start=1):
        rows.append(("limitation", f"limitation_{index}", limitation))
    return _csv_bytes(("section", "field", "value"), rows)


def _table_7(evidence: Mapping[str, Any], spec: Mapping[str, Any]) -> bytes:
    adjudication = evidence["prior_art_boundary"]["candidate_adjudication"]
    rows = []
    for source in spec["retained_primary_sources"]:
        rows.append(
            (
                "source",
                source["source_id"],
                source["title"],
                source["relationship_label"],
                source["equation_anchor"],
                source["doi"],
                source["url"],
            )
        )
    for key, value in sorted(adjudication.items()):
        rows.append(("adjudication", key, _format(value), "", "", "", ""))
    return _csv_bytes(
        (
            "row_type",
            "source_or_field",
            "title_or_value",
            "relationship",
            "equation_anchor",
            "doi",
            "url",
        ),
        rows,
    )


def _n(value: float) -> str:
    return format(value, ".3f").rstrip("0").rstrip(".")


def _text(
    x: float, y: float, value: Any, size: int = 13, anchor: str = "start", **attrs: Any
) -> str:
    extra = " ".join(
        f'{key.replace("_", "-")}="{escape(str(item))}"' for key, item in attrs.items()
    )
    return (
        f'<text x="{_n(x)}" y="{_n(y)}" font-size="{size}" text-anchor="{anchor}"'
        f"{(' ' + extra) if extra else ''}>{escape(str(value))}</text>"
    )


def _svg(title: str, subtitle: str, parts: Sequence[str]) -> bytes:
    content = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="960" height="640" viewBox="0 0 960 640">',
        '<rect width="960" height="640" fill="#ffffff"/>',
        '<g font-family="Arial, sans-serif" fill="#17212b">',
        _text(40, 40, title, 22, font_weight="700"),
        _text(40, 66, subtitle, 12, fill="#52606d"),
        *parts,
        "</g>",
        "</svg>",
        "",
    ]
    return "\n".join(content).encode("utf-8")


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    if high <= low:
        return [low]
    return [low + index * (high - low) / (count - 1) for index in range(count)]


def _scatter_panel(
    rows: Sequence[Mapping[str, Any]],
    observable: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> list[str]:
    selected = [row for row in rows if row["observable"] == observable]
    x_values = [math.log10(float(row["observed"])) for row in selected]
    y_values = [math.log10(float(row["predicted"])) for row in selected]
    low = min(x_values + y_values)
    high = max(x_values + y_values)
    padding = max(0.08, 0.06 * (high - low))
    low -= padding
    high += padding
    sx = lambda value: x0 + (value - low) / (high - low) * width
    sy = lambda value: y0 + height - (value - low) / (high - low) * height
    parts = [
        _text(x0 + width / 2, y0 - 18, observable.capitalize(), 16, "middle", font_weight="700")
    ]
    for tick in _ticks(low, high):
        px = sx(tick)
        py = sy(tick)
        parts.extend(
            [
                f'<line x1="{_n(px)}" y1="{_n(y0)}" x2="{_n(px)}" y2="{_n(y0 + height)}" stroke="#e7ebef"/>',
                f'<line x1="{_n(x0)}" y1="{_n(py)}" x2="{_n(x0 + width)}" y2="{_n(py)}" stroke="#e7ebef"/>',
                _text(px, y0 + height + 20, f"10^{tick:.1f}", 10, "middle", fill="#52606d"),
                _text(x0 - 8, py + 4, f"10^{tick:.1f}", 10, "end", fill="#52606d"),
            ]
        )
    parts.append(
        f'<line x1="{_n(sx(low))}" y1="{_n(sy(low))}" x2="{_n(sx(high))}" y2="{_n(sy(high))}" stroke="#17212b" stroke-dasharray="5 4"/>'
    )
    for row in selected:
        parts.append(
            f'<circle cx="{_n(sx(math.log10(float(row["observed"]))))}" cy="{_n(sy(math.log10(float(row["predicted"]))))}" r="3" fill="{SPLIT_COLORS[row["split"]]}" fill-opacity="0.72"/>'
        )
    parts.extend(
        [
            f'<rect x="{_n(x0)}" y="{_n(y0)}" width="{_n(width)}" height="{_n(height)}" fill="none" stroke="#52606d"/>',
            _text(x0 + width / 2, y0 + height + 46, "Observed", 12, "middle"),
        ]
    )
    return parts


def _figure_1(evidence: Mapping[str, Any]) -> bytes:
    rows = evidence["per_row_candidate_predictions"]
    parts = [
        *_scatter_panel(rows, "pressure", 80, 125, 340, 390),
        *_scatter_panel(rows, "temperature", 550, 125, 340, 390),
    ]
    parts.append(_text(18, 325, "Predicted", 12, "middle", transform="rotate(-90 18 325)"))
    x = 260
    for split in ("development_train", "development_holdout", "confirmation"):
        parts.append(f'<circle cx="{x}" cy="600" r="5" fill="{SPLIT_COLORS[split]}"/>')
        parts.append(_text(x + 10, 604, split.replace("_", " "), 11))
        x += 205
    return _svg(
        "Figure 1. Observed versus predicted profiles",
        "Frozen Item 59 candidate; logarithmic axes; diagonal is exact agreement.",
        parts,
    )


def _figure_2(evidence: Mapping[str, Any]) -> bytes:
    ranking = evidence["comparators_and_ablations"]["ranking"]["holdout_score_ascending"]
    values = [float(row["score"]) for row in ranking]
    low = math.log10(min(values) * 0.8)
    high = math.log10(max(values) * 1.25)
    x0, y0, width, row_height = 330.0, 110.0, 570.0, 62.0
    parts: list[str] = []
    for tick in _ticks(low, high, 6):
        x = x0 + (tick - low) / (high - low) * width
        parts.append(f'<line x1="{_n(x)}" y1="90" x2="{_n(x)}" y2="545" stroke="#e7ebef"/>')
        parts.append(_text(x, 570, f"10^{tick:.1f}", 10, "middle", fill="#52606d"))
    for index, row in enumerate(ranking):
        y = y0 + index * row_height
        score = float(row["score"])
        bar_width = (math.log10(score) - low) / (high - low) * width
        color = "#087e8b" if row["model_id"] == "ITEM59_CROSS_SCALE_BOUNDARY" else "#9aa5b1"
        parts.extend(
            [
                _text(x0 - 12, y + 21, row["model_id"].replace("_", " "), 11, "end"),
                f'<rect x="{_n(x0)}" y="{_n(y)}" width="{_n(bar_width)}" height="28" rx="3" fill="{color}"/>',
                _text(x0 + bar_width + 8, y + 20, f"{score:.3f}", 11),
            ]
        )
    parts.append(
        _text(
            x0 + width / 2,
            610,
            "Equal-cluster/equal-observable holdout score (lower is better; log scale)",
            12,
            "middle",
        )
    )
    return _svg(
        "Figure 2. Matched development-holdout comparison",
        "Absolute scores are shown; relative improvement alone is not a publication claim.",
        parts,
    )


def _heat_color(value: float, low: float, high: float) -> str:
    fraction = 0.5 if high <= low else (math.log10(value) - low) / (high - low)
    fraction = max(0.0, min(1.0, fraction))
    start = (232, 247, 249)
    end = (8, 126, 139)
    rgb = tuple(round(a + fraction * (b - a)) for a, b in zip(start, end, strict=True))
    return "#" + "".join(f"{component:02x}" for component in rgb)


def _figure_3(evidence: Mapping[str, Any]) -> bytes:
    data = evidence["split_summaries"]["development_holdout"]["candidate"]["by_cluster_observable"]
    clusters = sorted({key.split(":", 1)[0] for key in data})
    observables = ("pressure", "temperature")
    values = [float(value) for value in data.values()]
    low, high = math.log10(min(values)), math.log10(max(values))
    x0, y0, cell_width, cell_height = 260.0, 110.0, 280.0, 52.0
    parts: list[str] = []
    for column, observable in enumerate(observables):
        parts.append(
            _text(
                x0 + (column + 0.5) * cell_width,
                y0 - 18,
                observable.capitalize(),
                14,
                "middle",
                font_weight="700",
            )
        )
    for row_index, cluster in enumerate(clusters):
        y = y0 + row_index * cell_height
        parts.append(_text(x0 - 14, y + 32, cluster, 12, "end"))
        for column, observable in enumerate(observables):
            value = float(data[f"{cluster}:{observable}"])
            x = x0 + column * cell_width
            color = _heat_color(value, low, high)
            text_color = "#ffffff" if math.log10(value) > (low + high) / 2 else "#17212b"
            parts.extend(
                [
                    f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(cell_width)}" height="{_n(cell_height)}" fill="{color}" stroke="#ffffff"/>',
                    _text(
                        x + cell_width / 2, y + 32, f"{value:.3f}", 12, "middle", fill=text_color
                    ),
                ]
            )
    legend_y = 560
    for index in range(101):
        value = 10 ** (low + index / 100 * (high - low))
        parts.append(
            f'<rect x="{260 + 5 * index}" y="{legend_y}" width="5" height="16" fill="{_heat_color(value, low, high)}"/>'
        )
    parts.extend(
        [
            _text(260, 595, f"{min(values):.3g}", 10),
            _text(765, 595, f"{max(values):.3g}", 10, "end"),
            _text(512, 595, "score (log color scale; lower is better)", 11, "middle"),
        ]
    )
    return _svg(
        "Figure 3. Cluster-level holdout performance",
        "Every cluster and observable is retained; a single counterexample is evidence, not an automatic family veto.",
        parts,
    )


def _figure_4(evidence: Mapping[str, Any]) -> bytes:
    scenarios = evidence["uncertainty_and_alternative_cause_boundary"]["covariance_sensitivity"][
        "scenarios"
    ]
    x_values = [math.log10(float(row["candidate_score"])) for row in scenarios]
    y_values = [math.log10(float(row["nfw_score"])) for row in scenarios]
    low = min(x_values + y_values) - 0.08
    high = max(x_values + y_values) + 0.08
    x0, y0, size = 155.0, 105.0, 440.0
    sx = lambda value: x0 + (value - low) / (high - low) * size
    sy = lambda value: y0 + size - (value - low) / (high - low) * size
    palette = {1.0: "#087e8b", 1.5: "#ff9f1c", 2.0: "#ff5a5f"}
    parts: list[str] = []
    for tick in _ticks(low, high):
        x, y = sx(tick), sy(tick)
        parts.extend(
            [
                f'<line x1="{_n(x)}" y1="{y0}" x2="{_n(x)}" y2="{y0 + size}" stroke="#e7ebef"/>',
                f'<line x1="{x0}" y1="{_n(y)}" x2="{x0 + size}" y2="{_n(y)}" stroke="#e7ebef"/>',
                _text(x, y0 + size + 22, f"10^{tick:.1f}", 10, "middle"),
                _text(x0 - 10, y + 4, f"10^{tick:.1f}", 10, "end"),
            ]
        )
    parts.append(
        f'<line x1="{_n(sx(low))}" y1="{_n(sy(low))}" x2="{_n(sx(high))}" y2="{_n(sy(high))}" stroke="#17212b" stroke-dasharray="6 4"/>'
    )
    for row in scenarios:
        inflation = float(row["diagonal_error_inflation"])
        parts.append(
            f'<circle cx="{_n(sx(math.log10(float(row["candidate_score"]))))}" cy="{_n(sy(math.log10(float(row["nfw_score"]))))}" r="5" fill="{palette[inflation]}" fill-opacity="0.72"/>'
        )
    parts.extend(
        [
            f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" fill="none" stroke="#52606d"/>',
            _text(x0 + size / 2, 595, "Candidate score (log scale)", 12, "middle"),
            _text(
                75,
                y0 + size / 2,
                "NFW score (log scale)",
                12,
                "middle",
                transform=f"rotate(-90 75 {y0 + size / 2})",
            ),
            _text(650, 140, "Points above diagonal favor candidate", 13, font_weight="700"),
            _text(650, 166, "36/36 tested stress scenarios", 13),
            _text(650, 205, "Diagonal error inflation", 12, font_weight="700"),
        ]
    )
    y = 235
    for inflation, color in palette.items():
        parts.append(f'<circle cx="670" cy="{y}" r="6" fill="{color}"/>')
        parts.append(_text(686, y + 4, f"{inflation:.1f}x", 12))
        y += 30
    parts.extend(
        [
            _text(650, 360, "Important boundary", 12, font_weight="700"),
            _text(650, 386, "These are synthetic covariance stress", 11),
            _text(650, 405, "models, not released instrument/source", 11),
            _text(650, 424, "covariance matrices.", 11),
        ]
    )
    return _svg(
        "Figure 4. Covariance sensitivity",
        "Development holdout only; shared calibration and radial correlation were varied without target-row access.",
        parts,
    )


def _residual_panel(
    rows: Sequence[Mapping[str, Any]],
    observable: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> list[str]:
    selected = [row for row in rows if row["observable"] == observable]
    xs = [math.log10(float(row["radius_kpc"])) for row in selected]
    ys = [float(row["log_residual"]) for row in selected]
    x_low, x_high = min(xs), max(xs)
    y_abs = max(abs(min(ys)), abs(max(ys))) * 1.08
    sx = lambda value: x0 + (value - x_low) / (x_high - x_low) * width
    sy = lambda value: y0 + height / 2 - value / (2 * y_abs) * height
    parts = [
        _text(x0 + width / 2, y0 - 18, observable.capitalize(), 16, "middle", font_weight="700")
    ]
    for tick in _ticks(x_low, x_high):
        x = sx(tick)
        parts.append(
            f'<line x1="{_n(x)}" y1="{y0}" x2="{_n(x)}" y2="{y0 + height}" stroke="#e7ebef"/>'
        )
        parts.append(_text(x, y0 + height + 20, f"10^{tick:.1f}", 10, "middle"))
    for tick in _ticks(-y_abs, y_abs):
        y = sy(tick)
        parts.append(
            f'<line x1="{x0}" y1="{_n(y)}" x2="{x0 + width}" y2="{_n(y)}" stroke="#e7ebef"/>'
        )
        parts.append(_text(x0 - 8, y + 4, f"{tick:.2f}", 10, "end"))
    parts.append(
        f'<line x1="{x0}" y1="{_n(sy(0))}" x2="{x0 + width}" y2="{_n(sy(0))}" stroke="#17212b" stroke-width="1.5"/>'
    )
    for row in selected:
        parts.append(
            f'<circle cx="{_n(sx(math.log10(float(row["radius_kpc"]))))}" cy="{_n(sy(float(row["log_residual"])))}" r="3" fill="{SPLIT_COLORS[row["split"]]}" fill-opacity="0.72"/>'
        )
    parts.extend(
        [
            f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="none" stroke="#52606d"/>',
            _text(x0 + width / 2, y0 + height + 46, "Radius (kpc; log scale)", 12, "middle"),
        ]
    )
    return parts


def _figure_5(evidence: Mapping[str, Any]) -> bytes:
    rows = evidence["per_row_candidate_predictions"]
    parts = [
        *_residual_panel(rows, "pressure", 80, 125, 340, 390),
        *_residual_panel(rows, "temperature", 550, 125, 340, 390),
    ]
    parts.append(
        _text(18, 325, "log(observed / predicted)", 12, "middle", transform="rotate(-90 18 325)")
    )
    x = 260
    for split in ("development_train", "development_holdout", "confirmation"):
        parts.append(f'<circle cx="{x}" cy="600" r="5" fill="{SPLIT_COLORS[split]}"/>')
        parts.append(_text(x + 10, 604, split.replace("_", " "), 11))
        x += 205
    return _svg(
        "Figure 5. Radial residual structure",
        "All 233 scored rows are shown; the outer unscored pressure boundary is not represented as a prediction.",
        parts,
    )


def _figure_6(evidence: Mapping[str, Any]) -> bytes:
    controls = evidence["negative_and_numerical_controls"]
    false_selection = controls["false_selection"]
    power = controls["prospective_power_and_stopping"]
    parts: list[str] = [
        _text(250, 115, "False-selection control", 16, "middle", font_weight="700"),
        _text(710, 115, "Prospective replication power", 16, "middle", font_weight="700"),
    ]
    x0, y0, width, height = 100.0, 165.0, 300.0, 330.0
    for percent in range(6):
        y = y0 + height - percent / 5 * height
        parts.append(
            f'<line x1="{x0}" y1="{_n(y)}" x2="{x0 + width}" y2="{_n(y)}" stroke="#e7ebef"/>'
        )
        parts.append(_text(x0 - 8, y + 4, f"{percent}%", 10, "end"))
    fraction = float(false_selection["false_selection_fraction"])
    bar_height = fraction / 0.05 * height
    bar_x = x0 + 100
    parts.extend(
        [
            f'<rect x="{bar_x}" y="{_n(y0 + height - bar_height)}" width="100" height="{_n(bar_height)}" fill="#087e8b"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0 + width}" y2="{y0}" stroke="#ff5a5f" stroke-width="2" stroke-dasharray="6 4"/>',
            _text(
                bar_x + 50,
                y0 + height - bar_height - 10,
                f"{100 * fraction:.3f}%",
                13,
                "middle",
                font_weight="700",
            ),
            _text(
                x0 + width / 2, 530, f"70 / {false_selection['trials']} null trials", 11, "middle"
            ),
            _text(
                x0 + width / 2,
                550,
                "red dashed line: frozen 5% threshold",
                10,
                "middle",
                fill="#52606d",
            ),
        ]
    )
    x0, y0, width, height = 570.0, 165.0, 280.0, 330.0
    for power_tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y0 + height - power_tick * height
        parts.append(
            f'<line x1="{x0}" y1="{_n(y)}" x2="{x0 + width}" y2="{_n(y)}" stroke="#e7ebef"/>'
        )
        parts.append(_text(x0 - 8, y + 4, f"{power_tick:.0%}", 10, "end"))
    target = float(power["target_power"])
    planned = float(power["planned_approximate_power"])
    parts.append(
        f'<line x1="{x0}" y1="{_n(y0 + height - target * height)}" x2="{x0 + width}" y2="{_n(y0 + height - target * height)}" stroke="#ff5a5f" stroke-width="2" stroke-dasharray="6 4"/>'
    )
    for index, (clusters, value, color) in enumerate(
        (
            (power["planned_independent_clusters"], planned, "#9aa5b1"),
            (power["calculated_required_clusters"], target, "#087e8b"),
        )
    ):
        x = x0 + 45 + index * 140
        bar_height = value * height
        parts.append(
            f'<rect x="{x}" y="{_n(y0 + height - bar_height)}" width="70" height="{_n(bar_height)}" fill="{color}"/>'
        )
        parts.append(
            _text(
                x + 35,
                y0 + height - bar_height - 10,
                f"{value:.1%}",
                12,
                "middle",
                font_weight="700",
            )
        )
        parts.append(_text(x + 35, 525, f"n={clusters}", 12, "middle"))
    parts.extend(
        [
            _text(710, 550, "192 is the frozen confirmatory target", 10, "middle", fill="#52606d"),
            _text(
                480,
                607,
                "Controls reduce false discovery risk; they do not create independent evidence.",
                12,
                "middle",
                font_weight="700",
            ),
        ]
    )
    return _svg(
        "Figure 6. Search calibration and replication design",
        "False-selection is measured on Newtonian nulls; power is prospective and based on development effects.",
        parts,
    )


def build_artifacts(root: Path) -> dict[str, bytes]:
    root = root.resolve()
    config = load_config(root)
    sources = _load_sources(root, config)
    evidence = sources["manuscript_evidence_package"]
    spec = sources["candidate_specification"]
    table_builders = {
        TABLE_IDS[0]: lambda: _table_1(evidence, spec),
        TABLE_IDS[1]: lambda: _table_2(evidence),
        TABLE_IDS[2]: lambda: _table_3(evidence),
        TABLE_IDS[3]: lambda: _table_4(evidence),
        TABLE_IDS[4]: lambda: _table_5(evidence),
        TABLE_IDS[5]: lambda: _table_6(evidence),
        TABLE_IDS[6]: lambda: _table_7(evidence, spec),
    }
    figure_builders = {
        FIGURE_IDS[0]: lambda: _figure_1(evidence),
        FIGURE_IDS[1]: lambda: _figure_2(evidence),
        FIGURE_IDS[2]: lambda: _figure_3(evidence),
        FIGURE_IDS[3]: lambda: _figure_4(evidence),
        FIGURE_IDS[4]: lambda: _figure_5(evidence),
        FIGURE_IDS[5]: lambda: _figure_6(evidence),
    }
    artifacts = {}
    for row in config["primary_tables"]:
        artifacts[str(row["filename"])] = table_builders[str(row["artifact_id"])]()
    for row in config["primary_figures"]:
        artifacts[str(row["filename"])] = figure_builders[str(row["artifact_id"])]()
    return artifacts


def build_receipt(root: Path, artifacts: Mapping[str, bytes] | None = None) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sources = _load_sources(root, config)
    evidence = sources["manuscript_evidence_package"]
    rendered = dict(artifacts if artifacts is not None else build_artifacts(root))
    metadata = {row["filename"]: ("table", row) for row in config["primary_tables"]} | {
        row["filename"]: ("figure", row) for row in config["primary_figures"]
    }
    if set(rendered) != set(metadata):
        raise GravityClusterManuscriptRendererError("rendered artifact inventory changed")
    inventory = []
    for filename in sorted(rendered):
        role, row = metadata[filename]
        value = rendered[filename]
        inventory.append(
            {
                "artifact_id": row["artifact_id"],
                "role": role,
                "filename": filename,
                "title": row["title"],
                "media_type": "text/csv" if role == "table" else "image/svg+xml",
                "bytes": len(value),
                "file_sha256": _bytes_sha(value),
            }
        )
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "renderer_id": config["renderer_id"],
        "decision": "PRIMARY_DEVELOPMENT_TABLES_AND_FIGURES_REPRODUCIBLE_NOT_PAPER_READY",
        "config_binding": {"path": CONFIG_PATH.as_posix(), "content_sha256": _sha(config)},
        "source_bindings": config["source_bindings"],
        "completed_goal_evidence": {
            "CP12.1": "one_command_recreates_all_7_frozen_primary_tables_and_6_frozen_primary_figures"
        },
        "supersedes_snapshot_blocker": {
            "source_receipt": "manuscript_evidence_package",
            "goal_task_id": "CP12.1",
            "reason": "The upstream package recorded CP12.1 before this downstream renderer existed.",
        },
        "artifact_inventory": inventory,
        "counts": {
            "primary_tables": len(config["primary_tables"]),
            "primary_figures": len(config["primary_figures"]),
            "artifacts": len(inventory),
            "source_candidate_rows": evidence["counts"]["per_row_candidate_predictions"],
            "independent_target_rows_opened": evidence["access_ledger"][
                "independent_target_rows_opened"
            ],
        },
        "claims": {
            "development_artifacts_reproducible": True,
            "every_frozen_primary_table_and_figure_rendered": True,
            "external_reproduction": False,
            "independent_replication": False,
            "bounded_paper_ready": False,
            "physical_mechanism_ready": False,
            "universal_theory_ready": False,
        },
        "reproduction": {
            "command": "python -m sigma_theory_compiler.gravity_cluster_manuscript_renderer render",
            "check_command": "python -m sigma_theory_compiler.gravity_cluster_manuscript_renderer check",
            "output_directory": config["output_directory"],
            "external_dependencies": [],
        },
        "limitations": [
            "The artifacts summarize development and same-release confirmation evidence, not independent replication.",
            "SVG and CSV recreation is complete, but separately maintained external replay remains open.",
            "The covariance figure uses frozen stress models rather than released instrument/source covariance.",
        ],
        "next_action": "Complete source covariance, frozen independent-source selection, and external analyst replay before paper readiness.",
    }
    return {**body, "content_sha256": _sha(body)}


def write(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    artifacts = build_artifacts(root)
    output_directory = _under(root, str(config["output_directory"]), "artifact directory")
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, value in artifacts.items():
        (output_directory / filename).write_bytes(value)
    receipt = build_receipt(root, artifacts)
    receipt_path = _under(root, str(config["receipt_path"]), "artifact receipt")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(_canonical_bytes(receipt))
    return receipt_path


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterManuscriptRendererError("manuscript artifact receipt changed")
    config = load_config(root)
    output_directory = _under(root.resolve(), str(config["output_directory"]), "artifact directory")
    expected_files = {row["filename"] for row in receipt["artifact_inventory"]}
    actual_files = {path.name for path in output_directory.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise GravityClusterManuscriptRendererError("stored artifact inventory changed")
    for row in receipt["artifact_inventory"]:
        path = output_directory / str(row["filename"])
        if path.stat().st_size != row["bytes"] or _file_sha(path) != row["file_sha256"]:
            raise GravityClusterManuscriptRendererError(
                f"stored artifact changed: {row['filename']}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = load_config(root)
    receipt_path = _under(root, str(config["receipt_path"]), "artifact receipt")
    if args.command == "render":
        output: Any = str(write(root))
    else:
        receipt = _read_json(receipt_path)
        validate_receipt(receipt, root)
        if args.command == "check":
            output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
        else:
            output = {
                "decision": receipt["decision"],
                "counts": receipt["counts"],
                "claims": receipt["claims"],
                "next_action": receipt["next_action"],
            }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
