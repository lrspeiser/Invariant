"""Build target-blind structural plans for recombining the five gravity leads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_lead_recombination_v1.json")
IMPLEMENTATION_PATH = Path("src/sigma_theory_compiler/gravity_lead_recombination.py")
OUTPUT_PATH = Path("runs/gravity/lead-programs/gravity-lead-recombination-v1.json")
CONFIG_SCHEMA = "invariant-gravity-lead-recombination-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-lead-recombination-preflight-1.0"
PARENT_RECEIPT_SCHEMA = "invariant-gravity-lead-parent-registry-receipt-1.0"
PROGRAM_ID = "gravity-lead-recombination-v1"
CONFIG_CONTRACT_SHA256 = "5d205fc472f789edf5c83dd06d347cec3079a95ed9dd6debae392b72381d5c62"

LEAD_IDS = (
    "nonlocal_boundary_response",
    "baryonic_transition_variable",
    "dynamical_age_spectral_clock",
    "massive_field_orbital_resonance",
    "emergent_gravity_transition",
)
ROLE_BY_LEAD = {
    "nonlocal_boundary_response": "spatial_response_operator",
    "baryonic_transition_variable": "state_or_gate",
    "dynamical_age_spectral_clock": "nuisance_or_assembly_control",
    "massive_field_orbital_resonance": "temporal_or_phase_operator",
    "emergent_gravity_transition": "base_local_law",
}
SYMBOL_BY_LEAD = {
    "nonlocal_boundary_response": "N",
    "baryonic_transition_variable": "B",
    "dynamical_age_spectral_clock": "A",
    "massive_field_orbital_resonance": "M",
    "emergent_gravity_transition": "E",
}
INTERFACE_IDS = (
    "normalized_acceleration",
    "relativistic_compactness",
    "boundary_curvature",
    "thermal_state",
    "geometry_shape",
    "occupancy",
    "nonlocal_response",
    "orbital_phase",
    "physical_age_ratio",
    "environment_gate",
    "lensing_response",
)
FORBIDDEN_RULE_IDS = (
    "forbid_raw_spectral_clock_force_source",
    "forbid_arbitrary_all_channel_product",
    "forbid_object_survey_or_class_switch",
    "forbid_target_derived_gate",
    "forbid_independent_lensing_fit",
    "forbid_low_acceleration_double_counting",
    "forbid_static_resonance_as_dynamic_proof",
    "forbid_dimensionally_untyped_operation",
)
CONTROL_IDS = (
    "known_base_plus_null_channel",
    "flexible_ordinary_predictor_only",
    "seeded_random_dimensionless_expression",
    "seeded_random_gate_same_basis",
    "constant_gate_ablation",
)
PUBLICATION_TRACK_IDS = (
    "association_astronomy",
    "bounded_phenomenology",
    "failure_space_methods",
    "gravity_theory",
)
NOVELTY_LABELS = (
    "known_rewrite",
    "known_combination",
    "potentially_new_synthesis",
    "unclassified",
)
ABLATION_MODES = (
    "base_only",
    "base_plus_each_single_channel",
    "full_additive_model",
    "leave_one_channel_out",
    "gate_replaced_by_constant",
    "nuisance_on_off",
)

PAIR_DISPOSITIONS = {
    frozenset(("nonlocal_boundary_response", "baryonic_transition_variable")):
        "high_priority_mechanism_candidate",
    frozenset(("nonlocal_boundary_response", "dynamical_age_spectral_clock")):
        "control_only",
    frozenset(("nonlocal_boundary_response", "massive_field_orbital_resonance")):
        "theory_first_deferred",
    frozenset(("nonlocal_boundary_response", "emergent_gravity_transition")):
        "high_priority_mechanism_candidate",
    frozenset(("baryonic_transition_variable", "dynamical_age_spectral_clock")):
        "control_only",
    frozenset(("baryonic_transition_variable", "massive_field_orbital_resonance")):
        "conditional_mechanism_deferred",
    frozenset(("baryonic_transition_variable", "emergent_gravity_transition")):
        "medium_high_mechanism_candidate",
    frozenset(("dynamical_age_spectral_clock", "massive_field_orbital_resonance")):
        "control_and_falsification_only",
    frozenset(("dynamical_age_spectral_clock", "emergent_gravity_transition")):
        "control_only",
    frozenset(("massive_field_orbital_resonance", "emergent_gravity_transition")):
        "theory_first_deferred",
}
TRIPLE_DISPOSITIONS = {
    frozenset(
        (
            "nonlocal_boundary_response",
            "baryonic_transition_variable",
            "dynamical_age_spectral_clock",
        )
    ): "valid_mechanism_with_nuisance",
    frozenset(
        (
            "nonlocal_boundary_response",
            "baryonic_transition_variable",
            "massive_field_orbital_resonance",
        )
    ): "conditional_theory_first",
    frozenset(
        (
            "nonlocal_boundary_response",
            "baryonic_transition_variable",
            "emergent_gravity_transition",
        )
    ): "top_priority_structural_architecture",
    frozenset(
        (
            "nonlocal_boundary_response",
            "dynamical_age_spectral_clock",
            "massive_field_orbital_resonance",
        )
    ): "defer_theory_with_nuisance",
    frozenset(
        (
            "nonlocal_boundary_response",
            "dynamical_age_spectral_clock",
            "emergent_gravity_transition",
        )
    ): "valid_mechanism_with_nuisance",
    frozenset(
        (
            "nonlocal_boundary_response",
            "massive_field_orbital_resonance",
            "emergent_gravity_transition",
        )
    ): "reject_initial_empirical_grammar",
    frozenset(
        (
            "baryonic_transition_variable",
            "dynamical_age_spectral_clock",
            "massive_field_orbital_resonance",
        )
    ): "deferred_mechanism_with_nuisance",
    frozenset(
        (
            "baryonic_transition_variable",
            "dynamical_age_spectral_clock",
            "emergent_gravity_transition",
        )
    ): "valid_mechanism_with_nuisance",
    frozenset(
        (
            "baryonic_transition_variable",
            "massive_field_orbital_resonance",
            "emergent_gravity_transition",
        )
    ): "conditional_theory_first",
    frozenset(
        (
            "dynamical_age_spectral_clock",
            "massive_field_orbital_resonance",
            "emergent_gravity_transition",
        )
    ): "defer_low_priority_with_nuisance",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class GravityLeadRecombinationError(RuntimeError):
    """Raised when the target-blind recombination contract changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _content_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = _canonical_sha256(result)
    return result


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GravityLeadRecombinationError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise GravityLeadRecombinationError(f"{label} is not a JSON object")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise GravityLeadRecombinationError(f"{label} keys changed")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GravityLeadRecombinationError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _under(root: Path, relative: str, label: str) -> Path:
    if not relative or "\\" in relative:
        raise GravityLeadRecombinationError(f"{label} must use a relative POSIX path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise GravityLeadRecombinationError(f"{label} escaped the repository") from error
    if not path.is_file():
        raise GravityLeadRecombinationError(f"{label} is missing")
    return path


def _verify_parent_registry(
    root: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    _strict(
        binding,
        {
            "registry_id",
            "config_path",
            "config_sha256",
            "implementation_sha256",
            "receipt_path",
            "receipt_file_sha256",
            "receipt_content_sha256",
            "receipt_schema_version",
        },
        "parent registry binding",
    )
    if (
        binding["registry_id"] != "gravity-lead-parent-registry-v1"
        or binding["receipt_schema_version"] != PARENT_RECEIPT_SCHEMA
    ):
        raise GravityLeadRecombinationError("parent registry identity changed")
    config_path = _under(root, str(binding["config_path"]), "parent registry config")
    receipt_path = _under(root, str(binding["receipt_path"]), "parent registry receipt")
    expected_config = _digest(binding["config_sha256"], "parent config hash")
    expected_file = _digest(binding["receipt_file_sha256"], "parent receipt file hash")
    expected_content = _digest(
        binding["receipt_content_sha256"], "parent receipt content hash"
    )
    expected_implementation = _digest(
        binding["implementation_sha256"], "parent implementation hash"
    )
    if _file_sha256(config_path) != expected_config:
        raise GravityLeadRecombinationError("parent registry config binding changed")
    if _file_sha256(receipt_path) != expected_file:
        raise GravityLeadRecombinationError("parent registry receipt file binding changed")

    parent_config = _read_json(config_path, "parent registry config")
    parent = _read_json(receipt_path, "parent registry receipt")
    body = {key: item for key, item in parent.items() if key != "content_sha256"}
    if (
        parent.get("schema_version") != PARENT_RECEIPT_SCHEMA
        or parent.get("registry_id") != binding["registry_id"]
        or parent.get("content_sha256") != expected_content
        or parent.get("content_sha256") != _canonical_sha256(body)
        or parent.get("registry_config_sha256") != expected_config
        or parent.get("registry_implementation_sha256") != expected_implementation
        or parent.get("decision")
        != "PASS_ALL_FIVE_PARENTS_REGISTERED_EVIDENCE_INTACT"
        or parent.get("lead_count") != 5
    ):
        raise GravityLeadRecombinationError("parent registry receipt content changed")
    if parent.get("safety") != {
        "metadata_only": True,
        "raw_payloads_opened": 0,
        "sealed_target_rows_opened": 0,
        "network_calls": 0,
        "gpu_production_runs": 0,
        "paid_model_calls": 0,
    }:
        raise GravityLeadRecombinationError("parent registry safety boundary changed")
    if parent.get("claim_boundary") != {
        "registry_pass_establishes_empirical_replication": False,
        "registry_pass_establishes_physical_mechanism": False,
        "registry_pass_establishes_alternative_to_gr": False,
        "registry_pass_establishes_historical_novelty": False,
        "registry_pass_only_establishes_metadata_integrity": True,
    }:
        raise GravityLeadRecombinationError("parent registry claim boundary changed")
    if tuple(row.get("lead_id") for row in parent.get("lead_programs", [])) != LEAD_IDS:
        raise GravityLeadRecombinationError("parent registry lead inventory changed")
    if any(
        row.get("registry_status") != "REGISTERED_EVIDENCE_INTACT"
        for row in parent["lead_programs"]
    ):
        raise GravityLeadRecombinationError("a parent lead is not evidence-intact")
    if (
        parent_config.get("registry_id") != binding["registry_id"]
        or tuple(
            row.get("lead_id") for row in parent_config.get("lead_programs", [])
        )
        != LEAD_IDS
    ):
        raise GravityLeadRecombinationError("parent registry config inventory changed")
    return parent


def _validate_matrix(
    rows: Any,
    arity: int,
    expected_dispositions: Mapping[frozenset[str], str],
) -> None:
    label = "pairwise" if arity == 2 else "triple"
    expected_members = list(combinations(LEAD_IDS, arity))
    if not isinstance(rows, list) or len(rows) != len(expected_members):
        raise GravityLeadRecombinationError(f"{label} matrix coverage changed")
    if [tuple(row.get("members", [])) for row in rows] != expected_members:
        raise GravityLeadRecombinationError(f"{label} matrix ordering or coverage changed")
    prefix = "pair" if arity == 2 else "triple"
    for row in rows:
        _strict(
            row,
            {
                "recombination_id",
                "members",
                "disposition",
                "mechanism_members",
                "control_members",
                "deferred_members",
                "composition",
                "reason",
            },
            f"{label} recombination",
        )
        members = tuple(row["members"])
        expected_id = f"{prefix}." + "__".join(members)
        if (
            row["recombination_id"] != expected_id
            or row["disposition"] != expected_dispositions[frozenset(members)]
        ):
            raise GravityLeadRecombinationError(f"{label} disposition changed")
        mechanism = tuple(row["mechanism_members"])
        controls = tuple(row["control_members"])
        deferred = tuple(row["deferred_members"])
        classified = (*mechanism, *controls, *deferred)
        if (
            any(member not in members for member in classified)
            or len(set(classified)) != len(classified)
            or not isinstance(row["composition"], str)
            or not row["composition"].strip()
            or not isinstance(row["reason"], str)
            or not row["reason"].strip()
        ):
            raise GravityLeadRecombinationError(f"{label} role classification changed")
        if "dynamical_age_spectral_clock" in members and (
            "dynamical_age_spectral_clock" not in controls
        ):
            raise GravityLeadRecombinationError("spectral clock escaped nuisance-only role")
        if "massive_field_orbital_resonance" in members and (
            "massive_field_orbital_resonance" not in deferred
        ):
            raise GravityLeadRecombinationError("massive resonance escaped deferred role")


def validate_config(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    root = root.resolve()
    _strict(
        config,
        {
            "schema_version",
            "status",
            "program_id",
            "purpose",
            "parent_registry_binding",
            "safety_contract",
            "lead_roles",
            "dimensionless_interfaces",
            "pairwise_recombinations",
            "triple_recombinations",
            "forbidden_combinations",
            "target_blind_generation",
            "ablation_contract",
            "control_contract",
            "novelty_policy",
            "publication_interest_gates",
            "top_architecture",
            "output_path",
            "contract_sha256",
        },
        "recombination config",
    )
    body = {key: item for key, item in config.items() if key != "contract_sha256"}
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_metadata_only_recombination_preflight"
        or config["program_id"] != PROGRAM_ID
        or config["output_path"] != OUTPUT_PATH.as_posix()
        or config["contract_sha256"] != CONFIG_CONTRACT_SHA256
        or config["contract_sha256"] != _canonical_sha256(body)
    ):
        raise GravityLeadRecombinationError("recombination config identity or seal changed")
    parent = _verify_parent_registry(root, config["parent_registry_binding"])

    if config["safety_contract"] != {
        "metadata_only": True,
        "scientific_payload_rows_allowed": 0,
        "sealed_target_rows_allowed": 0,
        "outcome_scores_allowed": 0,
        "network_calls_allowed": False,
        "model_calls_allowed": False,
        "paid_calls_allowed": False,
        "gpu_production_allowed": False,
        "children_executed_by_preflight": False,
    }:
        raise GravityLeadRecombinationError("recombination safety boundary changed")

    roles = config["lead_roles"]
    if not isinstance(roles, Mapping) or tuple(roles) != LEAD_IDS:
        raise GravityLeadRecombinationError("lead role inventory changed")
    for lead_id, role in roles.items():
        _strict(
            role,
            {"symbol", "typed_role", "mechanism_status", "interface_ids"},
            f"lead role {lead_id}",
        )
        if (
            role["symbol"] != SYMBOL_BY_LEAD[lead_id]
            or role["typed_role"] != ROLE_BY_LEAD[lead_id]
            or not isinstance(role["mechanism_status"], str)
            or not role["mechanism_status"].strip()
            or not isinstance(role["interface_ids"], list)
            or not role["interface_ids"]
            or any(item not in INTERFACE_IDS for item in role["interface_ids"])
        ):
            raise GravityLeadRecombinationError(f"lead role changed: {lead_id}")

    interfaces = config["dimensionless_interfaces"]
    if (
        not isinstance(interfaces, list)
        or tuple(item.get("interface_id") for item in interfaces) != INTERFACE_IDS
    ):
        raise GravityLeadRecombinationError("dimensionless interface inventory changed")
    allowed_roles = set(ROLE_BY_LEAD.values())
    for interface in interfaces:
        _strict(
            interface,
            {
                "interface_id",
                "symbol",
                "definition",
                "dimensionless",
                "allowed_consumers",
                "status",
            },
            "dimensionless interface",
        )
        if (
            interface["dimensionless"] is not True
            or not set(interface["allowed_consumers"]) <= allowed_roles
            or not interface["allowed_consumers"]
            or not isinstance(interface["definition"], str)
            or not interface["definition"].strip()
        ):
            raise GravityLeadRecombinationError("dimensionless interface typing changed")
    if interfaces[-1]["status"] != (
        "derived_only_from_shared_action_or_field_equations_never_independently_fit"
    ):
        raise GravityLeadRecombinationError("lensing derivation boundary changed")

    _validate_matrix(config["pairwise_recombinations"], 2, PAIR_DISPOSITIONS)
    _validate_matrix(config["triple_recombinations"], 3, TRIPLE_DISPOSITIONS)

    forbidden = config["forbidden_combinations"]
    if (
        not isinstance(forbidden, list)
        or tuple(row.get("rule_id") for row in forbidden) != FORBIDDEN_RULE_IDS
    ):
        raise GravityLeadRecombinationError("forbidden-combination registry changed")
    for rule in forbidden:
        _strict(rule, {"rule_id", "pattern", "reason"}, "forbidden-combination rule")
        if not rule["pattern"] or not rule["reason"]:
            raise GravityLeadRecombinationError("forbidden-combination rule is empty")

    target_blind = config["target_blind_generation"]
    _strict(
        target_blind,
        {
            "allowed_inputs",
            "forbidden_inputs",
            "formula_generation_before_response_required",
            "post_response_formula_cells_allowed",
            "object_label_switches_allowed",
            "survey_label_switches_allowed",
            "class_label_switches_allowed",
            "deterministic_enumeration_required",
            "random_seed",
            "llm_proposals_in_this_preflight",
            "network_queries_in_this_preflight",
        },
        "target-blind generation contract",
    )
    required_forbidden = {
        "scientific_payload_rows",
        "outcome_scores",
        "object_identifiers",
        "object_labels",
        "survey_labels",
        "class_labels",
        "inferred_total_mass",
        "target_coefficients",
        "sealed_targets",
    }
    if (
        not required_forbidden <= set(target_blind["forbidden_inputs"])
        or target_blind["formula_generation_before_response_required"] is not True
        or target_blind["post_response_formula_cells_allowed"] != 0
        or target_blind["object_label_switches_allowed"] is not False
        or target_blind["survey_label_switches_allowed"] is not False
        or target_blind["class_label_switches_allowed"] is not False
        or target_blind["deterministic_enumeration_required"] is not True
        or target_blind["llm_proposals_in_this_preflight"] != 0
        or target_blind["network_queries_in_this_preflight"] != 0
        or not isinstance(target_blind["random_seed"], str)
        or not target_blind["random_seed"].startswith("sha256:")
        or _SHA256.fullmatch(target_blind["random_seed"].removeprefix("sha256:")) is None
    ):
        raise GravityLeadRecombinationError("target-blind generation boundary changed")

    ablations = config["ablation_contract"]
    if ablations != {
        "composition_mode": "additive_orthogonal_channels",
        "products_of_channels_allowed": False,
        "each_added_channel_must_be_ablatable": True,
        "required_modes": list(ABLATION_MODES),
        "added_channel_must_target_nonredundant_observable_or_scale": True,
    }:
        raise GravityLeadRecombinationError("additive ablation contract changed")

    controls = config["control_contract"]
    _strict(
        controls,
        {
            "matched_complexity_required",
            "whole_object_nested_selection_required",
            "single_counterexample_is_universal_veto",
            "controls",
        },
        "control contract",
    )
    if (
        controls["matched_complexity_required"] is not True
        or controls["whole_object_nested_selection_required"] is not True
        or controls["single_counterexample_is_universal_veto"] is not False
        or tuple(row.get("control_id") for row in controls["controls"]) != CONTROL_IDS
    ):
        raise GravityLeadRecombinationError("matched-control contract changed")
    for control in controls["controls"]:
        _strict(control, {"control_id", "kind", "target_blind"}, "matched control")
        if control["target_blind"] is not True or not control["kind"]:
            raise GravityLeadRecombinationError("matched control lost target blindness")

    novelty = config["novelty_policy"]
    if novelty != {
        "allowed_labels": list(NOVELTY_LABELS),
        "labels_are_historical_novelty_findings": False,
        "labels_are_authoritative": False,
        "corpus_absence_establishes_novelty": False,
        "specialist_prior_art_review_required": True,
        "behavioral_uniqueness_establishes_historical_novelty": False,
    }:
        raise GravityLeadRecombinationError("novelty claim boundary changed")

    publication = config["publication_interest_gates"]
    if (
        not isinstance(publication, list)
        or tuple(row.get("track_id") for row in publication) != PUBLICATION_TRACK_IDS
    ):
        raise GravityLeadRecombinationError("publication-interest gate inventory changed")
    for gate in publication:
        _strict(
            gate,
            {"track_id", "required_evidence", "preflight_satisfies_gate", "claim_ceiling"},
            "publication-interest gate",
        )
        if (
            gate["preflight_satisfies_gate"] is not False
            or not gate["required_evidence"]
            or not gate["claim_ceiling"]
        ):
            raise GravityLeadRecombinationError("publication-interest gate was weakened")

    top = config["top_architecture"]
    if top != {
        "architecture_id": "BEN-additive-cross-scale-v1",
        "members": [
            "nonlocal_boundary_response",
            "baryonic_transition_variable",
            "emergent_gravity_transition",
        ],
        "base_lead": "emergent_gravity_transition",
        "gate_lead": "baryonic_transition_variable",
        "additive_channel_lead": "nonlocal_boundary_response",
        "nuisance_lead": "dynamical_age_spectral_clock",
        "deferred_lead": "massive_field_orbital_resonance",
        "formula_template": (
            "g_dyn = g_base_E(u) + T_B(C,b,Theta,Ghat) Delta_g_N[rho_b]"
        ),
        "lensing_rule": (
            "derive Phi and Psi from the same frozen action before any direct-lensing "
            "target access"
        ),
        "first_fresh_data_gate": "galaxy_group_bridge_before_direct_lensing",
        "structural_descendants_only": True,
        "children_empirically_work": False,
    }:
        raise GravityLeadRecombinationError("top B+E+N architecture changed")
    return parent


def load_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    config = _read_json(root / CONFIG_PATH, "recombination config")
    parent = validate_config(config, root)
    return config, parent


def _interfaces_for_members(
    config: Mapping[str, Any], members: Sequence[str]
) -> list[str]:
    selected = {
        interface_id
        for member in members
        for interface_id in config["lead_roles"][member]["interface_ids"]
    }
    return [interface_id for interface_id in INTERFACE_IDS if interface_id in selected]


def _build_descendant(
    config: Mapping[str, Any], row: Mapping[str, Any], arity: int
) -> dict[str, Any]:
    members = list(row["members"])
    body: dict[str, Any] = {
        "recombination_id": row["recombination_id"],
        "arity": arity,
        "members": members,
        "typed_roles": {
            member: config["lead_roles"][member]["typed_role"] for member in members
        },
        "interface_ids": _interfaces_for_members(config, members),
        "disposition": row["disposition"],
        "mechanism_members": row["mechanism_members"],
        "control_members": row["control_members"],
        "deferred_members": row["deferred_members"],
        "composition": row["composition"],
        "scientific_rationale": row["reason"],
        "ablation_modes": list(ABLATION_MODES),
        "matched_control_ids": list(CONTROL_IDS),
        "novelty_labels_allowed_non_authoritatively": list(NOVELTY_LABELS),
        "target_data_bindings": [],
        "scientific_payload_rows_read": 0,
        "outcome_scores_computed": 0,
        "execution_authorized": False,
        "claim_status": "PREFLIGHT_ONLY_CHILD_NOT_EXECUTED",
    }
    if row["disposition"] == "top_priority_structural_architecture":
        body["top_architecture"] = config["top_architecture"]
    body["descendant_id"] = "descendant." + _canonical_sha256(body)[:24]
    body["plan_sha256"] = _canonical_sha256(body)
    return body


def build_descendant_plans(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    plans = [
        _build_descendant(config, row, 2) for row in config["pairwise_recombinations"]
    ]
    plans.extend(
        _build_descendant(config, row, 3) for row in config["triple_recombinations"]
    )
    if (
        len(plans) != 20
        or len({row["descendant_id"] for row in plans}) != 20
        or len({row["plan_sha256"] for row in plans}) != 20
        or sum("top_architecture" in row for row in plans) != 1
    ):
        raise GravityLeadRecombinationError("descendant structural coverage changed")
    return plans


def _source_bindings(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    parent = config["parent_registry_binding"]
    paths = {
        "recombination_config": CONFIG_PATH.as_posix(),
        "recombination_implementation": IMPLEMENTATION_PATH.as_posix(),
        "parent_registry_config": parent["config_path"],
        "parent_registry_receipt": parent["receipt_path"],
    }
    return {
        name: {"path": path, "sha256": _file_sha256(_under(root, path, name))}
        for name, path in sorted(paths.items())
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config, parent = load_config(root)
    descendants = build_descendant_plans(config)
    top = next(row for row in descendants if "top_architecture" in row)
    receipt = _content_hashed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "program_id": PROGRAM_ID,
            "decision": "PASS_TARGET_BLIND_STRUCTURAL_PREFLIGHT_CHILDREN_NOT_EXECUTED",
            "source_bindings": _source_bindings(root, config),
            "parent_registry": {
                "registry_id": parent["registry_id"],
                "receipt_content_sha256": parent["content_sha256"],
                "lead_count": parent["lead_count"],
                "registered_evidence_files": parent["registered_evidence_files"],
                "decision": parent["decision"],
            },
            "coverage": {
                "lead_roles": len(config["lead_roles"]),
                "dimensionless_interfaces": len(config["dimensionless_interfaces"]),
                "pairwise_recombinations": len(config["pairwise_recombinations"]),
                "triple_recombinations": len(config["triple_recombinations"]),
                "structural_descendants": len(descendants),
                "forbidden_combination_rules": len(config["forbidden_combinations"]),
                "matched_controls": len(config["control_contract"]["controls"]),
                "publication_interest_tracks": len(config["publication_interest_gates"]),
            },
            "role_registry": config["lead_roles"],
            "dimensionless_interfaces": config["dimensionless_interfaces"],
            "forbidden_combinations": config["forbidden_combinations"],
            "target_blind_generation": config["target_blind_generation"],
            "ablation_contract": config["ablation_contract"],
            "control_contract": config["control_contract"],
            "novelty_policy": config["novelty_policy"],
            "publication_interest_gates": config["publication_interest_gates"],
            "descendant_plans": descendants,
            "top_architecture": {
                "descendant_id": top["descendant_id"],
                **config["top_architecture"],
            },
            "safety": {
                "metadata_only": True,
                "scientific_payload_rows_read": 0,
                "sealed_target_rows_opened": 0,
                "outcome_scores_computed": 0,
                "network_calls": 0,
                "model_calls": 0,
                "paid_calls": 0,
                "gpu_production_runs": 0,
                "children_executed": 0,
            },
            "release_gate": {
                "status": "PASS_PREFLIGHT_ONLY_EXECUTION_BLOCKED",
                "fresh_data_contract_bound": False,
                "common_evaluator_bound": False,
                "group_bridge_bound": False,
                "direct_lensing_opening_authorized": False,
                "child_execution_authorized": False,
            },
            "claim_boundary": {
                "children_empirically_work": False,
                "physical_mechanism_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
                "publication_gate_passed": False,
                "structural_preflight_only": True,
            },
            "known_blockers": [
                "No shared object-level data schema spans all five parent leads.",
                "No common baryonic source and geometry mapper with full covariance is bound.",
                "No fresh galaxy-group bridge or common direct-lensing evaluator is bound.",
                "No covariant action produces dynamics and lensing for the structural children.",
                "The age lead remains a nuisance proxy and the resonance lead remains deferred.",
            ],
        }
    )
    validate_receipt(receipt, root, rebuild=False)
    return receipt


def validate_receipt(
    receipt: Mapping[str, Any], root: Path, *, rebuild: bool = True
) -> None:
    root = root.resolve()
    _strict(
        receipt,
        {
            "schema_version",
            "program_id",
            "decision",
            "source_bindings",
            "parent_registry",
            "coverage",
            "role_registry",
            "dimensionless_interfaces",
            "forbidden_combinations",
            "target_blind_generation",
            "ablation_contract",
            "control_contract",
            "novelty_policy",
            "publication_interest_gates",
            "descendant_plans",
            "top_architecture",
            "safety",
            "release_gate",
            "claim_boundary",
            "known_blockers",
            "content_sha256",
        },
        "recombination receipt",
    )
    body = {key: item for key, item in receipt.items() if key != "content_sha256"}
    if (
        receipt["schema_version"] != RECEIPT_SCHEMA
        or receipt["program_id"] != PROGRAM_ID
        or receipt["decision"]
        != "PASS_TARGET_BLIND_STRUCTURAL_PREFLIGHT_CHILDREN_NOT_EXECUTED"
        or receipt["content_sha256"] != _canonical_sha256(body)
    ):
        raise GravityLeadRecombinationError("recombination receipt seal changed")
    config, _ = load_config(root)
    if receipt["source_bindings"] != _source_bindings(root, config):
        raise GravityLeadRecombinationError("recombination source binding changed")
    if receipt["descendant_plans"] != build_descendant_plans(config):
        raise GravityLeadRecombinationError("recombination descendant plans changed")
    if receipt["safety"] != {
        "metadata_only": True,
        "scientific_payload_rows_read": 0,
        "sealed_target_rows_opened": 0,
        "outcome_scores_computed": 0,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "gpu_production_runs": 0,
        "children_executed": 0,
    }:
        raise GravityLeadRecombinationError("recombination receipt safety changed")
    if receipt["release_gate"] != {
        "status": "PASS_PREFLIGHT_ONLY_EXECUTION_BLOCKED",
        "fresh_data_contract_bound": False,
        "common_evaluator_bound": False,
        "group_bridge_bound": False,
        "direct_lensing_opening_authorized": False,
        "child_execution_authorized": False,
    } or receipt["claim_boundary"] != {
        "children_empirically_work": False,
        "physical_mechanism_established": False,
        "alternative_to_gr_established": False,
        "dark_matter_eliminated": False,
        "historical_novelty_established": False,
        "publication_gate_passed": False,
        "structural_preflight_only": True,
    }:
        raise GravityLeadRecombinationError("recombination receipt claim boundary changed")
    if rebuild and receipt != build_receipt(root):
        raise GravityLeadRecombinationError(
            "stored recombination receipt does not rebuild from current sources"
        )


def write_receipt(root: Path, output_path: Path = OUTPUT_PATH) -> Path:
    root = root.resolve()
    output = output_path if output_path.is_absolute() else (root / output_path).resolve()
    try:
        output.relative_to(root / "runs" / "gravity" / "lead-programs")
    except ValueError as error:
        raise GravityLeadRecombinationError(
            "recombination output escaped lead-programs directory"
        ) from error
    receipt = build_receipt(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(receipt))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-parent", "build", "validate"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--receipt", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "check-parent":
            config, parent = load_config(root)
            result: dict[str, Any] = {
                "ok": True,
                "registry_id": parent["registry_id"],
                "parent_content_sha256": parent["content_sha256"],
                "config_contract_sha256": config["contract_sha256"],
            }
        elif args.command == "build":
            output = write_receipt(root, args.output)
            receipt = _read_json(output, "recombination receipt")
            result = {
                "ok": True,
                "output": output.relative_to(root).as_posix(),
                "content_sha256": receipt["content_sha256"],
                "descendants": receipt["coverage"]["structural_descendants"],
                "status": receipt["release_gate"]["status"],
            }
        else:
            receipt_path = (
                args.receipt if args.receipt.is_absolute() else (root / args.receipt).resolve()
            )
            receipt = _read_json(receipt_path, "recombination receipt")
            validate_receipt(receipt, root)
            result = {
                "ok": True,
                "content_sha256": receipt["content_sha256"],
                "descendants": receipt["coverage"]["structural_descendants"],
                "status": receipt["release_gate"]["status"],
            }
    except GravityLeadRecombinationError as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
