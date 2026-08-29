"""Fail-closed no-data feasibility preflight for a covariant E+B*N template."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_matter_lensing_theory_preflight_v1.json")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-theory-preflight-v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_matter_lensing_theory_preflight.py")
TEST_PATH = Path("tests/test_gravity_matter_lensing_theory_preflight.py")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-theory-preflight-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-theory-preflight-receipt-1.0"
EXPECTED_CONFIG_FILE_SHA256 = "f6625b7d3a9d35b234d7be4eeeb29cac5b9f7ae7ebf1d3238b87d6c2739564bb"
EXPECTED_CONFIG_CONTENT_SHA256 = "f36413e0d27424e28254ff05a1f2272b9137823f53bab9a4c18e30ce0815bf06"
DECISION = (
    "BLOCKED_TWO_SCALAR_COVARIANT_TEMPLATE_DEFINED_"
    "HEALTHY_MATTER_LENSING_COMPLETION_NOT_ESTABLISHED"
)
SOURCE_IDS = (
    "gravity_lead_parent_registry",
    "gravity_lead_recombination",
    "shared_ben_synthetic_execution",
)
TERM_IDS = (
    "EH",
    "PHI_KESSENCE",
    "CHI_YUKAWA",
    "B_KINETIC_GATE",
    "UNIVERSAL_CONFORMAL",
    "UNIVERSAL_DISFORMAL",
    "ASSEMBLED_TWO_FIELD_TEMPLATE",
)
HEALTH_GATE_IDS = tuple(f"H{i}_" for i in range(1, 11))
ZERO_ACCESS_KEYS = (
    "observational_files_opened",
    "predictor_rows_opened",
    "response_rows_opened",
    "confirmation_rows_opened",
    "holdout_rows_opened",
    "independent_rows_opened",
    "lensing_rows_opened",
    "formula_scores_computed",
    "likelihood_calls",
    "network_calls",
    "LLM_calls",
    "paid_calls",
    "GPU_calls",
)


class GravityMatterLensingTheoryPreflightError(RuntimeError):
    """Raised when a frozen theory contract, predecessor, or receipt changes."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GravityMatterLensingTheoryPreflightError(
            f"cannot read JSON {path.as_posix()}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise GravityMatterLensingTheoryPreflightError(f"JSON object required: {path.as_posix()}")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise GravityMatterLensingTheoryPreflightError(
            f"{label} keys changed: expected {sorted(expected)}, got {sorted(value)}"
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GravityMatterLensingTheoryPreflightError(message)


def _validate_embedded_content_hash(value: Mapping[str, Any], expected: str, label: str) -> None:
    body = dict(value)
    observed = body.pop("content_sha256", None)
    _require(observed == expected, f"{label} declared content hash changed")
    compact = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    _require(expected in {_sha(body), compact}, f"{label} embedded content hash invalid")


def _load_source(binding: Mapping[str, Any], root: Path) -> dict[str, Any]:
    _require_exact_keys(
        binding,
        {
            "source_id",
            "path",
            "file_sha256",
            "content_sha256",
            "schema_version",
            "decision",
        },
        "source binding",
    )
    path = root / str(binding["path"])
    _require(path.is_file(), f"source is missing: {path.as_posix()}")
    _require(_file_sha(path) == binding["file_sha256"], f"source file hash changed: {path}")
    value = _read_json(path)
    _require(value.get("schema_version") == binding["schema_version"], "source schema changed")
    _require(value.get("decision") == binding["decision"], "source decision changed")
    _validate_embedded_content_hash(
        value, str(binding["content_sha256"]), str(binding["source_id"])
    )
    return value


def _validate_source_semantics(source_id: str, value: Mapping[str, Any]) -> None:
    if source_id == "gravity_lead_parent_registry":
        _require(value.get("lead_count") == 5, "parent registry must bind five leads")
        boundary = value.get("claim_boundary")
        _require(isinstance(boundary, dict), "parent registry claim boundary missing")
        _require(
            boundary.get("registry_pass_only_establishes_metadata_integrity") is True,
            "parent registry claim ceiling changed",
        )
        for key in (
            "registry_pass_establishes_alternative_to_gr",
            "registry_pass_establishes_empirical_replication",
            "registry_pass_establishes_historical_novelty",
            "registry_pass_establishes_physical_mechanism",
        ):
            _require(boundary.get(key) is False, f"parent registry overclaim changed: {key}")
    elif source_id == "gravity_lead_recombination":
        boundary = value.get("claim_boundary")
        _require(isinstance(boundary, dict), "recombination claim boundary missing")
        _require(boundary.get("structural_preflight_only") is True, "recombination scope changed")
        for key in (
            "alternative_to_gr_established",
            "children_empirically_work",
            "dark_matter_eliminated",
            "historical_novelty_established",
            "physical_mechanism_established",
            "publication_gate_passed",
        ):
            _require(boundary.get(key) is False, f"recombination overclaim changed: {key}")
        safety = value.get("safety")
        _require(isinstance(safety, dict), "recombination safety boundary missing")
        _require(safety.get("children_executed") == 0, "recombination child ran")
    elif source_id == "shared_ben_synthetic_execution":
        boundary = value.get("claim_boundary")
        data = value.get("data_boundary")
        _require(isinstance(boundary, dict), "synthetic claim boundary missing")
        _require(isinstance(data, dict), "synthetic data boundary missing")
        _require(
            boundary.get("synthetic_grammar_mechanics_validated") is True,
            "synthetic mechanics status changed",
        )
        for key in (
            "ben_child_empirically_works",
            "candidate_physics_supported",
            "gr_replaced",
            "historical_novelty_established",
            "publication_ready",
            "real_scientific_evaluation_unlocked",
            "same_action_derived",
            "synthetic_recovery_is_scientific_evidence",
        ):
            _require(boundary.get(key) is False, f"synthetic overclaim changed: {key}")
        for key, item in data.items():
            if key == "synthetic_rows_per_domain":
                continue
            if key == "real_target_fields_read":
                _require(item == [], "synthetic receipt accessed real target fields")
            elif isinstance(item, int):
                _require(item == 0, f"synthetic real-data/compute boundary changed: {key}")
    else:  # pragma: no cover - guarded by the exact source registry
        raise GravityMatterLensingTheoryPreflightError(f"unknown source id: {source_id}")


def validate_config_contract(config: Mapping[str, Any]) -> None:
    _require_exact_keys(
        config,
        {
            "schema_version",
            "preflight_id",
            "status",
            "purpose",
            "source_bindings",
            "role_mapping",
            "conventions_and_dimensions",
            "action_family",
            "term_provenance_ledger",
            "field_equations_and_symbolic_contract",
            "degrees_of_freedom",
            "conservation_identity",
            "weak_field_matter_and_lensing",
            "health_and_consistency_gates",
            "exact_parameter_and_limit_gates",
            "feasibility_adjudication",
            "claim_boundary",
            "zero_access_and_compute",
            "output_path",
        },
        "config",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content changed")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")

    sources = config["source_bindings"]
    _require(isinstance(sources, list), "source bindings must be a list")
    _require(tuple(item["source_id"] for item in sources) == SOURCE_IDS, "source order changed")

    roles = config["role_mapping"]
    _require(roles["A_nuisance"] == "ABSENT_FROM_THEORY_ACTION", "A must be absent")
    _require(roles["M_temporal_phase"] == "ABSENT_FROM_THEORY_ACTION", "M must be absent")
    _require(
        roles["labels_are_empirical_roles_not_fundamental_fields"] is True, "role scope changed"
    )

    terms = config["term_provenance_ledger"]
    _require(tuple(item["term_id"] for item in terms) == TERM_IDS, "term ledger changed")
    _require(all(item["historical_novelty_claim"] is False for item in terms), "novelty overclaim")
    _require(
        {item["provenance_label"] for item in terms}
        == {"known_rewrite", "known_combination", "potentially_new_synthesis"},
        "provenance vocabulary changed",
    )

    metric = config["action_family"]["universal_physical_metric"]
    _require(metric["matter_and_photons_use_same_metric"] is True, "universal coupling changed")
    _require(metric["separate_photon_coefficient_forbidden"] is True, "photon-only factor unlocked")
    action = config["action_family"]
    _require(
        "Z_chi(u)*(X_chi - m_chi^2*chi^2/2)" in action["action"],
        "chi kinetic and mass terms must share the gate normalization",
    )
    _require(action["gate_and_range"]["ell_chi"] == "1/m_chi", "finite range changed")
    _require(
        action["gate_and_range"]["nonconstant_background_status"].startswith("BLOCKED_"),
        "gradient-of-Z background obligation was unlocked",
    )

    equations = config["field_equations_and_symbolic_contract"]
    _require(equations["machine_verified_symbolic_derivation"] is False, "derivation overclaim")
    _require(len(equations["symbolic_derivation_requirements"]) == 7, "derivation contract changed")
    _require(
        "(X_chi-m_chi^2*chi^2/2)*Z_chi,u" in equations["phi_equation"],
        "phi cross term changed",
    )
    _require(
        "-Z_chi(u)*m_chi^2*chi" in equations["chi_equation"],
        "chi mass normalization changed",
    )
    _require(
        equations["gradient_and_background_mixing_status"].startswith("BLOCKED_"),
        "background mixing was falsely resolved",
    )

    dof = config["degrees_of_freedom"]
    _require(
        (dof["tensor_polarizations"], dof["scalar_fields"], dof["vector_gravitational_modes"])
        == (2, 2, 0),
        "conditional degree-of-freedom count changed",
    )
    _require(dof["status"] == "CONDITIONAL_NOT_HAMILTONIAN_VERIFIED", "DOF claim changed")

    weak = config["weak_field_matter_and_lensing"]
    _require(
        "cancels" in weak["static_conformal_limit"]["lensing_sum"], "lensing cancellation lost"
    )
    _require(weak["status"].startswith("BLOCKED_"), "weak-field lensing gate must remain blocked")

    gates = config["health_and_consistency_gates"]
    _require(len(gates) == 10, "health gate count changed")
    _require(
        all(item["gate_id"].startswith(HEALTH_GATE_IDS[index]) for index, item in enumerate(gates)),
        "health gate order changed",
    )
    _require(gates[0]["status"] == "PASS_TEMPLATE_LEVEL", "template gate changed")
    _require(
        all(item["status"].startswith("BLOCKED_") for item in gates[1:]), "health gate unlocked"
    )

    parameters = config["exact_parameter_and_limit_gates"]
    _require(parameters["parameter_values_frozen"] is False, "parameters falsely frozen")
    _require(parameters["ell_chi_mpc_frozen"] is None, "range falsely frozen")
    _require(parameters["eft_cutoff_frozen"] is None, "cutoff falsely frozen")

    feasibility = config["feasibility_adjudication"]
    _require(feasibility["decision"] == DECISION, "feasibility decision changed")
    _require(feasibility["theory_feasible_for_observational_scoring"] is False, "scoring unlocked")

    claims = config["claim_boundary"]
    _require(all(value is False for value in claims.values()), "claim boundary overstates result")
    zero = config["zero_access_and_compute"]
    _require_exact_keys(zero, set(ZERO_ACCESS_KEYS), "zero-access accounting")
    _require(all(zero[key] == 0 for key in ZERO_ACCESS_KEYS), "nonzero access/compute declared")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    _require(path.is_file(), f"config is missing: {path.as_posix()}")
    _require(_file_sha(path) == EXPECTED_CONFIG_FILE_SHA256, "config file hash changed")
    config = _read_json(path)
    validate_config_contract(config)
    return config


def build_receipt(root: Path = Path("."), config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_config(root / config_path)
    loaded_sources: dict[str, dict[str, Any]] = {}
    for binding in config["source_bindings"]:
        source_id = str(binding["source_id"])
        source = _load_source(binding, root)
        _validate_source_semantics(source_id, source)
        loaded_sources[source_id] = source
    _require(set(loaded_sources) == set(SOURCE_IDS), "not all sources validated")

    source_file = root / SOURCE_PATH
    test_file = root / TEST_PATH
    _require(source_file.is_file(), "implementation source is missing")
    _require(test_file.is_file(), "implementation test is missing")
    terms = config["term_provenance_ledger"]
    gates = config["health_and_consistency_gates"]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "preflight_id": config["preflight_id"],
        "status": "blocked_covariant_template_defined_health_and_lensing_not_established",
        "decision": DECISION,
        "config_binding": {
            "path": config_path.as_posix(),
            "file_sha256": _file_sha(root / config_path),
            "content_sha256": _sha(config),
        },
        "implementation_binding": {
            "source_path": SOURCE_PATH.as_posix(),
            "source_sha256": _file_sha(source_file),
            "test_path": TEST_PATH.as_posix(),
            "test_sha256": _file_sha(test_file),
        },
        "source_bindings": config["source_bindings"],
        "role_mapping": config["role_mapping"],
        "action_family": config["action_family"],
        "conventions_and_dimensions": config["conventions_and_dimensions"],
        "term_provenance_ledger": terms,
        "field_equations_and_symbolic_contract": config["field_equations_and_symbolic_contract"],
        "degrees_of_freedom": config["degrees_of_freedom"],
        "conservation_identity": config["conservation_identity"],
        "weak_field_matter_and_lensing": config["weak_field_matter_and_lensing"],
        "health_and_consistency_gates": gates,
        "exact_parameter_and_limit_gates": config["exact_parameter_and_limit_gates"],
        "feasibility_adjudication": config["feasibility_adjudication"],
        "counts": {
            "source_receipts_validated": len(loaded_sources),
            "terms_total": len(terms),
            "known_rewrite_terms": sum(
                item["provenance_label"] == "known_rewrite" for item in terms
            ),
            "known_combination_terms": sum(
                item["provenance_label"] == "known_combination" for item in terms
            ),
            "potentially_new_synthesis_terms": sum(
                item["provenance_label"] == "potentially_new_synthesis" for item in terms
            ),
            "health_gates_total": len(gates),
            "template_level_gates_passed": sum(
                item["status"] == "PASS_TEMPLATE_LEVEL" for item in gates
            ),
            "health_gates_blocked": sum(item["status"].startswith("BLOCKED_") for item in gates),
            "conditional_tensor_dof": config["degrees_of_freedom"]["tensor_polarizations"],
            "conditional_scalar_dof": config["degrees_of_freedom"]["scalar_fields"],
        },
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "This is a frozen covariant EFT feasibility template, not a completed or healthy theory.",
            "The global nonlinear P function, nonspherical RAR mapping, parameters, range, cutoff, branch, and boundary conditions are not frozen.",
            "The fixed 1/m_chi range and B^2 amplitude follow only for locally constant Z_chi; gradients of Z_chi and phi-background mixing are explicit unresolved blockers.",
            "Conformal scalar coupling cancels from the leading weak-field lensing sum; the disformal completion has unresolved GW-speed, screening, constraint, and hyperbolicity gates.",
            "No symbolic algebra system has verified the full functional variation, principal symbol, Hamiltonian constraints, or Noether identity.",
            "Potentially-new-synthesis labels flag combinations for later prior-art review and never establish novelty.",
            "No observational data, target row, likelihood, network, model, paid service, or GPU was used.",
        ],
    }
    receipt["content_sha256"] = _sha(receipt)
    validate_receipt(receipt, config)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema changed")
    _validate_embedded_content_hash(
        receipt, str(receipt.get("content_sha256")), "theory preflight receipt"
    )
    _require(receipt.get("decision") == DECISION, "receipt decision changed")
    _require(
        receipt.get("config_binding", {}).get("content_sha256") == _sha(config), "config unbound"
    )
    _require(receipt.get("action_family") == config["action_family"], "action changed")
    _require(
        receipt.get("field_equations_and_symbolic_contract")
        == config["field_equations_and_symbolic_contract"],
        "equation contract changed",
    )
    _require(receipt.get("claim_boundary") == config["claim_boundary"], "claims changed")
    _require(
        receipt.get("zero_access_and_compute") == config["zero_access_and_compute"],
        "zero-access accounting changed",
    )
    counts = receipt.get("counts")
    _require(isinstance(counts, dict), "receipt counts missing")
    _require(
        counts
        == {
            "source_receipts_validated": 3,
            "terms_total": 7,
            "known_rewrite_terms": 1,
            "known_combination_terms": 4,
            "potentially_new_synthesis_terms": 2,
            "health_gates_total": 10,
            "template_level_gates_passed": 1,
            "health_gates_blocked": 9,
            "conditional_tensor_dof": 2,
            "conditional_scalar_dof": 2,
        },
        "receipt counts changed",
    )


def _atomic_no_replace(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return "EXISTING_IDENTICAL"
        raise GravityMatterLensingTheoryPreflightError(
            f"refusing to overwrite non-identical output: {path.as_posix()}"
        )
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise GravityMatterLensingTheoryPreflightError(
                f"concurrent creator won; output preserved: {path.as_posix()}"
            ) from exc
        return "CREATED"
    finally:
        temp.unlink(missing_ok=True)


def write_receipt(
    root: Path = Path("."),
    config_path: Path = CONFIG_PATH,
    output_path: Path = OUTPUT_PATH,
) -> tuple[dict[str, Any], str]:
    _require(output_path == OUTPUT_PATH, "output path is not the frozen confined path")
    receipt = build_receipt(root, config_path)
    status = _atomic_no_replace(root / output_path, _canonical_bytes(receipt))
    return receipt, status


def check_receipt(
    root: Path = Path("."),
    config_path: Path = CONFIG_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    config = load_config(root / config_path)
    expected = build_receipt(root, config_path)
    stored = _read_json(root / output_path)
    validate_receipt(stored, config)
    _require(stored == expected, "stored receipt differs from deterministic rebuild")
    return stored


def _summary(receipt: Mapping[str, Any], publication: str | None = None) -> dict[str, Any]:
    summary = {
        "valid": True,
        "status": receipt["status"],
        "decision": receipt["decision"],
        "content_sha256": receipt["content_sha256"],
        "healthy_action_completed": receipt["claim_boundary"]["healthy_action_completed"],
        "matter_and_lensing_jointly_passed": receipt["feasibility_adjudication"][
            "matter_and_lensing_jointly_passed"
        ],
        "observational_execution_unlocked": receipt["claim_boundary"][
            "observational_execution_unlocked"
        ],
        "scientific_claim_allowed": receipt["claim_boundary"]["scientific_claim_allowed"],
        "health_gates_blocked": receipt["counts"]["health_gates_blocked"],
        "observational_files_opened": receipt["zero_access_and_compute"][
            "observational_files_opened"
        ],
    }
    if publication is not None:
        summary["publication"] = publication
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("write", "check", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, default=Path("."))
        command.add_argument("--config", type=Path, default=CONFIG_PATH)
        command.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    try:
        if args.command == "write":
            receipt, publication = write_receipt(args.root, args.config, args.output)
            result = _summary(receipt, publication)
        else:
            receipt = check_receipt(args.root, args.config, args.output)
            result = _summary(receipt)
        print(json.dumps(result, sort_keys=True))
        return 0
    except GravityMatterLensingTheoryPreflightError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
