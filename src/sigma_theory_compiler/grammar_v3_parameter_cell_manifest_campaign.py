from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-parameter-cell-manifest-config-1.0"
MANIFEST_SCHEMA = "sigma-grammar-v3-parameter-cell-manifest-1.0"
CELL_SCHEMA = "sigma-grammar-v3-expanded-parameter-cell-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _fraction(value: str, *, require_canonical: bool = True) -> Fraction:
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid exact rational") from error
    if require_canonical and str(parsed) != value:
        raise ValueError("noncanonical exact rational")
    return parsed


def _rational_range(spec: dict[str, Any]) -> list[str]:
    if set(spec) != {"numerator_start", "numerator_stop", "denominator"}:
        raise ValueError("rational range fields are invalid")
    start, stop, denominator = (
        int(spec["numerator_start"]),
        int(spec["numerator_stop"]),
        int(spec["denominator"]),
    )
    if denominator <= 0 or start > stop:
        raise ValueError("rational range is empty or has invalid denominator")
    return [str(Fraction(numerator, denominator)) for numerator in range(start, stop + 1)]


def _axis(axis: Any) -> list[str]:
    values = _rational_range(axis) if isinstance(axis, dict) else list(axis)
    if not values:
        raise ValueError("parameter grid axis is empty")
    parsed = [_fraction(str(value)) for value in values]
    if len(parsed) != len(set(parsed)):
        raise ValueError("parameter grid contains duplicate or equivalent rationals")
    return [str(value) for value in parsed]


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "source_seed_manifest",
        "evaluator_semantics_bindings",
        "finite_budget",
        "family_grids",
        "negative_controls",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("grammar-v3 parameter-cell manifest config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("parameter-cell manifest eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("parameter-cell manifest enabled paid LLM calls")
    budget = config.get("finite_budget", {})
    if set(budget) != {"maximum_cells", "chunk_size", "family_quotas"}:
        raise ValueError("parameter-cell manifest budget fields are invalid")
    maximum = int(budget["maximum_cells"])
    if not 128 <= maximum <= 512 or not 1 <= int(budget["chunk_size"]) <= 64:
        raise ValueError("parameter-cell manifest budget is not finite and defensible")
    quotas = budget["family_quotas"]
    if set(quotas) != set(config["family_grids"]) or sum(map(int, quotas.values())) != maximum:
        raise ValueError("parameter-cell family quotas do not equal the finite cap")
    if len(config["family_grids"]) != 4:
        raise ValueError("parameter-cell manifest must use the four reviewed enabled families")


def _verify_file_binding(root: Path, binding: dict[str, Any], label: str) -> Path:
    if not {"path", "file_sha256"}.issubset(binding) or set(binding) - {
        "path",
        "file_sha256",
        "content_sha256",
    }:
        raise ValueError(f"{label} binding fields are invalid")
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} binding escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _family_points(family_id: str, grid: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if family_id == "AETHER_K1234_PARAMETER_CELL":
        axes = [_axis(grid[key]) for key in ("c1", "c2", "c3", "c4")]
        for c1, c2, c3, c4 in itertools.product(*axes):
            if not (
                Fraction(0) < _fraction(c1) <= Fraction(1, 4)
                and Fraction(0) <= _fraction(c2) <= Fraction(1, 8)
                and Fraction(-1, 16) <= _fraction(c3) <= Fraction(1, 8)
                and Fraction(0) <= _fraction(c4) <= Fraction(1, 16)
            ):
                raise ValueError("Aether parameter cell violates its bounded coefficient domain")
            yield {
                "parameters": {"c1": c1, "c2": c2, "c3": c3, "c4": c4},
                "rational_coordinates": {"c1": c1, "c2": c2, "c3": c3, "c4": c4},
                "domain_contract": "bounded_coefficients_only; formal stability remains unresolved",
            }
    elif family_id == "KESSENCE_G2_CONVEX":
        for alpha, x_max in itertools.product(_axis(grid["alpha"]), _axis(grid["X_max"])):
            if _fraction(alpha) not in {Fraction(1, 8), Fraction(1, 4)}:
                raise ValueError("k-essence coefficient is outside unchanged evaluator semantics")
            if not Fraction(0) < _fraction(x_max) <= 1:
                raise ValueError("k-essence X domain is outside the certified convex interval")
            yield {
                "parameters": {
                    "G2": f"X_phi+({alpha})*X_phi^2",
                    "X_domain": f"0<=X_phi<={x_max}",
                },
                "rational_coordinates": {"alpha": alpha, "X_max": x_max},
                "domain_contract": "G2_X>=1 and G2_X+2XG2_XX>=1 on the declared cell",
            }
    elif family_id == "CUBIC_HORNDESKI_G3_WEAK_CELL":
        for beta in _axis(grid["beta"]):
            if not Fraction(0) < _fraction(beta) <= Fraction(1, 100):
                raise ValueError("cubic Horndeski coefficient exceeds the weak-cell envelope")
            yield {
                "parameters": {
                    "G2": "X_phi",
                    "G3": f"({beta})*X_phi",
                    "jet_domain": f"dimensionless derivative ratios<={beta}",
                },
                "rational_coordinates": {"beta": beta, "jet_max": beta},
                "domain_contract": "weak derivative cell only; common-cone proof remains unresolved",
            }
    elif family_id == "CONFORMAL_G4_PHI_SCALAR_TENSOR":
        xi_values = _axis(grid["xi"])
        if xi_values != ["1/100"]:
            raise ValueError("G4 xi would change the existing reviewed evaluator semantics")
        for xi, phi_max in itertools.product(xi_values, _axis(grid["phi_max"])):
            if not Fraction(0) < _fraction(phi_max) <= 1:
                raise ValueError("G4 phi domain exceeds the reviewed local interval")
            yield {
                "parameters": {
                    "G2": "X_phi",
                    "G4": f"1/2+({xi})*phi^2",
                    "phi_domain": f"abs(phi)<={phi_max}",
                },
                "rational_coordinates": {"xi": xi, "phi_max": phi_max},
                "domain_contract": "G4>=1/2 locally; global lapse and energy are not inferred",
            }
    else:
        raise ValueError("unreviewed grammar-v3 family requested")


def _cell(
    ordinal: int,
    family_cell_index: int,
    family: dict[str, Any],
    point: dict[str, Any],
    source: dict[str, str],
) -> dict[str, Any]:
    body = {
        "schema_version": CELL_SCHEMA,
        "ordinal": ordinal,
        "family_cell_index": family_cell_index,
        "source_seed_manifest_content_sha256": source["content_sha256"],
        "family_id": family["family_id"],
        "family_lineage_sha256": family["family_lineage_sha256"],
        "theory_contract": family["theory_contract"],
        "operator_atoms": family["operator_atoms"],
        **point,
        "data_eligibility": dict(ELIGIBILITY),
    }
    lineage = _sha(body)
    return {
        **body,
        "parameter_cell_id": "G3PC-" + lineage[:24],
        "parameter_cell_lineage_sha256": lineage,
    }


def _negative_control_audit(config: dict[str, Any], first: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for control in config["negative_controls"]:
        kind = control["kind"]
        reason = None
        if kind == "duplicate_cell":
            reason = "duplicate_parameter_cell_id"
        elif kind == "equivalent_rational":
            if Fraction(control["value"]) == Fraction(control["canonical_value"]):
                reason = "equivalent_parameter_cell_after_rational_normalization"
        elif kind == "invalid_parameter_domain":
            if Fraction(control["alpha"]) <= 0:
                reason = "invalid_parameter_domain"
        elif kind == "budget_overflow":
            if int(control["requested_cells"]) > int(config["finite_budget"]["maximum_cells"]):
                reason = "finite_cell_budget_overflow"
        elif kind == "forbidden_data_input":
            if control["data_eligibility"] != ELIGIBILITY:
                reason = "forbidden_data_input"
        elif kind == "invalid_rational":
            try:
                _fraction(control["value"], require_canonical=False)
            except ValueError:
                reason = "invalid_exact_rational"
        if reason is None:
            raise ValueError(f"negative control unexpectedly admitted: {control['control_id']}")
        evidence = {
            "control_id": control["control_id"],
            "kind": kind,
            "decision": "reject",
            "reason": reason,
            "reference_cell_id": first["parameter_cell_id"] if kind == "duplicate_cell" else None,
        }
        results.append({**evidence, "content_sha256": _sha(evidence)})
    return results


def build_parameter_cell_manifest(config: dict[str, Any], root: str | Path) -> dict[str, Any]:
    _validate_config(config)
    root = Path(root).resolve()
    source_binding = config["source_seed_manifest"]
    source_path = _verify_file_binding(root, source_binding, "source seed manifest")
    source_manifest = _load(source_path)
    source_body = {key: value for key, value in source_manifest.items() if key != "content_sha256"}
    if (
        source_manifest.get("content_sha256") != source_binding["content_sha256"]
        or _sha(source_body) != source_binding["content_sha256"]
        or source_manifest.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("source seed manifest content or eligibility mismatch")
    semantics = []
    for binding in config["evaluator_semantics_bindings"]:
        path = _verify_file_binding(root, binding, "evaluator semantics")
        semantics.append({"path": binding["path"], "file_sha256": _file_sha(path)})
    enabled = {
        family["family_id"]: family
        for family in source_manifest["typed_family_seeds"]
        if family["enabled_for_generation"]
    }
    if set(enabled) != set(config["family_grids"]):
        raise ValueError("expanded grids differ from the four reviewed enabled families")
    cells = []
    family_counts: Counter[str] = Counter()
    for family_id in config["family_grids"]:
        family = enabled[family_id]
        points = list(_family_points(family_id, config["family_grids"][family_id]))
        quota = int(config["finite_budget"]["family_quotas"][family_id])
        if len(points) != quota:
            raise ValueError(f"generated family cell count differs from quota: {family_id}")
        for index, point in enumerate(points):
            cells.append(
                _cell(
                    len(cells),
                    index,
                    family,
                    point,
                    {"content_sha256": source_manifest["content_sha256"]},
                )
            )
            family_counts[family_id] += 1
    maximum = int(config["finite_budget"]["maximum_cells"])
    identities = [(cell["parameter_cell_id"], cell["parameter_cell_lineage_sha256"]) for cell in cells]
    equivalence = [
        _sha(
            {
                "family_id": cell["family_id"],
                "rational_coordinates": cell["rational_coordinates"],
                "domain_contract": cell["domain_contract"],
            }
        )
        for cell in cells
    ]
    if len(cells) != maximum or len(set(identities)) != maximum or len(set(equivalence)) != maximum:
        raise ValueError("expanded cells are over budget, duplicated, or equivalent")
    chunk_size = int(config["finite_budget"]["chunk_size"])
    chunks = []
    for start in range(0, maximum, chunk_size):
        selected = cells[start : start + chunk_size]
        chunk_body = {
            "chunk_index": len(chunks),
            "range": {"start": start, "stop": start + len(selected)},
            "cell_identity_root_sha256": _sha(
                [
                    [cell["parameter_cell_id"], cell["parameter_cell_lineage_sha256"]]
                    for cell in selected
                ]
            ),
        }
        chunks.append({**chunk_body, "content_sha256": _sha(chunk_body)})
    negatives = _negative_control_audit(config, cells[0])
    samples = []
    for family_id in config["family_grids"]:
        family_cells = [cell for cell in cells if cell["family_id"] == family_id]
        for cell in (family_cells[0], family_cells[-1]):
            samples.append(
                {
                    "parameter_cell_id": cell["parameter_cell_id"],
                    "parameter_cell_lineage_sha256": cell[
                        "parameter_cell_lineage_sha256"
                    ],
                    "ordinal": cell["ordinal"],
                    "family_cell_index": cell["family_cell_index"],
                    "family_id": cell["family_id"],
                    "rational_coordinates": cell["rational_coordinates"],
                    "parameters_sha256": _sha(cell["parameters"]),
                }
            )
    compact_grid_contract = {
        "family_order": list(config["family_grids"]),
        "family_grids": config["family_grids"],
        "family_quotas": config["finite_budget"]["family_quotas"],
    }
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_seed_manifest": source_binding,
        "source_seed_manifest_content_sha256": source_manifest["content_sha256"],
        "evaluator_semantics_bindings": semantics,
        "evaluator_semantics_changed": False,
        "parameter_cell_count": len(cells),
        "family_cell_counts": dict(sorted(family_counts.items())),
        "finite_budget": config["finite_budget"],
        "compact_grid_contract": compact_grid_contract,
        "compact_grid_contract_sha256": _sha(compact_grid_contract),
        "parameter_cell_registry_root_sha256": _sha(identities),
        "parameter_cell_equivalence_root_sha256": _sha(equivalence),
        "chunks": chunks,
        "chunk_registry_root_sha256": _sha(chunks),
        "sample_cells": samples,
        "negative_control_results": negatives,
        "negative_control_counts": {"reject": len(negatives)},
        "formal_evaluation_performed": False,
        "scientific_decision_counts": {},
        "next_execution_hook": {
            "callable": (
                "sigma_theory_compiler.grammar_v3_parameter_cell_manifest_campaign:"
                "iter_parameter_cells"
            ),
            "required_next_adapter": (
                "a new reviewed candidate-compilation campaign bound to this 256-cell registry; "
                "the existing six-seed callback must reject unknown cell ids"
            ),
        },
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
        "paid_llm_spend_usd": 0.0,
    }
    return {**body, "content_sha256": _sha(body)}


def iter_parameter_cells(
    manifest: dict[str, Any], source_seed_manifest: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("content_sha256") != _sha(body):
        raise ValueError("expanded parameter-cell manifest hash or schema mismatch")
    if manifest.get("data_eligibility") != ELIGIBILITY or manifest.get("formal_evaluation_performed") is not False:
        raise ValueError("expanded parameter-cell manifest opened a forbidden gate")
    if source_seed_manifest.get("content_sha256") != manifest["source_seed_manifest_content_sha256"]:
        raise ValueError("expanded parameter-cell source manifest changed")
    enabled = {
        family["family_id"]: family
        for family in source_seed_manifest["typed_family_seeds"]
        if family["enabled_for_generation"]
    }
    cells = []
    contract = manifest["compact_grid_contract"]
    if _sha(contract) != manifest["compact_grid_contract_sha256"]:
        raise ValueError("expanded parameter-cell compact grid contract changed")
    for family_id in contract["family_order"]:
        points = list(_family_points(family_id, contract["family_grids"][family_id]))
        if len(points) != int(contract["family_quotas"][family_id]):
            raise ValueError("expanded parameter-cell family quota changed")
        for index, point in enumerate(points):
            cells.append(
                _cell(
                    len(cells),
                    index,
                    enabled[family_id],
                    point,
                    {"content_sha256": source_seed_manifest["content_sha256"]},
                )
            )
    identities = [(cell["parameter_cell_id"], cell["parameter_cell_lineage_sha256"]) for cell in cells]
    if len(cells) != manifest["parameter_cell_count"] or _sha(identities) != manifest[
        "parameter_cell_registry_root_sha256"
    ]:
        raise ValueError("expanded parameter-cell registry does not reproduce")
    yield from cells
