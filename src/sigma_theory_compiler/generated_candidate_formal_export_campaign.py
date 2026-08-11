"""Exact action export and network-isolated Cadabra parsing for scalable candidates.

This campaign deliberately separates three claims:

* all current scalable typed actions have an exact generated Cadabra action expression;
* those 163 expressions parse and canonicalise in one bounded network-isolated process;
* metric variation is only routed to hash-bound reviewed adapters here, not re-derived.

The last distinction keeps the campaign fail-closed for future operator families.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .formal_backend import probe_cadabra
from .promotion_orchestrator import ELIGIBILITY
from .scalable_formal_candidate_evidence_export import (
    iter_scalable_formal_candidate_evidence_records,
    validate_scalable_formal_candidate_evidence_export,
)

CONFIG_SCHEMA = "sigma-generated-candidate-formal-export-config-1.0"
EXPORT_SCHEMA = "sigma-generated-candidate-formal-export-1.0"
RECORD_SCHEMA = "sigma-generated-candidate-formal-export-record-1.0"
SANDBOX_SCHEMA = "sigma-generated-candidate-cadabra-sandbox-receipt-1.0"
MARKER = "SIGMA_ALL_163_GENERATED_ACTIONS_SANDBOX_OK"

FAMILY_ATOMS = {
    "AETHER_K1234_PARAMETER_CELL": (
        "EH_R",
        "AETHER_K1",
        "AETHER_K2",
        "AETHER_K3",
        "AETHER_K4",
        "UNIT_VECTOR_CONSTRAINT",
    ),
    "KESSENCE_G2_CONVEX": ("EH_R", "G2_PHI_X"),
    "CUBIC_HORNDESKI_G3_WEAK_CELL": (
        "EH_R",
        "G2_PHI_X",
        "G3_PHI_X_BOX_PHI",
    ),
    "CONFORMAL_G4_PHI_SCALAR_TENSOR": ("G2_PHI_X", "G4_PHI_R"),
}

METRIC_ADAPTERS = {
    "AETHER_K1234_PARAMETER_CELL": (
        "aether_status",
        "reviewed_complete_K1_K2_K3_K4_fixed_covector_metric_variation",
    ),
    "KESSENCE_G2_CONVEX": (
        "g2_followup",
        "reviewed_arbitrary_G2_metric_variation_and_noether",
    ),
    "CUBIC_HORNDESKI_G3_WEAK_CELL": (
        "g3_status",
        "reviewed_arbitrary_G2_G3_metric_variation_and_noether",
    ),
    "CONFORMAL_G4_PHI_SCALAR_TENSOR": (
        "g4_followup",
        "reviewed_conformal_G4_metric_scalar_equivalence",
    ),
}


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


def _bound_path(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) - {"path", "file_sha256", "content_sha256"} or not {
        "path",
        "file_sha256",
    }.issubset(binding):
        raise ValueError(f"{label} binding fields are invalid")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _bound_json(root: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    value = _load(_bound_path(root, binding, label))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != binding.get("content_sha256") or _sha(body) != binding.get(
        "content_sha256"
    ):
        raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "source_export",
            "campaign_source",
            "budget",
            "sandbox_policy",
            "data_eligibility",
            "external_paid_llm_calls",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("generated formal-export config is invalid")
    budget = config.get("budget", {})
    if set(budget) != {
        "maximum_candidates",
        "maximum_batch_script_bytes",
        "maximum_stdout_bytes",
        "maximum_stderr_bytes",
        "maximum_wall_seconds",
        "maximum_paid_llm_spend_usd",
    } or (
        int(budget.get("maximum_candidates", 0)) != 163
        or not 65_536 <= int(budget.get("maximum_batch_script_bytes", 0)) <= 2_097_152
        or not 16_384 <= int(budget.get("maximum_stdout_bytes", 0)) <= 2_097_152
        or not 4_096 <= int(budget.get("maximum_stderr_bytes", 0)) <= 1_048_576
        or not 5 <= int(budget.get("maximum_wall_seconds", 0)) <= 300
        or float(budget.get("maximum_paid_llm_spend_usd", -1)) != 0.0
    ):
        raise ValueError("generated formal-export budget is invalid")
    if config.get("sandbox_policy") != {
        "backend": "cadabra2_wsl_local",
        "network_namespace": "unshare_user_and_network",
        "shell": False,
        "generated_input_only": True,
        "timeout_required": True,
        "output_caps_required": True,
    }:
        raise ValueError("generated formal-export sandbox policy changed")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("generated formal export opened forbidden data")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("generated formal export enabled paid LLM calls")


def _rational(value: str) -> sp.Rational:
    parsed = sp.sympify(value)
    if not isinstance(parsed, sp.Rational):
        raise TypeError(f"parameter is not an exact rational: {value}")
    return parsed


def _cadabra_rational(value: str | sp.Rational) -> str:
    rational = _rational(str(value)) if not isinstance(value, sp.Rational) else value
    return str(rational.p) if rational.q == 1 else f"({rational.p}/{rational.q})"


def _coefficient_from_polynomial(value: str, *, linear_only: bool = False) -> sp.Rational:
    x = sp.Symbol("X_phi")
    parsed = sp.sympify(value, locals={"X_phi": x})
    polynomial = sp.Poly(parsed, x)
    expected_degree = 1 if linear_only else 2
    if polynomial.degree() != expected_degree or polynomial.coeff_monomial(x) != 1:
        raise ValueError(f"unsupported G2/G3 polynomial: {value}")
    if linear_only:
        if polynomial.coeff_monomial(1) != 0:
            raise ValueError(f"unsupported linear polynomial: {value}")
        return sp.Rational(1)
    if polynomial.coeff_monomial(1) != 0:
        raise ValueError(f"unsupported G2 constant term: {value}")
    quadratic = polynomial.coeff_monomial(x**2)
    if not isinstance(quadratic, sp.Rational) or quadratic <= 0:
        raise ValueError(f"unsupported G2 quadratic coefficient: {value}")
    return quadratic


def _g3_beta(value: str) -> sp.Rational:
    x = sp.Symbol("X_phi")
    parsed = sp.sympify(value, locals={"X_phi": x})
    polynomial = sp.Poly(parsed, x)
    beta = polynomial.coeff_monomial(x)
    if (
        polynomial.degree() != 1
        or polynomial.coeff_monomial(1) != 0
        or not isinstance(beta, sp.Rational)
    ):
        raise ValueError(f"unsupported G3 polynomial: {value}")
    return beta


def _action_expression(record: Mapping[str, Any]) -> str:
    family = str(record["family_id"])
    formula = record["theory_formula_inputs"]
    atoms = tuple(item["atom"] for item in formula["ordered_operator_densities"])
    if atoms != FAMILY_ATOMS.get(family):
        raise ValueError(f"operator sequence changed for {record['candidate_id']}")
    parameters = formula["parameters"]
    if family == "AETHER_K1234_PARAMETER_CELL":
        if set(parameters) != {"c1", "c2", "c3", "c4"}:
            raise ValueError("Aether parameter set changed")
        coefficients = {key: _rational(parameters[key]) for key in sorted(parameters)}
        terms = ["Mpl^2 R/2"]
        for key, atom, sign in (
            ("c1", "K1", -1),
            ("c2", "K2", -1),
            ("c3", "K3", -1),
            ("c4", "K4", 1),
        ):
            value = sign * coefficients[key] / 2
            if value:
                terms.append(f"{_cadabra_rational(value)} Mpl^2 {atom}")
        terms.append("lambda (U2 + 1)")
        return "sqrtg (" + " + ".join(terms).replace("+ -", "- ") + ")"
    if family == "KESSENCE_G2_CONVEX":
        if set(parameters) != {"G2", "X_domain"} or parameters["X_domain"] != "0<=X_phi<=1/32":
            raise ValueError("G2 parameter/domain contract changed")
        q = _coefficient_from_polynomial(parameters["G2"])
        return f"sqrtg (Mpl^2 R/2 + Lphi^4 (Xphi + {_cadabra_rational(q)} Xphi^2))"
    if family == "CUBIC_HORNDESKI_G3_WEAK_CELL":
        if set(parameters) != {"G2", "G3", "jet_domain"} or parameters["G2"] != "X_phi":
            raise ValueError("G3 parameter contract changed")
        beta = _g3_beta(parameters["G3"])
        if parameters["jet_domain"] != f"dimensionless derivative ratios<={beta}":
            raise ValueError("G3 jet-domain lineage changed")
        return f"sqrtg (Mpl^2 R/2 + Lphi^4 Xphi - {_cadabra_rational(beta)} Lphi Xphi Bphi)"
    if family == "CONFORMAL_G4_PHI_SCALAR_TENSOR":
        if parameters != {
            "G2": "X_phi",
            "G4": "1/2+(1/100)*phi^2",
            "phi_domain": "abs(phi)<=1/32",
        }:
            raise ValueError("G4 action/domain contract changed")
        return "sqrtg (Lphi^4 Xphi + Lphi^2 (1/2 + phi^2/100) R)"
    raise ValueError(f"unsupported scalable family: {family}")


def render_cadabra_batch(records: list[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    lines = [
        "{a,b,c,d}::Indices(position=free).",
        "\\nabla{#}::Derivative.",
        "R::Depends(\\nabla{#}).",
        "K1::Depends(\\nabla{#}).",
        "K2::Depends(\\nabla{#}).",
        "K3::Depends(\\nabla{#}).",
        "K4::Depends(\\nabla{#}).",
        "Xphi::Depends(\\nabla{#}).",
        "Bphi::Depends(\\nabla{#}).",
        "phi::Depends(\\nabla{#}).",
        "",
    ]
    rendered = []
    for index, record in enumerate(records):
        expression = _action_expression(record)
        symbol = f"action{index:03d}"
        lines.extend((f"{symbol} := {expression};", f"canonicalise({symbol});"))
        rendered.append(
            {
                "candidate_id": record["candidate_id"],
                "action_sha256": record["action_sha256"],
                "formula_inputs_sha256": record["theory_formula_inputs"]["formula_inputs_sha256"],
                "cadabra_symbol": symbol,
                "cadabra_action_expression": expression,
                "cadabra_action_expression_sha256": hashlib.sha256(expression.encode()).hexdigest(),
            }
        )
    lines.extend((f'print("{MARKER}");', ""))
    return "\n".join(lines), rendered


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").casefold()
    if not drive or not re.fullmatch(r"[a-z]", drive):
        raise ValueError("sandbox path is not on a Windows drive")
    return f"/mnt/{drive}{resolved.as_posix().split(':', 1)[1]}"


def execute_cadabra_batch_sandbox(
    script_text: str,
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    budget = config["budget"]
    encoded = script_text.encode()
    if len(encoded) > int(budget["maximum_batch_script_bytes"]):
        raise ValueError("generated Cadabra batch exceeds script budget")
    backend = probe_cadabra(root)
    if not backend.get("available") or backend.get("mode") != "wsl-local":
        raise RuntimeError("network-isolated generated export requires wsl-local Cadabra")
    wsl = shutil.which("wsl")
    unshare = "unshare"
    if not wsl:
        raise RuntimeError("WSL disappeared after Cadabra probe")
    cadabra_root = Path(str(backend["root"])).resolve()
    with tempfile.TemporaryDirectory(prefix="sigma-generated-action-") as temporary:
        script = Path(temporary) / "generated-actions.cdb"
        script.write_text(script_text, encoding="utf-8")
        root_wsl = _windows_to_wsl(cadabra_root)
        command = [
            wsl,
            "-d",
            "Ubuntu-24.04",
            "--",
            unshare,
            "-Urn",
            "--",
            "env",
            f"PYTHONPATH={root_wsl}/usr/lib/python3/dist-packages",
            f"LD_LIBRARY_PATH={root_wsl}/usr/lib/x86_64-linux-gnu",
            f"{root_wsl}/usr/bin/cadabra2",
            _windows_to_wsl(script),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=int(budget["maximum_wall_seconds"]),
        )
    stdout = completed.stdout.encode()
    stderr = completed.stderr.encode()
    if len(stdout) > int(budget["maximum_stdout_bytes"]):
        raise ValueError("generated Cadabra stdout exceeds budget")
    if len(stderr) > int(budget["maximum_stderr_bytes"]):
        raise ValueError("generated Cadabra stderr exceeds budget")
    marker_count = completed.stdout.count(MARKER)
    receipt_body = {
        "schema_version": SANDBOX_SCHEMA,
        "status": "pass" if completed.returncode == 0 and marker_count == 1 else "reject",
        "backend_mode": "wsl-local",
        "backend_version": backend.get("version"),
        "network_namespace_created": True,
        "user_namespace_created": True,
        "shell_invoked": False,
        "return_code": completed.returncode,
        "marker": MARKER,
        "marker_count": marker_count,
        "batch_script_sha256": hashlib.sha256(encoded).hexdigest(),
        "batch_script_bytes": len(encoded),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_bytes": len(stdout),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_bytes": len(stderr),
        "timeout_seconds": int(budget["maximum_wall_seconds"]),
    }
    receipt = {**receipt_body, "content_sha256": _sha(receipt_body)}
    if receipt["status"] != "pass":
        raise RuntimeError("generated Cadabra action batch failed closed")
    return receipt


def _validate_sandbox_receipt(receipt: Mapping[str, Any], script: str) -> None:
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if (
        receipt.get("schema_version") != SANDBOX_SCHEMA
        or receipt.get("content_sha256") != _sha(body)
        or receipt.get("status") != "pass"
        or receipt.get("network_namespace_created") is not True
        or receipt.get("user_namespace_created") is not True
        or receipt.get("shell_invoked") is not False
        or receipt.get("return_code") != 0
        or receipt.get("marker") != MARKER
        or receipt.get("marker_count") != 1
        or receipt.get("batch_script_sha256") != hashlib.sha256(script.encode()).hexdigest()
        or receipt.get("batch_script_bytes") != len(script.encode())
    ):
        raise ValueError("generated Cadabra sandbox receipt is invalid")


def validate_generated_candidate_formal_export(export: Mapping[str, Any]) -> None:
    body = {key: value for key, value in export.items() if key != "content_sha256"}
    if export.get("schema_version") != EXPORT_SCHEMA or export.get("content_sha256") != _sha(body):
        raise ValueError("generated formal export content hash mismatch")
    records = export.get("candidate_records")
    if not isinstance(records, list) or len(records) != 163:
        raise ValueError("generated formal export candidate count changed")
    if len({record.get("candidate_id") for record in records}) != 163:
        raise ValueError("generated formal export candidate identity collision")
    family_counts = dict(sorted(Counter(record["family_id"] for record in records).items()))
    if (
        family_counts
        != {
            "AETHER_K1234_PARAMETER_CELL": 128,
            "CONFORMAL_G4_PHI_SCALAR_TENSOR": 1,
            "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
            "KESSENCE_G2_CONVEX": 2,
        }
        or export.get("family_counts") != family_counts
    ):
        raise ValueError("generated formal export family accounting changed")
    for record in records:
        record_body = {key: value for key, value in record.items() if key != "content_sha256"}
        if record.get("schema_version") != RECORD_SCHEMA or record.get("content_sha256") != _sha(
            record_body
        ):
            raise ValueError("generated formal-export record hash mismatch")
        metric = record.get("metric_variation_route", {})
        formula = record.get("theory_formula_inputs", {})
        formula_body = {
            "fields": formula.get("fields"),
            "parameters": formula.get("parameters"),
            "ordered_operator_densities": formula.get("ordered_operator_densities"),
            "action_content_sha256": formula.get("action_content_sha256"),
        }
        pseudo_source = {
            "candidate_id": record.get("candidate_id"),
            "family_id": record.get("family_id"),
            "action_sha256": record.get("action_sha256"),
            "theory_formula_inputs": formula,
        }
        if (
            formula.get("formula_inputs_sha256") != _sha(formula_body)
            or formula.get("action_content_sha256") != record.get("action_sha256")
            or record.get("formula_inputs_sha256") != formula.get("formula_inputs_sha256")
            or record.get("cadabra_action_expression") != _action_expression(pseudo_source)
            or metric.get("status") != "reviewed_adapter_bound_not_executed_by_this_campaign"
            or metric.get("metric_variation_executed_by_this_campaign") is not False
            or record.get("formal_pass_inferred") is not False
            or record.get("cadabra_action_expression_sha256")
            != hashlib.sha256(str(record.get("cadabra_action_expression", "")).encode()).hexdigest()
        ):
            raise ValueError("generated action export promoted metric/formal evidence")
    if export.get("action_export_counts") != {
        "exact_rendered": 163,
        "sandbox_parsed_and_canonicalised": 163,
        "rejected": 0,
    } or export.get("metric_variation_counts") != {
        "reviewed_adapter_routes_bound": 163,
        "executed_by_this_campaign": 0,
        "formal_passes_inferred": 0,
    }:
        raise ValueError("generated formal-export claim counts changed")
    script = str(export.get("cadabra_batch_script"))
    _validate_sandbox_receipt(export.get("sandbox_receipt", {}), script)
    if export.get("candidate_record_registry_root_sha256") != _sha(
        [record["content_sha256"] for record in records]
    ):
        raise ValueError("generated formal-export registry root changed")
    if export.get("data_eligibility") != {**ELIGIBILITY, "passed": True}:
        raise ValueError("generated formal export opened forbidden data")


def build_generated_candidate_formal_export(
    config: Mapping[str, Any],
    root: str | Path,
    *,
    sandbox_executor: Callable[
        [str, Path, Mapping[str, Any]], dict[str, Any]
    ] = execute_cadabra_batch_sandbox,
) -> dict[str, Any]:
    _validate_config(config)
    root = Path(root).resolve()
    _bound_path(root, config["campaign_source"], "campaign_source")
    source = _bound_json(root, config["source_export"], "source_export")
    validate_scalable_formal_candidate_evidence_export(source)
    source_records = sorted(
        iter_scalable_formal_candidate_evidence_records(source),
        key=lambda item: item["candidate_id"],
    )
    if len(source_records) != 163:
        raise ValueError("generated formal-export source population changed")
    script, rendered = render_cadabra_batch(source_records)
    receipt = sandbox_executor(script, root, config)
    _validate_sandbox_receipt(receipt, script)
    source_bindings = source.get("source_bindings", {})
    rendered_by_id = {item["candidate_id"]: item for item in rendered}
    records = []
    for source_record in source_records:
        family = source_record["family_id"]
        source_key, adapter = METRIC_ADAPTERS[family]
        evidence = source_bindings.get(source_key)
        if not isinstance(evidence, dict) or not evidence.get("content_sha256"):
            raise ValueError(f"reviewed metric evidence binding missing: {source_key}")
        rendered_record = rendered_by_id[source_record["candidate_id"]]
        record_body = {
            "schema_version": RECORD_SCHEMA,
            "candidate_id": source_record["candidate_id"],
            "family_id": family,
            "action_sha256": source_record["action_sha256"],
            "formula_inputs_sha256": source_record["theory_formula_inputs"][
                "formula_inputs_sha256"
            ],
            "theory_formula_inputs": {
                "fields": list(source_record["theory_formula_inputs"]["fields"]),
                "parameters": dict(source_record["theory_formula_inputs"]["parameters"]),
                "ordered_operator_densities": [
                    dict(item)
                    for item in source_record["theory_formula_inputs"]["ordered_operator_densities"]
                ],
                "action_content_sha256": source_record["theory_formula_inputs"][
                    "action_content_sha256"
                ],
                "formula_inputs_sha256": source_record["theory_formula_inputs"][
                    "formula_inputs_sha256"
                ],
            },
            "cadabra_symbol": rendered_record["cadabra_symbol"],
            "cadabra_action_expression": rendered_record["cadabra_action_expression"],
            "cadabra_action_expression_sha256": rendered_record["cadabra_action_expression_sha256"],
            "action_export_status": "exact_rendered_and_sandbox_parsed",
            "metric_variation_route": {
                "status": "reviewed_adapter_bound_not_executed_by_this_campaign",
                "adapter": adapter,
                "evidence_source_key": source_key,
                "evidence_content_sha256": evidence["content_sha256"],
                "metric_variation_executed_by_this_campaign": False,
            },
            "formal_context": {
                "decision": source_record["final_decision"],
                "result_sha256": source_record["result_sha256"],
                "changed_by_action_export": False,
            },
            "formal_pass_inferred": False,
        }
        records.append({**record_body, "content_sha256": _sha(record_body)})
    export_body = {
        "schema_version": EXPORT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "source_export": dict(config["source_export"]),
            "campaign_source": dict(config["campaign_source"]),
        },
        "candidate_count": len(records),
        "family_counts": dict(sorted(Counter(record["family_id"] for record in records).items())),
        "action_export_counts": {
            "exact_rendered": len(records),
            "sandbox_parsed_and_canonicalised": len(records),
            "rejected": 0,
        },
        "metric_variation_counts": {
            "reviewed_adapter_routes_bound": len(records),
            "executed_by_this_campaign": 0,
            "formal_passes_inferred": 0,
        },
        "cadabra_batch_script": script,
        "sandbox_receipt": receipt,
        "candidate_record_registry_root_sha256": _sha(
            [record["content_sha256"] for record in records]
        ),
        "candidate_records": records,
        "first_missing_premise": (
            "candidate_specific_metric_variation_execution_from_the_generated_action_export_"
            "for_each_action_hash_and_future_operator_family"
        ),
        "scope": (
            "exact action-level export and parser/canonicaliser execution only; reviewed metric "
            "adapter routes are provenance, not executions or new formal passes"
        ),
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": {**ELIGIBILITY, "passed": True},
    }
    export = {**export_body, "content_sha256": _sha(export_body)}
    validate_generated_candidate_formal_export(export)
    return export
