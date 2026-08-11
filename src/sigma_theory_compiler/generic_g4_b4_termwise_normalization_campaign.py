"""Exact termwise normalization of the generic quartic-Horndeski metric Euler tensor.

This campaign compares the independently executed Cadabra inverse-metric variation of
``G4(phi,X) R + G4_X[(box phi)^2-H_ab H^ab]`` with the coefficient tensor printed as
equation B.4 of Kobayashi--Yamaguchi--Yokoyama (2011).  The comparison is deliberately
at the coefficient-of-``delta g^{ab}`` level: 24 independent canonical contractions are
matched over the rationals, and no global or stability conclusion is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sigma-generic-g4-b4-termwise-normalization-1.0"

# Independent transcription of KYY 2011 equation B.4 after contraction with a
# symmetric inverse-metric variation h^{ab}, use of X=-p^2/2 and Q_a=nabla_a X,
# and expansion of each symmetrized pair.  The keys name linearly independent
# tensor contractions in the Cadabra normal form.
B4_COEFFICIENTS: dict[str, Fraction] = {
    "F_Ricci_h": Fraction(1),
    "F_R_scalar_trace_h": Fraction(-1, 2),
    "B_theta_squared_trace_h": Fraction(1, 2),
    "B_H_squared_trace_h": Fraction(-1, 2),
    "B_R_scalar_pp_h": Fraction(-1, 2),
    "G4_XX_theta_squared_pp_h": Fraction(-1, 2),
    "G4_XX_H_squared_pp_h": Fraction(1, 2),
    "B_theta_H_h": Fraction(-1),
    "gradB_theta_hp_h": Fraction(-2),
    "B_Ricci_sym_pp_h": Fraction(2),
    "gradB_theta_p_trace_h": Fraction(1),
    "B_Ricci_pp_trace_h": Fraction(-1),
    "B_HH_h": Fraction(1),
    "gradB_H_hp_h": Fraction(2),
    "gradB_p_H_h": Fraction(-1),
    "B_Riemann_pp_h": Fraction(1),
    "G4_phi_theta_trace_h": Fraction(1),
    "G4_phiphi_p2_trace_h": Fraction(1),
    "G4_phiX_Qp_trace_h": Fraction(2),
    "G4_XX_Q2_trace_h": Fraction(1),
    "G4_phi_H_h": Fraction(-1),
    "G4_phiphi_pp_h": Fraction(-1),
    "G4_phiX_Qp_h": Fraction(-2),
    "G4_XX_QQ_h": Fraction(-1),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _leading_coefficient(fragment: str) -> Fraction:
    value = fragment.strip()
    sign = 1
    if value.startswith("+"):
        value = value[1:].lstrip()
    elif value.startswith("-"):
        sign = -1
        value = value[1:].lstrip()
    match = re.match(r"(?:(\d+)(?:/(\d+))?)?", value)
    if match is None or not match.group(1):
        return Fraction(sign)
    return sign * Fraction(int(match.group(1)), int(match.group(2) or "1"))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError("configured path escapes project root")
    return candidate


def _control(artifact: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches = [row for row in artifact.get("checks", []) if row.get("name") == name]
    if len(matches) != 1:
        raise ValueError("generic G4 Cadabra control is missing or duplicated")
    return matches[0]


def _coefficient_map(value: Mapping[str, Any]) -> dict[str, Fraction]:
    try:
        return {str(key): Fraction(str(coefficient)) for key, coefficient in value.items()}
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("B.4 coefficient transcription is malformed") from exc


def _assert_exact_b4_coefficients(value: Mapping[str, Any]) -> None:
    coefficients = _coefficient_map(value)
    if coefficients != B4_COEFFICIENTS:
        raise ValueError("B.4 coefficient transcription differs from the registered equation")


def _assert_primary_source(primary: Mapping[str, Any]) -> None:
    if (
        primary.get("arxiv_id") != "1105.5723v4"
        or primary.get("equation") != "B.4"
        or primary.get("authors") != "Kobayashi, Yamaguchi, Yokoyama"
        or primary.get("url") != "https://arxiv.org/abs/1105.5723"
    ):
        raise ValueError("primary KYY equation source binding is invalid")


def _mutation_is_rejected(value: Mapping[str, Any]) -> bool:
    try:
        _assert_exact_b4_coefficients(value)
    except ValueError:
        return True
    return False


def build_generic_g4_b4_termwise_normalization_campaign(
    config: Mapping[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source = config["campaign_source"]
    if _file_sha(_inside(root, source["path"])) != source["file_sha256"]:
        raise ValueError("campaign source hash mismatch")

    formal_binding = config["formal_controls_artifact"]
    formal_path = _inside(root, formal_binding["path"])
    if _file_sha(formal_path) != formal_binding["file_sha256"]:
        raise ValueError("formal-controls artifact hash mismatch")
    formal = _load_json(formal_path)
    control = _control(formal, config["cadabra_control_name"])
    evidence = control.get("evidence", {})
    stdout = str(evidence.get("stdout_tail", ""))
    if (
        control.get("status") != "pass"
        or evidence.get("return_code") != 0
        or evidence.get("backend_mode") != "wsl-local"
        or hashlib.sha256(stdout.encode("utf-8")).hexdigest()
        != formal_binding["control_stdout_sha256"]
    ):
        raise ValueError("Cadabra generic G4 execution receipt is invalid")

    script = config["cadabra_script"]
    if _file_sha(_inside(root, script["path"])) != script["file_sha256"]:
        raise ValueError("Cadabra generic G4 script hash mismatch")
    recorded_markers = set(evidence.get("expected_fragments", []))
    for marker in config["required_cadabra_markers"]:
        if marker not in recorded_markers:
            raise ValueError(f"Cadabra marker missing: {marker}")

    primary = config["primary_source"]
    _assert_primary_source(primary)
    transcription_binding = config["primary_source_transcription"]
    transcription_path = _inside(root, transcription_binding["path"])
    if _file_sha(transcription_path) != transcription_binding["file_sha256"]:
        raise ValueError("primary KYY equation transcription hash mismatch")
    transcription = _load_json(transcription_path)
    if (
        transcription.get("arxiv_id") != primary["arxiv_id"]
        or transcription.get("equation") != primary["equation"]
        or not isinstance(transcription.get("canonical_coefficients"), dict)
    ):
        raise ValueError("primary KYY equation transcription metadata is invalid")
    _assert_exact_b4_coefficients(transcription["canonical_coefficients"])

    rendered = _normalise_space(stdout)
    fragments = config["cadabra_canonical_fragments"]
    if set(fragments) != set(B4_COEFFICIENTS):
        raise ValueError("Cadabra/B.4 canonical term sets differ")

    records: list[dict[str, Any]] = []
    for term_id in sorted(B4_COEFFICIENTS):
        fragment = _normalise_space(str(fragments[term_id]))
        if rendered.count(fragment) != 1:
            raise ValueError(f"Cadabra canonical term is missing or duplicated: {term_id}")
        cadabra_coefficient = _leading_coefficient(fragment)
        b4_coefficient = B4_COEFFICIENTS[term_id]
        residual = cadabra_coefficient - b4_coefficient
        body = {
            "term_id": term_id,
            "cadabra_fragment_sha256": hashlib.sha256(fragment.encode("utf-8")).hexdigest(),
            "cadabra_coefficient": str(cadabra_coefficient),
            "B4_coefficient": str(b4_coefficient),
            "residual": str(residual),
        }
        records.append({**body, "content_sha256": _sha(body)})

    if len(records) != 24 or any(record["residual"] != "0" for record in records):
        raise ValueError("generic G4 termwise normalization did not close")
    if len({record["cadabra_fragment_sha256"] for record in records}) != 24:
        raise ValueError("generic G4 canonical Cadabra fragments are not unique")

    transcribed_coefficients = dict(transcription["canonical_coefficients"])
    flipped = dict(transcribed_coefficients)
    flipped["B_R_scalar_pp_h"] = str(-Fraction(flipped["B_R_scalar_pp_h"]))
    omitted = dict(transcribed_coefficients)
    omitted.pop("G4_XX_QQ_h")
    wrong_source = dict(primary)
    wrong_source["equation"] = "B.8"
    try:
        _assert_primary_source(wrong_source)
    except ValueError:
        wrong_source_rejected = True
    else:
        wrong_source_rejected = False
    negative_controls = {
        "flip_R_pp_sign_rejected": _mutation_is_rejected(flipped),
        "omit_G4_XX_QQ_rejected": _mutation_is_rejected(omitted),
        "wrong_source_equation_rejected": wrong_source_rejected,
    }
    if not all(negative_controls.values()):
        raise ValueError("generic G4 normalization negative controls failed")

    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "status": "pass_exact_24_term_generic_nonlinear_G4X_metric_Euler_normalization_to_KYY_B4",
        "campaign_source": dict(source),
        "config_content_sha256": _sha(dict(config)),
        "formal_controls_artifact": dict(formal_binding),
        "cadabra_script": dict(script),
        "primary_source": dict(primary),
        "primary_source_transcription": dict(transcription_binding),
        "sign_and_variable_conventions": {
            "metric_signature": "(-,+,+,+)",
            "X": "-p^2/2",
            "Q_a": "nabla_a X=-H_ab p^b",
            "variation": "h^ab=delta g^ab with phi and p_a fixed",
            "paper_tensor": "G^4_ab is the coefficient of delta g^ab",
        },
        "canonical_term_count": 24,
        "matched_term_count": 24,
        "nonzero_residual_count": 0,
        "term_records": records,
        "term_registry_root_sha256": _sha(records),
        "negative_controls": negative_controls,
        "metric_variation_normalization_pass": True,
        "scalar_equation_or_noether_rederived_here": False,
        "full_candidate_formal_pass_inferred": False,
        "global_energy_inferred": False,
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "interpretation": (
            "The independently executed Cadabra coefficient of delta g^ab matches all 24 "
            "canonical contractions obtained from KYY equation B.4. This closes tensor spelling "
            "and coefficient normalization only; it does not prove global energy, nonlinear "
            "stability, observational validity, or future unregistered operator families."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def validate_generic_g4_b4_termwise_normalization_campaign(
    artifact: Mapping[str, Any], project_root: str | Path | None = None
) -> None:
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    records = artifact.get("term_records", [])
    if (
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("content_sha256") != _sha(body)
        or artifact.get("canonical_term_count") != 24
        or artifact.get("matched_term_count") != 24
        or artifact.get("nonzero_residual_count") != 0
        or artifact.get("term_registry_root_sha256") != _sha(records)
        or artifact.get("status")
        != "pass_exact_24_term_generic_nonlinear_G4X_metric_Euler_normalization_to_KYY_B4"
        or artifact.get("metric_variation_normalization_pass") is not True
        or artifact.get("scalar_equation_or_noether_rederived_here") is not False
        or artifact.get("full_candidate_formal_pass_inferred") is not False
        or artifact.get("global_energy_inferred") is not False
        or artifact.get("observational_data_opened") is not False
        or artifact.get("dark_matter_or_halo_inputs") is not False
        or artifact.get("redshift_distance_inputs") is not False
        or artifact.get("paid_llm_spend_usd") != 0.0
        or artifact.get("negative_controls")
        != {
            "flip_R_pp_sign_rejected": True,
            "omit_G4_XX_QQ_rejected": True,
            "wrong_source_equation_rejected": True,
        }
        or len(records) != 24
    ):
        raise ValueError("generic G4 B.4 normalization artifact is invalid")
    try:
        _assert_primary_source(artifact.get("primary_source", {}))
    except ValueError as exc:
        raise ValueError("generic G4 B.4 normalization artifact is invalid") from exc
    transcription_binding = artifact.get("primary_source_transcription", {})
    if (
        transcription_binding.get("path")
        != "formal/sources/kyy_1105.5723v4_eq_B4_canonical_coefficients.json"
        or not re.fullmatch(r"[0-9a-f]{64}", str(transcription_binding.get("file_sha256", "")))
    ):
        raise ValueError("generic G4 B.4 normalization artifact is invalid")
    if project_root is not None:
        root = Path(project_root).resolve()
        transcription_path = _inside(root, transcription_binding["path"])
        if _file_sha(transcription_path) != transcription_binding["file_sha256"]:
            raise ValueError("generic G4 B.4 source transcription is invalid")
        transcription = _load_json(transcription_path)
        _assert_exact_b4_coefficients(transcription.get("canonical_coefficients", {}))
    seen_term_ids: set[str] = set()
    seen_fragments: set[str] = set()
    for record in records:
        record_body = {key: value for key, value in record.items() if key != "content_sha256"}
        term_id = record.get("term_id")
        if (
            record.get("content_sha256") != _sha(record_body)
            or term_id not in B4_COEFFICIENTS
            or record.get("cadabra_coefficient") != str(B4_COEFFICIENTS[term_id])
            or record.get("B4_coefficient") != str(B4_COEFFICIENTS[term_id])
            or record.get("residual") != "0"
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("cadabra_fragment_sha256", ""))
            )
            or term_id in seen_term_ids
            or record.get("cadabra_fragment_sha256") in seen_fragments
        ):
            raise ValueError("generic G4 B.4 term record is invalid")
        seen_term_ids.add(term_id)
        seen_fragments.add(record["cadabra_fragment_sha256"])
    if seen_term_ids != set(B4_COEFFICIENTS):
        raise ValueError("generic G4 B.4 term registry is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generic_g4_b4_termwise_normalization_campaign.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/engine/generic-g4-b4-termwise-normalization-campaign.json"),
    )
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    config_path = arguments.config if arguments.config.is_absolute() else root / arguments.config
    output_path = arguments.output if arguments.output.is_absolute() else root / arguments.output
    artifact = build_generic_g4_b4_termwise_normalization_campaign(_load_json(config_path), root)
    validate_generic_g4_b4_termwise_normalization_campaign(artifact, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
