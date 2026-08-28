"""Frozen Item 16 QED-like weak-field search on fresh S4TM strong lenses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import math
import subprocess
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/gravity_item16_s4tm_qed_field_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item16_s4tm_qed_field.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem16Error(RuntimeError):
    """Raised when an Item 16 freeze or replay invariant is violated."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _content_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("content_sha256", None)
    result["content_sha256"] = _sha256_bytes(_canonical_bytes(result))
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityItem16Error(f"expected JSON object: {path}")
    return value


def _verify_content_hash(payload: Mapping[str, Any], label: str) -> None:
    expected = payload.get("content_sha256")
    if not isinstance(expected, str):
        raise GravityItem16Error(f"{label} has no content hash")
    actual_payload = dict(payload)
    actual_payload.pop("content_sha256", None)
    actual = _sha256_bytes(_canonical_bytes(actual_payload))
    if actual != expected:
        raise GravityItem16Error(f"{label} content hash changed")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    if config.get("schema_version") != "invariant-gravity-item16-s4tm-qed-field-config-1.0":
        raise GravityItem16Error("unexpected Item 16 config schema")
    if int(config.get("item", -1)) != 16:
        raise GravityItem16Error("Item 16 config changed item number")
    if int(config["candidate_generator"]["candidate_cells"]) != 262144:
        raise GravityItem16Error("candidate boundary changed")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem16Error("post-response candidate generation entered config")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem16Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem16Error("paid API calls are outside Item 16")
    expected_goal = str(config["stable_goal_sha256"])
    if _sha256_file(root / GOAL_PATH) != expected_goal:
        raise GravityItem16Error("stable gravity goal changed")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _git(root: Path, *args: str, text_mode: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=text_mode
    )
    return result.stdout.strip() if text_mode else result.stdout


def _require_ancestor(root: Path, commit: str, label: str) -> None:
    if commit.startswith("TO_BE_BOUND"):
        raise GravityItem16Error(f"{label} has not been bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise GravityItem16Error(f"{label} is not an ancestor of HEAD")


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen_config = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen_config) != _contract_digest(config):
        raise GravityItem16Error("scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes):
        raise GravityItem16Error("could not read frozen module bytes")
    if _sha256_bytes(frozen_module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem16Error("Item 16 module differs from scientific freeze")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    return {
        key: base / str(config["paths"][key])
        for key in (
            "predictor_table1",
            "predictor_table2",
            "predictor_source_manifest",
            "sample_manifest",
            "candidate_manifest",
            "exploration_responses",
            "response_source_manifest",
            "compute_manifest",
        )
    }


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in (
        "predictor_table1",
        "predictor_table2",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        repo_path = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{repo_path}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem16Error(f"{key} differs from sample freeze")


def _download(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant-Item16/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        body = response.read()
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    if not body:
        raise GravityItem16Error(f"empty source response: {url}")
    return body, headers


def _parse_vizier_tsv(data: bytes, expected: Sequence[str]) -> list[dict[str, str]]:
    text = data.decode("utf-8")
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    header_index = next(
        (index for index, line in enumerate(lines) if line.split("\t") == list(expected)), None
    )
    if header_index is None:
        observed = [line for line in lines if "\t" in line][:3]
        raise GravityItem16Error(f"VizieR columns changed; observed {observed!r}")
    rows: list[dict[str, str]] = []
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])), delimiter="\t")
    for row in reader:
        values = {str(key): str(value).strip() for key, value in row.items() if key is not None}
        if not values.get(expected[0], ""):
            continue
        if all(set(value) <= {"-", " "} for value in values.values()):
            continue
        if values.get(expected[0], "").startswith("---"):
            continue
        rows.append(values)
    return rows


def _format_float(value: float) -> str:
    return f"{float(value):.12e}"


def _normal_identity(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["candidate_cells"])
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    sizes = {
        "family": len(generator["families"]),
        "amplitude": len(generator["amplitudes"]),
        "secondary_fraction": len(generator["secondary_amplitude_fractions"]),
        "lambda": len(generator["lambda_kpc"]),
        "scale_ratio": len(generator["secondary_scale_ratios"]),
        "power": len(generator["powers"]),
        "density_scale": len(generator["surface_density_scales_Msun_kpc2"]),
        "density_power": len(generator["density_powers"]),
        "polarization": len(generator["polarizations"]),
    }
    arrays = {
        "family1": random.integers(0, sizes["family"], count, dtype=np.int16),
        "family2": random.integers(0, sizes["family"], count, dtype=np.int16),
        "amplitude": random.integers(0, sizes["amplitude"], count, dtype=np.int16),
        "secondary_fraction": random.integers(
            0, sizes["secondary_fraction"], count, dtype=np.int16
        ),
        "lambda": random.integers(0, sizes["lambda"], count, dtype=np.int16),
        "scale_ratio": random.integers(0, sizes["scale_ratio"], count, dtype=np.int16),
        "power1": random.integers(0, sizes["power"], count, dtype=np.int16),
        "power2": random.integers(0, sizes["power"], count, dtype=np.int16),
        "density_scale": random.integers(0, sizes["density_scale"], count, dtype=np.int16),
        "density_power": random.integers(0, sizes["density_power"], count, dtype=np.int16),
        "polarization1": random.integers(0, sizes["polarization"], count, dtype=np.int16),
        "polarization2": random.integers(0, sizes["polarization"], count, dtype=np.int16),
    }
    return arrays


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(values.dtype).encode("ascii") + b"\0")
        digest.update(values.tobytes())
    return digest.hexdigest()


def _exact_equivalence_classes(arrays: Mapping[str, np.ndarray]) -> int:
    keys = sorted(arrays)
    matrix = np.column_stack([arrays[key].astype(np.int32) for key in keys])
    zero_secondary = arrays["secondary_fraction"] == 0
    secondary_keys = ("family2", "scale_ratio", "power2", "polarization2")
    for key in secondary_keys:
        matrix[zero_secondary, keys.index(key)] = 0
    packed = np.ascontiguousarray(matrix).view(
        np.dtype((np.void, matrix.dtype.itemsize * matrix.shape[1]))
    )
    return int(np.unique(packed).size)


def _hmac_rank(key: str, message: str) -> str:
    return hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def _prior_identity_hits(
    root: Path, source_dir: Path, identities: Sequence[str]
) -> dict[str, list[str]]:
    wanted = {_normal_identity(value): value for value in identities}
    hits: dict[str, list[str]] = {value: [] for value in identities}
    gravity_root = root / "runs" / "gravity"
    for path in gravity_root.rglob("*"):
        if not path.is_file() or source_dir in path.parents:
            continue
        if path.suffix.lower() not in {".json", ".tsv", ".csv", ".txt", ".dat"}:
            continue
        try:
            text = _normal_identity(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        for normalized, original in wanted.items():
            if normalized and normalized in text:
                hits[original].append(path.relative_to(root).as_posix())
    return {key: value for key, value in hits.items() if value}


def _build_sample(
    predictors1: Sequence[Mapping[str, str]],
    predictors2: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    first = {str(row["Target"]): row for row in predictors1}
    second = {str(row["Target"]): row for row in predictors2}
    names = sorted(set(first) & set(second))
    objects: list[dict[str, Any]] = []
    for name in names:
        left, right = first[name], second[name]
        if not str(left["Class"]).endswith("A"):
            continue
        numeric = {
            "z_lens": float(left["zL"]),
            "z_source": float(left["zS"]),
            "imag_ab": float(left["Imag"]),
            "galactic_extinction_mag": float(left["Ai"]),
            "reff_arcsec": float(left["Reff"]),
            "axis_ratio": float(left["q"]),
            "log10_stellar_mass_msun": float(right["logM"]),
        }
        if (
            numeric["z_lens"] <= 0
            or numeric["z_source"] <= numeric["z_lens"]
            or numeric["reff_arcsec"] <= 0
            or numeric["axis_ratio"] <= 0
            or not all(math.isfinite(value) for value in numeric.values())
        ):
            continue
        objects.append({"name": name, "class": str(left["Class"]), **numeric})
    expected = int(config["source"]["expected_grade_a_objects"])
    if len(objects) != expected:
        raise GravityItem16Error(
            f"expected {expected} complete grade-A predictors, found {len(objects)}"
        )
    objects.sort(key=lambda row: (float(row["log10_stellar_mass_msun"]), str(row["name"])))
    role_key = str(config["sample"]["role_key"])
    confirmation: set[str] = set()
    for stratum in range(5):
        group = objects[stratum * 8 : (stratum + 1) * 8]
        for row in group:
            message = (
                f"{row['name']}|{row['z_lens']:.8f}|{row['z_source']:.8f}|"
                f"{row['log10_stellar_mass_msun']:.8f}"
            )
            row["mass_stratum"] = stratum
            row["role_rank"] = _hmac_rank(role_key, message)
        confirmation.update(
            str(row["name"]) for row in sorted(group, key=lambda row: row["role_rank"])[:2]
        )
    exploration = [row for row in objects if str(row["name"]) not in confirmation]
    fold_key = str(config["sample"]["fold_key"])
    exploration.sort(key=lambda row: _hmac_rank(fold_key, str(row["name"])))
    for index, row in enumerate(exploration):
        row["role"] = "exploration"
        row["outer_fold"] = index % int(config["sample"]["outer_folds"])
    for row in objects:
        if str(row["name"]) in confirmation:
            row["role"] = "reserved_confirmation"
            row["outer_fold"] = None
    objects.sort(key=lambda row: str(row["name"]))
    return objects


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["predictor_table1"].parent.mkdir(parents=True, exist_ok=True)
    query1 = str(config["source"]["predictor_queries"]["table1"])
    query2 = str(config["source"]["predictor_queries"]["table2"])
    body1, headers1 = _download(query1)
    body2, headers2 = _download(query2)
    rows1 = _parse_vizier_tsv(body1, ("Target", "zL", "zS", "Imag", "Ai", "Reff", "q", "Class"))
    rows2 = _parse_vizier_tsv(body2, ("Target", "logM"))
    paths["predictor_table1"].write_bytes(body1)
    paths["predictor_table2"].write_bytes(body2)
    objects = _build_sample(rows1, rows2, config)
    source_dir = paths["predictor_table1"].parent
    prior_hits = _prior_identity_hits(root, source_dir, [str(row["name"]) for row in objects])
    if prior_hits:
        raise GravityItem16Error(f"S4TM identities overlap prior response artifacts: {prior_hits}")
    exploration = [row for row in objects if row["role"] == "exploration"]
    confirmation = [row for row in objects if row["role"] == "reserved_confirmation"]
    sample = _content_hashed(
        {
            "schema_version": "invariant-gravity-item16-s4tm-sample-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selection_used_response_values": False,
            "prior_response_identity_hits": {},
            "counts": {
                "eligible": len(objects),
                "exploration": len(exploration),
                "reserved_confirmation": len(confirmation),
                "response_values_read": 0,
            },
            "fold_counts_exploration": dict(
                sorted(Counter(str(row["outer_fold"]) for row in exploration).items())
            ),
            "objects": [
                {
                    key: (_format_float(value) if isinstance(value, float) else value)
                    for key, value in row.items()
                }
                for row in objects
            ],
            "claims": {"confirmation_opened": False},
        }
    )
    arrays = generate_candidates(config)
    candidates = _content_hashed(
        {
            "schema_version": "invariant-gravity-item16-s4tm-candidates-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "algorithm": config["candidate_generator"]["algorithm"],
            "seed": config["candidate_generator"]["seed"],
            "candidate_digest_sha256": _candidate_digest(arrays),
            "counts": {
                "raw_candidate_cells": len(arrays["family1"]),
                "exact_parameter_equivalence_classes": _exact_equivalence_classes(arrays),
                "post_response_cells": 0,
            },
            "families": config["candidate_generator"]["families"],
            "equivalence_boundaries": config["candidate_generator"]["equivalence_boundaries"],
            "response_values_read": 0,
        }
    )
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item16-s4tm-predictors-1.0",
            "catalog": config["source"]["catalog"],
            "paper_bibcode": config["source"]["paper_bibcode"],
            "queries": {
                "table1": {"url": query1, "selected_columns": list(rows1[0]), "rows": len(rows1)},
                "table2": {"url": query2, "selected_columns": list(rows2[0]), "rows": len(rows2)},
            },
            "files": [
                {
                    "path": paths["predictor_table1"].relative_to(root).as_posix(),
                    "sha256": _sha256_file(paths["predictor_table1"]),
                    "last_modified": headers1.get("last-modified"),
                },
                {
                    "path": paths["predictor_table2"].relative_to(root).as_posix(),
                    "sha256": _sha256_file(paths["predictor_table2"]),
                    "last_modified": headers2.get("last-modified"),
                },
            ],
            "response_columns_requested": [],
            "forbidden_columns_read": [],
        }
    )
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample)
    _write_json(paths["candidate_manifest"], candidates)
    return paths


def _load_prepared(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = _source_paths(root, config)
    predictor_manifest = _read_json(paths["predictor_source_manifest"])
    sample = _read_json(paths["sample_manifest"])
    candidates = _read_json(paths["candidate_manifest"])
    for payload, label in (
        (predictor_manifest, "predictor manifest"),
        (sample, "sample manifest"),
        (candidates, "candidate manifest"),
    ):
        _verify_content_hash(payload, label)
    arrays = generate_candidates(config)
    if candidates["candidate_digest_sha256"] != _candidate_digest(arrays):
        raise GravityItem16Error("prepared candidate digest changed")
    if int(candidates["counts"]["raw_candidate_cells"]) != 262144:
        raise GravityItem16Error("prepared candidate count changed")
    if int(candidates["counts"]["post_response_cells"]) != 0:
        raise GravityItem16Error("prepared manifest admits post-response candidates")
    for item in predictor_manifest["files"]:
        path = root / str(item["path"])
        if _sha256_file(path) != str(item["sha256"]):
            raise GravityItem16Error(f"predictor source changed: {path}")
    if int(sample["counts"]["response_values_read"]) != 0 or bool(
        sample["claims"]["confirmation_opened"]
    ):
        raise GravityItem16Error("sample freeze was contaminated by response values")
    return predictor_manifest, sample, candidates


def _response_url(config: Mapping[str, Any], table: str, target: str) -> str:
    columns = ",".join(str(value) for value in config["source"]["response_columns"][table])
    query = urllib.parse.urlencode(
        [
            ("-source", f"J/ApJ/851/48/{table}"),
            ("-out", columns),
            ("-out.max", "unlimited"),
            ("Target", target),
        ]
    )
    return f"{config['source']['response_query_base']}?{query}"


def fetch_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    _, sample, candidates = _load_prepared(root, config)
    paths = _source_paths(root, config)
    exploration = [row for row in sample["objects"] if str(row["role"]) == "exploration"]
    confirmations = {
        str(row["name"]) for row in sample["objects"] if str(row["role"]) == "reserved_confirmation"
    }
    output_rows: list[dict[str, str]] = []
    source_receipts: list[dict[str, Any]] = []
    for object_row in exploration:
        target = str(object_row["name"])
        url1 = _response_url(config, "table1", target)
        url2 = _response_url(config, "table2", target)
        body1, headers1 = _download(url1)
        body2, headers2 = _download(url2)
        rows1 = _parse_vizier_tsv(body1, ("Target", "Sigma", "e_Sigma"))
        rows2 = _parse_vizier_tsv(body2, ("Target", "bSIE"))
        if len(rows1) != 1 or len(rows2) != 1:
            raise GravityItem16Error(f"response query did not return one row for {target}")
        if rows1[0]["Target"] != target or rows2[0]["Target"] != target:
            raise GravityItem16Error(f"response identity mismatch for {target}")
        if target in confirmations:
            raise GravityItem16Error("confirmation response was requested")
        output_rows.append(
            {
                "Target": target,
                "Sigma": rows1[0]["Sigma"],
                "e_Sigma": rows1[0]["e_Sigma"],
                "bSIE": rows2[0]["bSIE"],
            }
        )
        source_receipts.append(
            {
                "target": target,
                "table1_query_sha256": _sha256_bytes(url1.encode("utf-8")),
                "table1_response_sha256": _sha256_bytes(body1),
                "table1_last_modified": headers1.get("last-modified"),
                "table2_query_sha256": _sha256_bytes(url2.encode("utf-8")),
                "table2_response_sha256": _sha256_bytes(body2),
                "table2_last_modified": headers2.get("last-modified"),
            }
        )
    output_rows.sort(key=lambda row: row["Target"])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("Target", "Sigma", "e_Sigma", "bSIE"),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(output_rows)
    paths["exploration_responses"].write_text(buffer.getvalue(), encoding="utf-8", newline="")
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item16-s4tm-response-source-1.0",
            "sample_freeze_commit": config["sample_freeze_commit"],
            "catalog": config["source"]["catalog"],
            "requested_objects": len(exploration),
            "returned_objects": len(output_rows),
            "confirmation_objects_requested": 0,
            "confirmation_response_values_read": 0,
            "post_response_candidate_cells": int(candidates["counts"]["post_response_cells"]),
            "columns": ["Target", "Sigma", "e_Sigma", "bSIE"],
            "excluded_columns": config["source"]["excluded_response_columns"],
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
            },
            "per_object_source_receipts": source_receipts,
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["exploration_responses"]


def _comoving_distance_kpc(z: float, config: Mapping[str, Any]) -> float:
    cosmology = config["physics"]["cosmology"]
    nodes, weights = np.polynomial.legendre.leggauss(96)
    points = 0.5 * z * (nodes + 1.0)
    expansion = np.sqrt(
        float(cosmology["omega_m"]) * (1.0 + points) ** 3 + float(cosmology["omega_lambda"])
    )
    integral = 0.5 * z * float(np.sum(weights / expansion))
    c = float(config["physics"]["constants"]["c_km_s"])
    h0 = float(cosmology["H0_km_s_Mpc"])
    return (c / h0) * integral * 1000.0


def _angular_diameter_distances(
    z_lens: float, z_source: float, config: Mapping[str, Any]
) -> tuple[float, float, float]:
    chi_lens = _comoving_distance_kpc(z_lens, config)
    chi_source = _comoving_distance_kpc(z_source, config)
    return (
        chi_lens / (1.0 + z_lens),
        chi_source / (1.0 + z_source),
        (chi_source - chi_lens) / (1.0 + z_source),
    )


def hernquist_projected_mass_fraction(radius_over_a: np.ndarray | float) -> np.ndarray:
    x = np.asarray(radius_over_a, dtype=np.float64)
    if np.any(x <= 0):
        raise GravityItem16Error("Hernquist projected radius must be positive")
    result = np.empty_like(x)
    below = x < 1.0 - 1e-7
    above = x > 1.0 + 1e-7
    middle = ~(below | above)
    if np.any(below):
        values = x[below]
        kernel = np.arccosh(1.0 / values) / np.sqrt(1.0 - values**2)
        result[below] = values**2 * (kernel - 1.0) / (1.0 - values**2)
    if np.any(above):
        values = x[above]
        kernel = np.arccos(1.0 / values) / np.sqrt(values**2 - 1.0)
        result[above] = values**2 * (kernel - 1.0) / (1.0 - values**2)
    result[middle] = 1.0 / 3.0
    return np.clip(result, 0.0, 1.0)


def _load_response_rows(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(root, config)
    _, sample, _ = _load_prepared(root, config)
    response_manifest = _read_json(paths["response_source_manifest"])
    _verify_content_hash(response_manifest, "response source manifest")
    if int(response_manifest["confirmation_response_values_read"]) != 0:
        raise GravityItem16Error("confirmation response was opened")
    if _sha256_file(paths["exploration_responses"]) != str(
        response_manifest["response_file"]["sha256"]
    ):
        raise GravityItem16Error("exploration response file changed")
    with paths["exploration_responses"].open(encoding="utf-8", newline="") as handle:
        response = {row["Target"]: row for row in csv.DictReader(handle, delimiter="\t")}
    rows: list[dict[str, Any]] = []
    for predictor in sample["objects"]:
        if predictor["role"] != "exploration":
            continue
        name = str(predictor["name"])
        observed = response.get(name)
        if observed is None:
            continue
        try:
            sigma = float(observed["Sigma"])
            e_sigma = float(observed["e_Sigma"])
            theta_ein = float(observed["bSIE"])
            z_lens = float(predictor["z_lens"])
            z_source = float(predictor["z_source"])
            reff_arcsec = float(predictor["reff_arcsec"])
            log_mass = float(predictor["log10_stellar_mass_msun"])
            axis_ratio = float(predictor["axis_ratio"])
        except (TypeError, ValueError):
            continue
        quality = (
            sigma > 0
            and e_sigma > 0
            and e_sigma < sigma
            and theta_ein > 0
            and reff_arcsec > 0
            and z_source > z_lens > 0
            and math.isfinite(log_mass)
        )
        if not quality:
            continue
        d_lens, d_source, d_lens_source = _angular_diameter_distances(z_lens, z_source, config)
        arcsec = float(config["physics"]["constants"]["arcsec_to_radian"])
        reff_kpc = reff_arcsec * arcsec * d_lens
        rein_kpc = theta_ein * arcsec * d_lens
        stellar_mass = 10.0**log_mass
        re_over_a = float(config["physics"]["hernquist_re_over_a"])
        projected_fraction = float(
            hernquist_projected_mass_fraction(rein_kpc / (reff_kpc / re_over_a))
        )
        c = float(config["physics"]["constants"]["c_km_s"])
        gravitational = float(config["physics"]["constants"]["G_kpc_km2_s2_Msun"])
        sigma_critical = (c**2 / (4.0 * math.pi * gravitational)) * (
            d_source / (d_lens * d_lens_source)
        )
        lens_required_mass = math.pi * rein_kpc**2 * sigma_critical
        virial = float(config["physics"]["dynamical_virial_coefficient"])
        y_dyn = math.log(virial * reff_kpc * sigma**2 / (gravitational * stellar_mass))
        y_lens = math.log(lens_required_mass / (stellar_mass * projected_fraction))
        surface_density = stellar_mass / (2.0 * math.pi * reff_kpc**2)
        rows.append(
            {
                "name": name,
                "fold": int(predictor["outer_fold"]),
                "z_lens": z_lens,
                "z_source": z_source,
                "log_mass": log_mass,
                "axis_ratio": axis_ratio,
                "reff_kpc": reff_kpc,
                "rein_kpc": rein_kpc,
                "surface_density": surface_density,
                "sigma": sigma,
                "e_sigma": e_sigma,
                "theta_ein_arcsec": theta_ein,
                "projected_stellar_fraction_at_rein": projected_fraction,
                "y_dyn": y_dyn,
                "y_lens": y_lens,
            }
        )
    rows.sort(key=lambda row: row["name"])
    return rows, response_manifest


def _candidate_values(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], begin: int, end: int, xp: Any
) -> dict[str, Any]:
    generator = config["candidate_generator"]
    return {
        "family1": xp.asarray(arrays["family1"][begin:end]),
        "family2": xp.asarray(arrays["family2"][begin:end]),
        "amplitude": xp.asarray(
            np.asarray(generator["amplitudes"])[arrays["amplitude"][begin:end]]
        ),
        "secondary_fraction": xp.asarray(
            np.asarray(generator["secondary_amplitude_fractions"])[
                arrays["secondary_fraction"][begin:end]
            ]
        ),
        "lambda1": xp.asarray(np.asarray(generator["lambda_kpc"])[arrays["lambda"][begin:end]]),
        "scale_ratio": xp.asarray(
            np.asarray(generator["secondary_scale_ratios"])[arrays["scale_ratio"][begin:end]]
        ),
        "power1": xp.asarray(np.asarray(generator["powers"])[arrays["power1"][begin:end]]),
        "power2": xp.asarray(np.asarray(generator["powers"])[arrays["power2"][begin:end]]),
        "density_scale": xp.asarray(
            np.asarray(generator["surface_density_scales_Msun_kpc2"])[
                arrays["density_scale"][begin:end]
            ]
        ),
        "density_power": xp.asarray(
            np.asarray(generator["density_powers"])[arrays["density_power"][begin:end]]
        ),
        "polarization1": xp.asarray(
            np.asarray([row["lensing_to_matter"] for row in generator["polarizations"]])[
                arrays["polarization1"][begin:end]
            ]
        ),
        "polarization2": xp.asarray(
            np.asarray([row["lensing_to_matter"] for row in generator["polarizations"]])[
                arrays["polarization2"][begin:end]
            ]
        ),
    }


def _basis(family: Any, u: Any, power: Any, density: Any, xp: Any) -> Any:
    u = xp.maximum(u, 1e-300)
    powered = xp.power(u, power)
    rational = powered / (1.0 + powered)
    yukawa = 1.0 - (1.0 + u) * xp.exp(-u)
    loop = xp.log1p(powered)
    loop = loop / (1.0 + loop)
    result = xp.where(
        family == 0,
        yukawa,
        xp.where(
            family == 1,
            loop,
            xp.where(
                family == 2,
                rational,
                xp.where(
                    family == 3,
                    rational * density,
                    xp.where(
                        family == 4,
                        xp.sqrt(xp.maximum(yukawa * rational, 0.0)),
                        xp.sqrt(xp.maximum(yukawa * loop, 0.0)) * xp.sqrt(density),
                    ),
                ),
            ),
        ),
    )
    return xp.clip(result, 0.0, 1.0)


def _candidate_log_mu(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    values = _candidate_values(config, arrays, begin, end, xp)
    radii = xp.asarray(
        np.asarray([[row["reff_kpc"], row["rein_kpc"]] for row in rows], dtype=np.float64)
    )[None, :, :]
    surface = xp.asarray(np.asarray([row["surface_density"] for row in rows], dtype=np.float64))[
        None, :, None
    ]
    lambda1 = values["lambda1"][:, None, None]
    lambda2 = (values["lambda1"] * values["scale_ratio"])[:, None, None]
    density = 1.0 / (
        1.0
        + xp.power(
            surface / values["density_scale"][:, None, None],
            values["density_power"][:, None, None],
        )
    )
    first = _basis(
        values["family1"][:, None, None],
        radii / lambda1,
        values["power1"][:, None, None],
        density,
        xp,
    )
    second = _basis(
        values["family2"][:, None, None],
        radii / lambda2,
        values["power2"][:, None, None],
        density,
        xp,
    )
    amplitude1 = values["amplitude"][:, None, None]
    amplitude2 = (values["amplitude"] * values["secondary_fraction"])[:, None, None]
    matter = 1.0 + amplitude1 * first + amplitude2 * second
    lens_weight1 = xp.stack(
        [xp.ones_like(values["polarization1"]), values["polarization1"]], axis=1
    )[:, None, :]
    lens_weight2 = xp.stack(
        [xp.ones_like(values["polarization2"]), values["polarization2"]], axis=1
    )[:, None, :]
    light = 1.0 + amplitude1 * first * lens_weight1 + amplitude2 * second * lens_weight2
    # Channel 0 is matter at Re and channel 1 is light at REin.  The unused cross terms
    # are formed above only to keep the Phi/Psi mapping explicit and testable.
    combined = xp.stack([matter[:, :, 0], light[:, :, 1]], axis=2)
    return xp.log(combined)


def _backend() -> tuple[Any, str, str]:
    try:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return cp, "gpu_cupy", str(name)
    except (ImportError, RuntimeError, OSError) as error:
        raise GravityItem16Error(
            f"Item 16 requires the configured CUDA screening lane: {error}"
        ) from error


def _to_numpy(value: Any, xp: Any) -> np.ndarray:
    if xp is np:
        return np.asarray(value)
    return xp.asnumpy(value)


def _build_log_mu_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    xp: Any,
) -> Any:
    batch = int(config["evaluation"]["candidate_batch_size"])
    pieces = []
    for begin in range(0, len(arrays["family1"]), batch):
        end = min(begin + batch, len(arrays["family1"]))
        pieces.append(_candidate_log_mu(config, arrays, rows, begin, end, xp))
    return xp.concatenate(pieces, axis=0)


def _fit_offset(values: np.ndarray, bounds: tuple[float, float]) -> tuple[float, float]:
    raw = float(np.mean(values))
    return raw, float(np.clip(raw, math.log(bounds[0]), math.log(bounds[1])))


def _screen_log_mu(
    log_mu: Any,
    y: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    bounds = tuple(float(value) for value in config["physics"]["shared_stellar_mass_scale_bounds"])
    y_device = xp.asarray(y)
    predictions = np.empty_like(y)
    selected: list[int] = []
    offsets: list[float] = []
    raw_offsets: list[float] = []
    training_mse: list[float] = []
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        heldout = np.where(folds == fold)[0]
        residual = y_device[None, train, :] - log_mu[:, train, :]
        raw = xp.mean(residual, axis=(1, 2))
        fitted = xp.clip(raw, math.log(bounds[0]), math.log(bounds[1]))
        mse = xp.mean((residual - fitted[:, None, None]) ** 2, axis=(1, 2))
        index = int(_to_numpy(xp.argmin(mse), xp))
        selected.append(index)
        raw_value = float(_to_numpy(raw[index], xp))
        fitted_value = float(_to_numpy(fitted[index], xp))
        raw_offsets.append(raw_value)
        offsets.append(fitted_value)
        training_mse.append(float(_to_numpy(mse[index], xp)))
        prediction = _to_numpy(log_mu[index, heldout, :], xp) + fitted_value
        predictions[heldout, :] = prediction
    return {
        "prediction": predictions,
        "selected_indices": selected,
        "log_mass_offsets": offsets,
        "raw_log_mass_offsets": raw_offsets,
        "training_mse": training_mse,
    }


def _baseline_predictions(
    y: np.ndarray,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    bounds = tuple(float(value) for value in config["physics"]["shared_stellar_mass_scale_bounds"])
    shared = np.empty_like(y)
    separate = np.empty_like(y)
    flexible = np.empty_like(y)
    feature = np.asarray(
        [
            [
                row["log_mass"],
                math.log(row["reff_kpc"]),
                row["z_lens"],
                row["axis_ratio"],
                math.log(row["surface_density"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    alpha = float(config["evaluation"]["ridge_alpha"])
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        heldout = np.where(folds == fold)[0]
        _, shared_offset = _fit_offset(y[train, :].reshape(-1), bounds)
        shared[heldout, :] = shared_offset
        for channel in range(2):
            _, channel_offset = _fit_offset(y[train, channel], bounds)
            separate[heldout, channel] = channel_offset
        mean = feature[train].mean(axis=0)
        scale = feature[train].std(axis=0)
        scale[scale == 0] = 1.0
        train_design = np.column_stack([np.ones(len(train)), (feature[train] - mean) / scale])
        held_design = np.column_stack([np.ones(len(heldout)), (feature[heldout] - mean) / scale])
        penalty = np.diag([0.0] + [alpha] * feature.shape[1])
        for channel in range(2):
            coefficient = np.linalg.solve(
                train_design.T @ train_design + penalty,
                train_design.T @ y[train, channel],
            )
            flexible[heldout, channel] = held_design @ coefficient
    return {"shared_GR": shared, "separate_calibration": separate, "flexible_nuisance": flexible}


def _mse(y: np.ndarray, prediction: np.ndarray, indices: np.ndarray | None = None) -> float:
    if indices is not None:
        y = y[indices]
        prediction = prediction[indices]
    return float(np.mean((y - prediction) ** 2))


def _improvement(reference: float, candidate: float) -> float:
    return float((reference - candidate) / reference) if reference > 0 else float("-inf")


def _selected_cell(
    index: int, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    generator = config["candidate_generator"]
    family1 = int(arrays["family1"][index])
    family2 = int(arrays["family2"][index])
    polarization1 = int(arrays["polarization1"][index])
    polarization2 = int(arrays["polarization2"][index])
    return {
        "candidate_index": index,
        "family1": generator["families"][family1],
        "family2": generator["families"][family2],
        "amplitude": _format_float(generator["amplitudes"][int(arrays["amplitude"][index])]),
        "secondary_amplitude_fraction": _format_float(
            generator["secondary_amplitude_fractions"][int(arrays["secondary_fraction"][index])]
        ),
        "lambda1_kpc": _format_float(generator["lambda_kpc"][int(arrays["lambda"][index])]),
        "lambda2_over_lambda1": _format_float(
            generator["secondary_scale_ratios"][int(arrays["scale_ratio"][index])]
        ),
        "power1": _format_float(generator["powers"][int(arrays["power1"][index])]),
        "power2": _format_float(generator["powers"][int(arrays["power2"][index])]),
        "surface_density_scale_Msun_kpc2": _format_float(
            generator["surface_density_scales_Msun_kpc2"][int(arrays["density_scale"][index])]
        ),
        "density_power": _format_float(
            generator["density_powers"][int(arrays["density_power"][index])]
        ),
        "polarization1": generator["polarizations"][polarization1],
        "polarization2": generator["polarizations"][polarization2],
    }


def _local_limit_max(config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], xp: Any) -> float:
    radius = float(config["physics"]["constants"]["au_to_kpc"])
    synthetic_rows = [
        {"reff_kpc": radius, "rein_kpc": radius, "surface_density": density}
        for density in (1e7, 1e9, 1e11)
    ]
    maximum = 0.0
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["family1"]), batch):
        end = min(begin + batch, len(arrays["family1"]))
        log_mu = _candidate_log_mu(config, arrays, synthetic_rows, begin, end, xp)
        deviation = xp.max(xp.abs(xp.exp(log_mu) - 1.0))
        maximum = max(maximum, float(_to_numpy(deviation, xp)))
    return maximum


def _synthetic_controls(
    log_mu: Any,
    folds: np.ndarray,
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    count = int(log_mu.shape[0])
    injection_index = 16016 % count
    injection = _to_numpy(log_mu[injection_index], xp)
    pattern = np.asarray(
        [[math.sin(index + 0.3), math.cos(index + 0.7)] for index in range(len(folds))]
    )
    y_injection = math.log(1.25) + injection + 0.002 * pattern
    injected_screen = _screen_log_mu(log_mu, y_injection, folds, config, xp)
    injected_baseline = _baseline_predictions(
        y_injection,
        folds,
        [
            {
                "log_mass": 11.0 + 0.01 * index,
                "reff_kpc": 2.0 + 0.1 * index,
                "z_lens": 0.1 + 0.001 * index,
                "axis_ratio": 0.7,
                "surface_density": 1e9 + 1e7 * index,
            }
            for index in range(len(folds))
        ],
        config,
    )["shared_GR"]
    injection_candidate_mse = _mse(y_injection, injected_screen["prediction"])
    injection_gr_mse = _mse(y_injection, injected_baseline)
    y_gr = np.full((len(folds), 2), math.log(1.25), dtype=np.float64)
    gr_screen = _screen_log_mu(log_mu, y_gr, folds, config, xp)
    gr_candidate_mse = _mse(y_gr, gr_screen["prediction"])
    gr_baseline = _baseline_predictions(
        y_gr,
        folds,
        [
            {
                "log_mass": 11.0 + 0.01 * index,
                "reff_kpc": 2.0 + 0.1 * index,
                "z_lens": 0.1 + 0.001 * index,
                "axis_ratio": 0.7,
                "surface_density": 1e9 + 1e7 * index,
            }
            for index in range(len(folds))
        ],
        config,
    )["shared_GR"]
    gr_baseline_mse = _mse(y_gr, gr_baseline)
    return {
        "injection_candidate_index": injection_index,
        "injection_candidate_mse": injection_candidate_mse,
        "injection_GR_mse": injection_gr_mse,
        "injection_improves_over_GR": injection_candidate_mse < injection_gr_mse,
        "GR_candidate_mse": gr_candidate_mse,
        "GR_baseline_mse": gr_baseline_mse,
        "GR_control_prefers_nonzero_carrier": gr_candidate_mse < gr_baseline_mse - 1e-18,
    }


def _weighted_mse(
    y: np.ndarray,
    prediction: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> float:
    mass_log_error = math.log(10.0) * float(config["evaluation"]["stellar_mass_systematic_dex"])
    lens_fraction = float(config["evaluation"]["lens_radius_fractional_uncertainty"])
    errors = np.asarray(
        [
            [
                math.hypot(2.0 * float(row["e_sigma"]) / float(row["sigma"]), mass_log_error),
                math.hypot(2.0 * lens_fraction, mass_log_error),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    weights = 1.0 / errors**2
    return float(np.sum(weights * (y - prediction) ** 2) / np.sum(weights))


def _evaluation(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    record_compute: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = generate_candidates(config)
    xp, backend, device = _backend()
    if len(rows) < int(config["gates"]["minimum_complete_exploration_objects"]):
        raise GravityItem16Error("too few complete exploration objects for frozen Item 16 test")
    y = np.asarray([[row["y_dyn"], row["y_lens"]] for row in rows], dtype=np.float64)
    folds = np.asarray([row["fold"] for row in rows], dtype=np.int64)
    if set(folds.tolist()) != set(range(int(config["sample"]["outer_folds"]))):
        raise GravityItem16Error("exploration folds are incomplete")
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    log_mu = _build_log_mu_matrix(config, arrays, rows, xp)
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    matrix_seconds = time.perf_counter() - start
    crosscheck_count = int(config["evaluation"]["cpu_crosscheck_candidates"])
    cpu_crosscheck = _candidate_log_mu(config, arrays, rows, 0, crosscheck_count, np)
    gpu_crosscheck = _to_numpy(log_mu[:crosscheck_count], xp)
    crosscheck_max = float(np.max(np.abs(cpu_crosscheck - gpu_crosscheck)))
    local_limit = _local_limit_max(config, arrays, xp)
    controls = _synthetic_controls(log_mu, folds, config, xp)
    start_screen = time.perf_counter()
    selected = _screen_log_mu(log_mu, y, folds, config, xp)
    baselines = _baseline_predictions(y, folds, rows, config)
    observed_candidate_mse = _mse(y, selected["prediction"])
    observed_shared_mse = _mse(y, baselines["shared_GR"])
    observed_separate_mse = _mse(y, baselines["separate_calibration"])
    observed_flexible_mse = _mse(y, baselines["flexible_nuisance"])
    observed_statistic = _improvement(observed_separate_mse, observed_candidate_mse)
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    null_statistics: list[float] = []
    trials = int(config["evaluation"]["permutation_trials"])
    for trial in range(trials):
        permutation = random.permutation(len(rows))
        permuted_y = y[permutation, :]
        permuted = _screen_log_mu(log_mu, permuted_y, folds, config, xp)
        permuted_baseline = _baseline_predictions(permuted_y, folds, rows, config)[
            "separate_calibration"
        ]
        null_statistics.append(
            _improvement(
                _mse(permuted_y, permuted_baseline),
                _mse(permuted_y, permuted["prediction"]),
            )
        )
        if record_compute and (trial + 1) % 10 == 0:
            print(f"Item 16 selection-aware null {trial + 1}/{trials}", flush=True)
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    screen_seconds = time.perf_counter() - start_screen
    permutation_p = (1 + sum(value >= observed_statistic for value in null_statistics)) / (
        trials + 1
    )
    selected_cells = [
        _selected_cell(index, config, arrays) for index in selected["selected_indices"]
    ]
    primary_families = [str(cell["family1"]["id"]) for cell in selected_cells]
    family_counts = Counter(primary_families)
    same_family_folds = max(family_counts.values())
    channel = {}
    for channel_index, label in enumerate(("stellar_dynamics", "Einstein_radius_lensing")):
        candidate_mse = float(
            np.mean((y[:, channel_index] - selected["prediction"][:, channel_index]) ** 2)
        )
        shared_mse = float(
            np.mean((y[:, channel_index] - baselines["shared_GR"][:, channel_index]) ** 2)
        )
        separate_mse = float(
            np.mean(
                (y[:, channel_index] - baselines["separate_calibration"][:, channel_index]) ** 2
            )
        )
        channel[label] = {
            "candidate_mse": candidate_mse,
            "shared_GR_mse": shared_mse,
            "separate_calibration_mse": separate_mse,
            "improvement_vs_shared_GR": _improvement(shared_mse, candidate_mse),
            "improvement_vs_separate_calibration": _improvement(separate_mse, candidate_mse),
        }
    strata: dict[str, Any] = {}
    for value_key, label in (("log_mass", "stellar_mass"), ("reff_kpc", "effective_radius")):
        values = np.asarray([float(row[value_key]) for row in rows])
        median = float(np.median(values))
        for side, indices in (
            ("low", np.where(values <= median)[0]),
            ("high", np.where(values > median)[0]),
        ):
            candidate_mse = _mse(y, selected["prediction"], indices)
            shared_mse = _mse(y, baselines["shared_GR"], indices)
            strata[f"{label}_{side}"] = {
                "objects": len(indices),
                "candidate_mse": candidate_mse,
                "shared_GR_mse": shared_mse,
                "improvement_vs_shared_GR": _improvement(shared_mse, candidate_mse),
            }
    bounds = tuple(float(value) for value in config["physics"]["shared_stellar_mass_scale_bounds"])
    raw_scales = [math.exp(value) for value in selected["raw_log_mass_offsets"]]
    mass_scale_in_bounds = all(bounds[0] <= value <= bounds[1] for value in raw_scales)
    all_positive = bool(np.all(np.exp(_to_numpy(log_mu, xp)) > 0.0))
    gates = {
        "minimum_complete_exploration_objects": len(rows)
        >= int(config["gates"]["minimum_complete_exploration_objects"]),
        "confirmation_values_read_zero": True,
        "post_response_candidate_cells_zero": int(
            config["candidate_generator"]["post_response_cells"]
        )
        == 0,
        "local_classical_limit": local_limit
        <= float(config["gates"]["max_local_fractional_deviation_at_1AU"]),
        "positive_matter_and_light_response": all_positive,
        "synthetic_injection_recovered": bool(controls["injection_improves_over_GR"]),
        "known_GR_control": not bool(controls["GR_control_prefers_nonzero_carrier"]),
        "joint_improvement_vs_shared_GR": _improvement(observed_shared_mse, observed_candidate_mse)
        >= float(config["gates"]["minimum_joint_mse_improvement_vs_shared_GR"]),
        "joint_improvement_vs_separate_calibration": observed_statistic
        >= float(config["gates"]["minimum_joint_mse_improvement_vs_separate_calibration"]),
        "both_channels_improve_vs_shared_GR": all(
            value["improvement_vs_shared_GR"]
            > float(config["gates"]["minimum_each_channel_improvement_vs_shared_GR"])
            for value in channel.values()
        ),
        "all_mass_and_size_halves_improve_vs_shared_GR": all(
            value["improvement_vs_shared_GR"]
            > float(config["gates"]["minimum_each_mass_and_size_half_improvement_vs_shared_GR"])
            for value in strata.values()
        ),
        "selection_aware_permutation": permutation_p
        <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
        "stable_primary_family": same_family_folds
        >= int(config["gates"]["minimum_same_primary_family_folds"]),
        "shared_stellar_mass_scale_in_bounds": mass_scale_in_bounds,
    }
    decision = (
        "PASS_ITEM16_QED_LIKE_WEAK_FIELD_EXPLORATION"
        if all(gates.values())
        else "REJECT_ITEM16_QED_LIKE_WEAK_FIELD_EXPLORATION"
    )
    score_evaluations_per_screen = (
        len(arrays["family1"])
        * 2
        * sum(
            int(np.count_nonzero(folds != fold))
            for fold in range(int(config["sample"]["outer_folds"]))
        )
    )
    compute = {
        "schema_version": "invariant-gravity-item16-compute-1.0",
        "backend": backend,
        "device": device,
        "numpy_version": np.__version__,
        "cupy_version": getattr(xp, "__version__", None),
        "candidate_cells": len(arrays["family1"]),
        "objects": len(rows),
        "channels": 2,
        "candidate_observable_matrix_values": len(arrays["family1"]) * len(rows) * 2,
        "candidate_training_residual_evaluations_observed": score_evaluations_per_screen,
        "candidate_training_residual_evaluations_with_nulls": score_evaluations_per_screen
        * (trials + 1),
        "matrix_build_seconds_observed": matrix_seconds,
        "screen_and_null_seconds_observed": screen_seconds,
        "cpu_gpu_max_absolute_log_mu_difference": crosscheck_max,
    }
    scientific = {
        "decision": decision,
        "counts": {
            "valid_exploration_objects": len(rows),
            "response_channels": 2,
            "candidate_cells": len(arrays["family1"]),
            "post_response_candidate_cells": 0,
            "permutation_trials": trials,
            "passed_gates": sum(bool(value) for value in gates.values()),
            "total_gates": len(gates),
        },
        "quality": {
            "all_responses_finite": bool(np.all(np.isfinite(y))),
            "fold_counts": dict(sorted(Counter(str(value) for value in folds).items())),
        },
        "primary_metrics": {
            "candidate_mse": observed_candidate_mse,
            "shared_GR_mse": observed_shared_mse,
            "separate_calibration_mse": observed_separate_mse,
            "flexible_nuisance_mse": observed_flexible_mse,
            "improvement_vs_shared_GR": _improvement(observed_shared_mse, observed_candidate_mse),
            "improvement_vs_separate_calibration": observed_statistic,
            "improvement_vs_flexible_nuisance": _improvement(
                observed_flexible_mse, observed_candidate_mse
            ),
            "selection_aware_permutation_p": permutation_p,
        },
        "weighted_robustness": {
            "candidate_mse": _weighted_mse(y, selected["prediction"], rows, config),
            "shared_GR_mse": _weighted_mse(y, baselines["shared_GR"], rows, config),
            "separate_calibration_mse": _weighted_mse(
                y, baselines["separate_calibration"], rows, config
            ),
        },
        "channel_metrics": channel,
        "stratum_metrics": strata,
        "outer_selections": [
            {
                "fold": fold,
                "cell": selected_cells[fold],
                "training_mse": selected["training_mse"][fold],
                "stellar_mass_scale": math.exp(selected["log_mass_offsets"][fold]),
                "unclipped_stellar_mass_scale": raw_scales[fold],
                "heldout_objects": [rows[index]["name"] for index in np.where(folds == fold)[0]],
            }
            for fold in range(int(config["sample"]["outer_folds"]))
        ],
        "selection_stability": {
            "primary_family_counts": dict(sorted(family_counts.items())),
            "maximum_same_primary_family_folds": same_family_folds,
            "exact_candidate_indices": selected["selected_indices"],
        },
        "null_distribution": {
            "statistic": "OOF improvement versus separate-channel GR calibration",
            "observed": observed_statistic,
            "minimum": min(null_statistics),
            "median": float(np.median(null_statistics)),
            "maximum": max(null_statistics),
            "sha256": _sha256_bytes(np.asarray(null_statistics, dtype="<f8").tobytes()),
        },
        "classical_and_pipeline_controls": {
            **controls,
            "maximum_fractional_deviation_at_1AU": local_limit,
            "cpu_gpu_max_absolute_log_mu_difference": crosscheck_max,
        },
        "gates": gates,
    }
    if xp is not np:
        del log_mu
        xp.get_default_memory_pool().free_all_blocks()
    return scientific, compute


def _build_receipt(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    response_manifest: Mapping[str, Any],
    scientific: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _source_paths(root, config)
    predictor_manifest, sample, candidates = _load_prepared(root, config)
    receipt = _content_hashed(
        {
            "schema_version": "invariant-gravity-item16-s4tm-qed-field-receipt-1.0",
            "item": 16,
            "title": config["title"],
            "hypothesis": config["hypothesis"],
            "provenance_and_creativity_labels": config["candidate_generator"]["families"],
            "mathematical_definition": {
                "matter_potential": config["physics"]["matter_potential"],
                "lensing_potential": config["physics"]["lensing_potential"],
                "baryon_profile": config["physics"]["baryon_profile"],
                "dynamical_response": "log[5 Re sigma^2/(G Mstar)]",
                "lensing_response": "log[pi REin^2 SigmaCrit/(Mstar fHernquist(REin))]",
            },
            "dimensional_and_symmetry_checks": {
                "responses_dimensionless": True,
                "carrier_arguments_dimensionless": True,
                "static_spherical_parity_even": True,
                "short_distance_GR_limit_enforced": True,
                "positive_residue_response_enforced": True,
                "covariant_action_or_Ward_identity_proven": False,
                "full_ghost_freedom_proven": False,
            },
            "claim_boundary": config["scope"]["claim_ceiling"],
            "source_bindings": {
                "predictor_manifest_sha256": predictor_manifest["content_sha256"],
                "sample_manifest_sha256": sample["content_sha256"],
                "candidate_manifest_sha256": candidates["content_sha256"],
                "response_manifest_sha256": response_manifest["content_sha256"],
                "response_file_sha256": response_manifest["response_file"]["sha256"],
                "observable_lineage": config["source"]["observable_lineage"],
            },
            "frozen_boundary": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "stable_goal_sha256": config["stable_goal_sha256"],
                "confirmation_opened": False,
                "confirmation_response_values_read": 0,
                "post_response_formula_generation": False,
            },
            "baselines": {
                "shared_GR": config["evaluation"]["baseline_shared_imf"],
                "separate_calibration": config["evaluation"]["baseline_separate_calibration"],
                "flexible_nuisance": config["evaluation"]["baseline_flexible_nuisance"],
            },
            "scientific_result": scientific,
            "compute_and_api_cost": {
                **compute,
                "paid_model_calls": 0,
                "paid_api_spend_usd": 0.0,
            },
            "counterexamples_and_limitations": [
                "A universal stellar-mass scale is fitted on training folds, but object-specific IMF variation is not measured.",
                "The spherical Hernquist and virial approximations are not a resolved Jeans analysis and can create model error.",
                "The published SIE Einstein radius is image-model derived; this is not the roadmap's direct image-likelihood lensing gate.",
                "Gas is treated as negligible for the selected early-type lenses.",
                "Positive weak-field response and a Solar-System limit do not prove a covariant, conserved, causal, or ghost-free completion.",
                "No sealed confirmation response is opened, so even a passing exploration result would remain a candidate lead.",
            ],
            "result": scientific["decision"].split("_ITEM16")[0],
            "exact_next_action": "Advance to Item 17 with Item 16 carrier families retained only as labeled weak-field counterexamples or an unconfirmed lead; do not open the ten S4TM confirmations without a new authorization and unchanged frozen survivor.",
            "reproducibility": {
                "config_path": CONFIG_PATH.as_posix(),
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "module_path": MODULE_PATH.as_posix(),
                "module_sha256": _sha256_file(root / MODULE_PATH),
                "compute_manifest_path": paths["compute_manifest"].relative_to(root).as_posix(),
                "valid_object_names": [row["name"] for row in rows],
            },
        }
    )
    return receipt


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest = _load_response_rows(root, config)
    scientific, compute_raw = _evaluation(root, config, rows, record_compute=True)
    paths = _source_paths(root, config)
    compute = _content_hashed(compute_raw)
    _write_json(paths["compute_manifest"], compute)
    receipt = _build_receipt(root, config, rows, response_manifest, scientific, compute)
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, receipt)
    return result_path


def replay(root: Path, receipt_path: Path | None = None) -> dict[str, Any]:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest = _load_response_rows(root, config)
    paths = _source_paths(root, config)
    compute = _read_json(paths["compute_manifest"])
    _verify_content_hash(compute, "compute manifest")
    scientific, _ = _evaluation(root, config, rows, record_compute=False)
    rebuilt = _build_receipt(root, config, rows, response_manifest, scientific, compute)
    expected_path = receipt_path or (root / str(config["paths"]["result"]))
    expected = _read_json(expected_path)
    _verify_content_hash(expected, "Item 16 receipt")
    if _canonical_bytes(rebuilt) != _canonical_bytes(expected):
        raise GravityItem16Error("Item 16 replay differs from committed receipt")
    return {
        "status": "PASS",
        "receipt": expected_path.relative_to(root).as_posix()
        if expected_path.is_relative_to(root)
        else str(expected_path),
        "content_sha256": rebuilt["content_sha256"],
        "decision": rebuilt["scientific_result"]["decision"],
    }


def verify_pre_response(root: Path) -> dict[str, Any]:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    predictor, sample, candidates = _load_prepared(root, config)
    return {
        "status": "PASS",
        "scientific_freeze_commit": config["scientific_freeze_commit"],
        "sample_freeze_commit": config["sample_freeze_commit"],
        "predictor_manifest_sha256": predictor["content_sha256"],
        "sample_manifest_sha256": sample["content_sha256"],
        "candidate_manifest_sha256": candidates["content_sha256"],
        "candidate_cells": candidates["counts"]["raw_candidate_cells"],
        "exploration_objects": sample["counts"]["exploration"],
        "reserved_confirmation_objects": sample["counts"]["reserved_confirmation"],
        "response_values_read": sample["counts"]["response_values_read"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-predictors")
    subparsers.add_parser("verify-pre-response")
    subparsers.add_parser("fetch-responses")
    subparsers.add_parser("run")
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare-predictors":
        paths = prepare_predictors(root)
        print(
            json.dumps(
                {
                    "sample_manifest": paths["sample_manifest"].relative_to(root).as_posix(),
                    "candidate_manifest": paths["candidate_manifest"].relative_to(root).as_posix(),
                    "response_values_read": 0,
                },
                sort_keys=True,
            )
        )
    elif args.command == "verify-pre-response":
        print(json.dumps(verify_pre_response(root), sort_keys=True))
    elif args.command == "fetch-responses":
        print(fetch_responses(root).relative_to(root).as_posix())
    elif args.command == "run":
        print(run_experiment(root).relative_to(root).as_posix())
    elif args.command == "replay":
        receipt = args.receipt
        if receipt is not None and not receipt.is_absolute():
            receipt = root / receipt
        print(json.dumps(replay(root, receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
