"""Frozen Item 18 action-motivated gravitational antiscreening experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import time
import urllib.parse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import iv, kv

from sigma_theory_compiler.gravity_item16_s4tm_qed_field import (
    _backend,
    _canonical_bytes,
    _content_hashed,
    _download,
    _format_float,
    _hmac_rank,
    _improvement,
    _mse,
    _parse_vizier_tsv,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _to_numpy,
    _verify_content_hash,
    _write_json,
)

CONFIG_PATH = Path("configs/gravity_item18_diskmass_antiscreening_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item18_diskmass_antiscreening.py")
DEPENDENCY_PATH = Path("src/sigma_theory_compiler/gravity_item16_s4tm_qed_field.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem18Error(RuntimeError):
    """Raised when an Item 18 freeze or replay invariant is violated."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    if config.get("schema_version") != "invariant-gravity-item18-diskmass-antiscreening-config-1.0":
        raise GravityItem18Error("unexpected Item 18 config schema")
    if int(config.get("item", -1)) != 18:
        raise GravityItem18Error("Item 18 config changed item number")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem18Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem18Error("paid API calls are outside Item 18")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem18Error("post-response cells entered Item 18")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem18Error("stable gravity goal changed")
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
        raise GravityItem18Error(f"{label} has not been bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise GravityItem18Error(f"{label} is not an ancestor of HEAD")


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen_config = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen_config) != _contract_digest(config):
        raise GravityItem18Error("scientific contract differs from frozen commit")
    for path in (MODULE_PATH, DEPENDENCY_PATH):
        frozen = _git(root, "show", f"{commit}:{path.as_posix()}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(root / path):
            raise GravityItem18Error(f"Item 18 dependency differs from freeze: {path}")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "predictor_diskmass",
        "predictor_alfalfa",
        "predictor_kinematic",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "exploration_responses",
        "response_source_manifest",
        "compute_manifest",
    )
    return {key: base / str(config["paths"][key]) for key in keys}


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in (
        "predictor_diskmass",
        "predictor_alfalfa",
        "predictor_kinematic",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        repository_path = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{repository_path}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem18Error(f"{key} differs from sample freeze")


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    amplitudes = np.linspace(
        float(generator["amplitude_min"]),
        float(generator["amplitude_max"]),
        int(generator["amplitude_count"]),
    )
    a0 = np.logspace(
        float(generator["a0_log10_m_s2_min"]),
        float(generator["a0_log10_m_s2_max"]),
        int(generator["a0_count"]),
    )
    powers = np.asarray(generator["powers"], dtype=np.float64)
    stellar = np.linspace(
        float(generator["stellar_mass_to_light_min"]),
        float(generator["stellar_mass_to_light_max"]),
        int(generator["stellar_mass_to_light_count"]),
    )
    gas = np.asarray(generator["gas_mass_scales"], dtype=np.float64)
    aa, a0a, pp, ss, gg = np.meshgrid(amplitudes, a0, powers, stellar, gas, indexing="ij")
    raw = {
        "amplitude": aa.reshape(-1),
        "a0_m_s2": a0a.reshape(-1),
        "power": pp.reshape(-1),
        "stellar_mass_to_light": ss.reshape(-1),
        "gas_mass_scale": gg.reshape(-1),
    }
    constants = config["physics"]["constants"]
    g_sun = float(constants["G_SI"]) * float(constants["M_sun_kg"]) / float(constants["AU_m"]) ** 2
    ratio = (g_sun / raw["a0_m_s2"]) ** raw["power"]
    denominator = 1.0 - raw["amplitude"] / (1.0 + ratio)
    local_nu = 1.0 / denominator
    weak_nu = 1.0 / (1.0 - raw["amplitude"])
    filters = config["pre_response_filters"]
    keep = (
        (denominator >= float(filters["minimum_denominator"]))
        & (np.abs(local_nu - 1.0) <= float(filters["maximum_local_fractional_deviation"]))
        & (weak_nu <= float(filters["maximum_weak_field_enhancement"]))
        & np.isfinite(local_nu)
    )
    return {key: np.ascontiguousarray(value[keep]) for key, value in raw.items()}


def _raw_candidate_count(config: Mapping[str, Any]) -> int:
    generator = config["candidate_generator"]
    return (
        int(generator["amplitude_count"])
        * int(generator["a0_count"])
        * len(generator["powers"])
        * int(generator["stellar_mass_to_light_count"])
        * len(generator["gas_mass_scales"])
    )


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key], dtype="<f8")
        digest.update(key.encode() + b"\0")
        digest.update(values.tobytes())
    return digest.hexdigest()


def _as_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def _as_float(value: str) -> float | None:
    try:
        result = float(value.strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _build_sample(
    diskmass_rows: Sequence[Mapping[str, str]],
    alfalfa_rows: Sequence[Mapping[str, str]],
    kinematic_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    diskmass = {_as_int(row.get("UGC", "")): row for row in diskmass_rows}
    alfalfa = {_as_int(row.get("AGC", "")): row for row in alfalfa_rows}
    kinematic = {_as_int(row.get("UGC", "")): row for row in kinematic_rows}
    joined: list[dict[str, Any]] = []
    for ugc in sorted(set(diskmass) & set(alfalfa) & set(kinematic)):
        if ugc is None:
            continue
        d, a, k = diskmass[ugc], alfalfa[ugc], kinematic[ugc]
        hrv, vhel = _as_float(d.get("HRV", "")), _as_float(a.get("Vhel", ""))
        h_r = _as_float(d.get("hR", ""))
        log_m_hi = _as_float(a.get("logMHI", ""))
        d_alf = _as_float(a.get("Dist", ""))
        d_kin = _as_float(k.get("Dist", ""))
        k_abs = _as_float(k.get("KMag", ""))
        mu0 = _as_float(d.get("mu0", ""))
        color = _as_float(d.get("B-K", ""))
        snr = _as_float(a.get("SNR", ""))
        if (
            None in (hrv, vhel, h_r, log_m_hi, d_alf, d_kin, k_abs, mu0, snr)
            or abs(float(hrv) - float(vhel)) > 300.0
            or min(float(h_r), float(d_alf), float(d_kin), float(snr)) <= 0
        ):
            continue
        joined.append(
            {
                "ugc": int(ugc),
                "name": f"UGC{int(ugc):05d}",
                "type": str(d.get("Type", "")),
                "hR_arcsec": float(h_r),
                "mu0_R_mag_arcsec2": float(mu0),
                "B_minus_K": color,
                "K_abs_mag": float(k_abs),
                "distance_2025_Mpc": float(d_kin),
                "distance_alfalfa_Mpc": float(d_alf),
                "logMHI_alfalfa_Msun": float(log_m_hi),
                "e_logMHI_dex": _as_float(a.get("e_logMHI", "")),
                "HI_SNR": float(snr),
                "HI_code": _as_int(a.get("HI", "")),
                "velocity_match_km_s": abs(float(hrv) - float(vhel)),
            }
        )
    expected_join = int(config["sources"]["expected_predictor_join_before_exclusions"])
    if len(joined) != expected_join:
        raise GravityItem18Error(f"expected {expected_join} joined predictors, found {len(joined)}")
    old = {int(value) for value in config["sources"]["predecessor_exposed_ugc"]}
    snippets = {int(value) for value in config["sources"]["search_snippet_exposed_ugc"]}
    old_joined = sum(row["ugc"] in old for row in joined)
    if old_joined != int(config["sources"]["expected_predecessor_exposed_joined"]):
        raise GravityItem18Error("predecessor-exposed join count changed")
    objects = [row for row in joined if row["ugc"] not in old | snippets]
    if len(objects) != int(config["sources"]["expected_prefreeze_clean_eligible"]):
        raise GravityItem18Error(
            f"expected {config['sources']['expected_prefreeze_clean_eligible']} clean objects, "
            f"found {len(objects)}"
        )
    solar_k = float(config["physics"]["constants"]["M_K_sun"])
    for row in objects:
        luminosity = 10.0 ** (-0.4 * (float(row["K_abs_mag"]) - solar_k))
        distance_ratio = float(row["distance_2025_Mpc"]) / float(row["distance_alfalfa_Mpc"])
        m_hi = 10.0 ** float(row["logMHI_alfalfa_Msun"]) * distance_ratio**2
        row["K_luminosity_Lsun"] = luminosity
        row["HI_mass_rescaled_Msun"] = m_hi
        row["mass_proxy_Msun"] = 0.5 * luminosity + 1.4 * m_hi
    objects.sort(key=lambda row: (row["mass_proxy_Msun"], row["name"]))
    confirmation: set[str] = set()
    role_key = str(config["sample"]["role_key"])
    begin = 0
    for stratum, size in enumerate(config["sample"]["mass_stratum_sizes"]):
        group = objects[begin : begin + int(size)]
        begin += int(size)
        for row in group:
            row["mass_stratum"] = stratum
            row["role_rank"] = _hmac_rank(role_key, row["name"])
        confirmation.update(row["name"] for row in sorted(group, key=lambda x: x["role_rank"])[:2])
    if begin != len(objects):
        raise GravityItem18Error("mass stratum sizes do not exhaust the clean sample")
    exploration = [row for row in objects if row["name"] not in confirmation]
    fold_key = str(config["sample"]["fold_key"])
    exploration.sort(key=lambda row: _hmac_rank(fold_key, row["name"]))
    for index, row in enumerate(exploration):
        row["role"] = "exploration"
        row["outer_fold"] = index % int(config["sample"]["outer_folds"])
    for row in objects:
        if row["name"] in confirmation:
            row["role"] = "reserved_confirmation"
            row["outer_fold"] = None
    counts = {
        "joined_before_exclusions": len(joined),
        "excluded_predecessor_exposed": old_joined,
        "excluded_search_snippet_exposed": sum(row["ugc"] in snippets for row in joined),
        "eligible": len(objects),
        "exploration": len(exploration),
        "reserved_confirmation": len(confirmation),
    }
    return sorted(objects, key=lambda row: row["name"]), counts


def _prior_ugc_hits(root: Path, names: Sequence[str]) -> dict[str, list[str]]:
    """Find exact normalized UGC identities in predecessor results/responses only."""
    hits: dict[str, list[str]] = {}
    base = root / "runs" / "gravity" / "roadmap"
    candidates = list(base.glob("item-*.json"))
    candidates.extend(path for path in base.rglob("*response*") if path.is_file())
    for path in candidates:
        if "item-18-" in path.as_posix():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").upper()
        except OSError:
            continue
        for name in names:
            if name.upper() in text:
                hits.setdefault(name, []).append(path.relative_to(root).as_posix())
    return hits


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["predictor_diskmass"].parent.mkdir(parents=True, exist_ok=True)
    query_map = dict(config["sources"]["predictor_queries"])
    bodies: dict[str, bytes] = {}
    headers: dict[str, dict[str, str]] = {}
    for key, url in query_map.items():
        bodies[key], headers[key] = _download(str(url))
    columns = {
        "diskmass": ("UGC", "Type", "HRV", "Dist", "KMag", "B-K", "mu0", "hR", "Sel", "Ha", "HI"),
        "alfalfa": (
            "AGC",
            "Name",
            "Vhel",
            "HIflux",
            "e_HIflux",
            "SNR",
            "Dist",
            "e_Dist",
            "logMHI",
            "e_logMHI",
            "HI",
        ),
        "kinematic_predictors": (
            "UGC",
            "Kmag",
            "e_Kmag",
            "Dist",
            "e_Dist",
            "Ak",
            "Kcor",
            "KMag",
            "e_KMag",
        ),
    }
    rows = {key: _parse_vizier_tsv(bodies[key], columns[key]) for key in columns}
    file_keys = {
        "diskmass": "predictor_diskmass",
        "alfalfa": "predictor_alfalfa",
        "kinematic_predictors": "predictor_kinematic",
    }
    for key, path_key in file_keys.items():
        paths[path_key].write_bytes(bodies[key])
    objects, counts = _build_sample(
        rows["diskmass"], rows["alfalfa"], rows["kinematic_predictors"], config
    )
    prior_hits = _prior_ugc_hits(root, [str(row["name"]) for row in objects])
    if prior_hits:
        raise GravityItem18Error(f"remaining UGC identities overlap prior responses: {prior_hits}")
    exploration = [row for row in objects if row["role"] == "exploration"]
    confirmation = [row for row in objects if row["role"] == "reserved_confirmation"]
    sample = _content_hashed(
        {
            "schema_version": "invariant-gravity-item18-diskmass-sample-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selection_used_2025_response_values": False,
            "counts": {**counts, "response_values_read": 0},
            "prior_response_identity_hits": {},
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
            "excluded_identity_disclosure": {
                "2013_target_adjacent_values_seen": config["sources"]["predecessor_exposed_ugc"],
                "2025_response_rows_seen_in_search_snippet": config["sources"][
                    "search_snippet_exposed_ugc"
                ],
            },
            "claims": {"confirmation_opened": False},
        }
    )
    arrays = generate_candidates(config)
    candidate_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item18-antiscreening-candidates-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "algorithm": config["candidate_generator"]["algorithm"],
            "candidate_digest_sha256": _candidate_digest(arrays),
            "counts": {
                "raw_parameter_cells": _raw_candidate_count(config),
                "physics_admissible_cells": len(arrays["amplitude"]),
                "exact_parameter_equivalence_classes": len(arrays["amplitude"]),
                "post_response_cells": 0,
            },
            "law": config["physics"]["law"],
            "action": config["physics"]["action"],
            "creativity_label": config["candidate_generator"]["creativity_label"],
            "pre_response_filters": config["pre_response_filters"],
            "response_values_read": 0,
        }
    )
    files = []
    for source_key, path_key in file_keys.items():
        files.append(
            {
                "source": source_key,
                "path": paths[path_key].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths[path_key]),
                "rows": len(rows[source_key]),
                "selected_columns": list(columns[source_key]),
                "last_modified": headers[source_key].get("last-modified"),
            }
        )
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item18-predictors-1.0",
            "queries": query_map,
            "files": files,
            "response_columns_requested": [],
            "forbidden_columns_read": [],
        }
    )
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample)
    _write_json(paths["candidate_manifest"], candidate_manifest)
    if len(exploration) != int(config["sample"]["exploration_count"]) or len(confirmation) != int(
        config["sample"]["confirmation_count"]
    ):
        raise GravityItem18Error("sample role counts changed")
    return paths


def _load_prepared(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = _source_paths(root, config)
    payloads = [
        _read_json(paths["predictor_source_manifest"]),
        _read_json(paths["sample_manifest"]),
        _read_json(paths["candidate_manifest"]),
    ]
    for payload, label in zip(
        payloads, ("predictor manifest", "sample manifest", "candidate manifest"), strict=True
    ):
        _verify_content_hash(payload, label)
    predictor, sample, candidates = payloads
    for item in predictor["files"]:
        if _sha256_file(root / str(item["path"])) != str(item["sha256"]):
            raise GravityItem18Error(f"predictor file changed: {item['path']}")
    arrays = generate_candidates(config)
    if _candidate_digest(arrays) != candidates["candidate_digest_sha256"]:
        raise GravityItem18Error("candidate digest changed")
    if int(sample["counts"]["response_values_read"]) != 0 or bool(
        sample["claims"]["confirmation_opened"]
    ):
        raise GravityItem18Error("sample freeze contains response access")
    return predictor, sample, candidates


def _response_url(config: Mapping[str, Any], ugc: int) -> str:
    query = urllib.parse.urlencode(
        [
            ("-source", "J/ApJS/276/59/sample"),
            ("-out", ",".join(config["sources"]["response_columns"])),
            ("-out.max", "unlimited"),
            ("UGC", str(int(ugc))),
        ]
    )
    return f"{config['sources']['response_query_base']}?{query}"


def fetch_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    _, sample, candidates = _load_prepared(root, config)
    paths = _source_paths(root, config)
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    confirmation = {
        int(row["ugc"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    output: list[dict[str, str]] = []
    receipts: list[dict[str, Any]] = []
    columns = tuple(config["sources"]["response_columns"])
    for item in exploration:
        ugc = int(item["ugc"])
        if ugc in confirmation:
            raise GravityItem18Error("confirmation response requested")
        url = _response_url(config, ugc)
        body, headers = _download(url)
        rows = _parse_vizier_tsv(body, columns)
        if len(rows) != 1 or int(rows[0]["UGC"]) != ugc:
            raise GravityItem18Error(f"response identity mismatch for UGC {ugc}")
        output.append(dict(rows[0]))
        receipts.append(
            {
                "ugc": ugc,
                "query_sha256": _sha256_bytes(url.encode()),
                "response_sha256": _sha256_bytes(body),
                "last_modified": headers.get("last-modified"),
            }
        )
    output.sort(key=lambda row: int(row["UGC"]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(output)
    paths["exploration_responses"].write_text(buffer.getvalue(), encoding="utf-8", newline="")
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item18-response-source-1.0",
            "sample_freeze_commit": config["sample_freeze_commit"],
            "requested_objects": len(exploration),
            "returned_objects": len(output),
            "confirmation_objects_requested": 0,
            "confirmation_response_values_read": 0,
            "post_response_candidate_cells": int(candidates["counts"]["post_response_cells"]),
            "columns": list(columns),
            "excluded_columns": config["sources"]["excluded_response_columns"],
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
            },
            "per_object_source_receipts": receipts,
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["exploration_responses"]


def _disk_velocity_sq(
    mass: float | np.ndarray, scale_kpc: float, radius_kpc: float
) -> float | np.ndarray:
    y = radius_kpc / (2.0 * scale_kpc)
    shape = float(iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y))
    return 2.0 * 4.30091e-6 * mass / scale_kpc * y**2 * shape


def _load_rows(
    root: Path, config: Mapping[str, Any], gas_scale_hR: float | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(root, config)
    _, sample, _ = _load_prepared(root, config)
    manifest = _read_json(paths["response_source_manifest"])
    _verify_content_hash(manifest, "response source manifest")
    if int(manifest["confirmation_response_values_read"]) != 0:
        raise GravityItem18Error("confirmation response opened")
    if _sha256_file(paths["exploration_responses"]) != str(manifest["response_file"]["sha256"]):
        raise GravityItem18Error("response file changed")
    with paths["exploration_responses"].open(encoding="utf-8", newline="") as handle:
        response = {int(row["UGC"]): row for row in csv.DictReader(handle, delimiter="\t")}
    q = config["quality"]
    scale_ratio = float(gas_scale_hR or config["physics"]["primary_atomic_gas_scale_hR"])
    arcsec = float(config["physics"]["constants"]["arcsec_to_radian"])
    rows: list[dict[str, Any]] = []
    for predictor in sample["objects"]:
        if predictor["role"] != "exploration":
            continue
        ugc = int(predictor["ugc"])
        observed = response.get(ugc)
        if observed is None:
            continue
        values = {
            key: _as_float(observed.get(key, ""))
            for key in (
                "N",
                "sigMod",
                "inc",
                "e_inc",
                "Vrot",
                "e_Vrot",
                "hrot",
                "e_hrot",
                "Aphi",
                "Arc",
            )
        }
        low_flag = str(observed.get("f_inc", "")).strip().lower()
        required = (
            values["Vrot"],
            values["e_Vrot"],
            values["hrot"],
            values["e_hrot"],
            values["sigMod"],
        )
        quality = all(value is not None for value in required)
        if quality and bool(q["require_positive_finite_rotation_parameters"]):
            quality = (
                min(float(value) for value in required) >= 0
                and float(values["Vrot"]) > 0
                and float(values["hrot"]) > 0
            )
        quality = quality and float(predictor["HI_SNR"]) >= float(q["minimum_HI_SNR"])
        if bool(q["reject_low_inclination_flag"]):
            quality = quality and low_flag not in {"l", "low"}
        if values["Aphi"] is not None:
            quality = quality and float(values["Aphi"]) <= float(q["maximum_Aphi"])
        if values["Arc"] is not None:
            quality = quality and float(values["Arc"]) <= float(q["maximum_Arc"])
        if not quality:
            continue
        h_r_arcsec = float(predictor["hR_arcsec"])
        x = float(config["physics"]["evaluation_radius_hR"]) * h_r_arcsec / float(values["hrot"])
        v_obs = float(values["Vrot"]) * math.tanh(x)
        if v_obs <= 0:
            continue
        distance_kpc = float(predictor["distance_2025_Mpc"]) * 1000.0
        h_r_kpc = h_r_arcsec * arcsec * distance_kpc
        radius_kpc = float(config["physics"]["evaluation_radius_hR"]) * h_r_kpc
        l_k = float(predictor["K_luminosity_Lsun"])
        m_hi = float(predictor["HI_mass_rescaled_Msun"])
        star_v2_unit = float(_disk_velocity_sq(l_k, h_r_kpc, radius_kpc))
        gas_v2_unit = float(_disk_velocity_sq(1.4 * m_hi, scale_ratio * h_r_kpc, radius_kpc))
        sech2 = 1.0 / math.cosh(x) ** 2 if x < 350 else 0.0
        hrot_fraction = (
            x * sech2 / max(math.tanh(x), 1e-12) * float(values["e_hrot"]) / float(values["hrot"])
        )
        sigma_log_v = math.sqrt(
            (float(values["e_Vrot"]) / float(values["Vrot"])) ** 2
            + hrot_fraction**2
            + (float(values["sigMod"]) / max(v_obs, 1e-12)) ** 2
        )
        rows.append(
            {
                "ugc": ugc,
                "name": predictor["name"],
                "fold": int(predictor["outer_fold"]),
                "mass_stratum": int(predictor["mass_stratum"]),
                "log_v_obs": math.log(v_obs),
                "sigma_log_v": sigma_log_v,
                "radius_kpc": radius_kpc,
                "star_v2_unit": star_v2_unit,
                "gas_v2_unit": gas_v2_unit,
                "K_luminosity_Lsun": l_k,
                "HI_mass_Msun": m_hi,
                "hR_kpc": h_r_kpc,
                "mu0": float(predictor["mu0_R_mag_arcsec2"]),
                "gas_fraction_proxy": 1.4 * m_hi / (0.5 * l_k + 1.4 * m_hi),
            }
        )
    return sorted(rows, key=lambda row: row["name"]), manifest


def _prediction_matrix(
    arrays: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], xp: Any
) -> Any:
    amplitude = xp.asarray(arrays["amplitude"][:, None])
    a0 = xp.asarray(arrays["a0_m_s2"][:, None])
    power = xp.asarray(arrays["power"][:, None])
    stellar = xp.asarray(arrays["stellar_mass_to_light"][:, None])
    gas = xp.asarray(arrays["gas_mass_scale"][:, None])
    star_v2 = xp.asarray([row["star_v2_unit"] for row in rows])[None, :]
    gas_v2 = xp.asarray([row["gas_v2_unit"] for row in rows])[None, :]
    radius_kpc = xp.asarray([row["radius_kpc"] for row in rows])[None, :]
    v2 = stellar * star_v2 + gas * gas_v2
    kpc_to_m = 3.085677581491367e19
    gbar = v2 * 1e6 / (radius_kpc * kpc_to_m)
    denominator = 1.0 - amplitude / (1.0 + (gbar / a0) ** power)
    return 0.5 * xp.log(v2 / denominator)


def _gr_matrix(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], xp: Any
) -> tuple[Any, np.ndarray, np.ndarray]:
    generator = config["candidate_generator"]
    stellar = np.linspace(
        float(generator["stellar_mass_to_light_min"]),
        float(generator["stellar_mass_to_light_max"]),
        int(generator["stellar_mass_to_light_count"]),
    )
    gas = np.asarray(generator["gas_mass_scales"], dtype=np.float64)
    ss, gg = np.meshgrid(stellar, gas, indexing="ij")
    sf, gf = ss.reshape(-1), gg.reshape(-1)
    matrix = 0.5 * xp.log(
        xp.asarray(sf[:, None]) * xp.asarray([row["star_v2_unit"] for row in rows])[None, :]
        + xp.asarray(gf[:, None]) * xp.asarray([row["gas_v2_unit"] for row in rows])[None, :]
    )
    return matrix, sf, gf


def _oof_select(matrix: Any, y: Any, folds: np.ndarray, xp: Any) -> tuple[np.ndarray, list[int]]:
    prediction = xp.empty(len(folds), dtype=xp.float64)
    selected: list[int] = []
    for fold in range(5):
        train = xp.asarray(folds != fold)
        test = xp.asarray(folds == fold)
        loss = xp.mean((matrix[:, train] - y[train][None, :]) ** 2, axis=1)
        index = int(_to_numpy(xp.argmin(loss), xp))
        prediction[test] = matrix[index, test]
        selected.append(index)
    return _to_numpy(prediction, xp), selected


def _ridge_oof(rows: Sequence[Mapping[str, Any]], y: np.ndarray, alpha: float) -> np.ndarray:
    features = np.asarray(
        [
            [
                math.log(row["K_luminosity_Lsun"]),
                math.log(row["HI_mass_Msun"]),
                math.log(row["hR_kpc"]),
                row["mu0"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    folds = np.asarray([row["fold"] for row in rows])
    output = np.empty(len(rows))
    for fold in range(5):
        train, test = folds != fold, folds == fold
        mean, std = features[train].mean(axis=0), features[train].std(axis=0)
        std[std == 0] = 1.0
        x_train = np.column_stack([np.ones(train.sum()), (features[train] - mean) / std])
        x_test = np.column_stack([np.ones(test.sum()), (features[test] - mean) / std])
        penalty = np.eye(x_train.shape[1]) * alpha
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y[train])
        output[test] = x_test @ coef
    return output


def _btfr_oof(rows: Sequence[Mapping[str, Any]], y: np.ndarray) -> np.ndarray:
    x = 0.25 * np.log(
        np.asarray([0.5 * row["K_luminosity_Lsun"] + 1.4 * row["HI_mass_Msun"] for row in rows])
    )
    folds = np.asarray([row["fold"] for row in rows])
    output = np.empty(len(rows))
    for fold in range(5):
        train, test = folds != fold, folds == fold
        offset = float(np.mean(y[train] - x[train]))
        output[test] = x[test] + offset
    return output


def _slice_improvements(
    rows: Sequence[Mapping[str, Any]],
    candidate: np.ndarray,
    baseline: np.ndarray,
    y: np.ndarray,
    key: str,
) -> dict[str, float]:
    values = np.asarray(
        [row[key] if key in row else row["mass_stratum"] for row in rows], dtype=float
    )
    median = float(np.median(values))
    result = {}
    for label, mask in (("low", values <= median), ("high", values > median)):
        result[label] = _improvement(_mse(y[mask], baseline[mask]), _mse(y[mask], candidate[mask]))
    return result


def _synthetic_controls(config: Mapping[str, Any], xp: Any) -> dict[str, Any]:
    synthetic_rows = []
    for index in range(50):
        radius = 1.0 + 0.25 * index
        synthetic_rows.append(
            {
                "fold": index % 5,
                "radius_kpc": radius,
                "star_v2_unit": 2000.0 + 80.0 * index,
                "gas_v2_unit": 800.0 + 35.0 * index,
            }
        )
    arrays = generate_candidates(config)
    matrix = _prediction_matrix(arrays, synthetic_rows, xp)
    gr, _, _ = _gr_matrix(config, synthetic_rows, xp)
    folds = np.asarray([row["fold"] for row in synthetic_rows])
    truth_gr = _to_numpy(gr[37], xp) + 1e-6 * np.sin(np.arange(len(synthetic_rows)))
    cand_gr, _ = _oof_select(matrix, xp.asarray(truth_gr), folds, xp)
    gr_gr, _ = _oof_select(gr, xp.asarray(truth_gr), folds, xp)
    gr_improvement = _improvement(_mse(truth_gr, gr_gr), _mse(truth_gr, cand_gr))
    injection_index = int(len(arrays["amplitude"]) * 0.63)
    truth_injected = _to_numpy(matrix[injection_index], xp)
    cand_injected, selected = _oof_select(matrix, xp.asarray(truth_injected), folds, xp)
    gr_injected, _ = _oof_select(gr, xp.asarray(truth_injected), folds, xp)
    injected_improvement = _improvement(
        _mse(truth_injected, gr_injected), _mse(truth_injected, cand_injected)
    )
    return {
        "known_GR": {"candidate_improvement_vs_GR": gr_improvement, "pass": gr_improvement <= 0.0},
        "antiscreening_injection": {
            "injection_index": injection_index,
            "candidate_improvement_vs_GR": injected_improvement,
            "selected_indices": selected,
            "pass": injected_improvement > 0.5,
        },
    }


def run_experiment(root: Path) -> Path:
    start = time.perf_counter()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    _, sample, candidate_manifest = _load_prepared(root, config)
    rows, response_manifest = _load_rows(root, config)
    if len(rows) < int(config["gates"]["minimum_complete_exploration_objects"]):
        quality_pass = False
    else:
        quality_pass = True
    if not rows:
        raise GravityItem18Error("no valid exploration rows")
    arrays = generate_candidates(config)
    xp, backend, device = _backend()
    matrix = _prediction_matrix(arrays, rows, xp)
    gr_matrix, gr_stellar, gr_gas = _gr_matrix(config, rows, xp)
    y_np = np.asarray([row["log_v_obs"] for row in rows], dtype=np.float64)
    y = xp.asarray(y_np)
    folds = np.asarray([row["fold"] for row in rows], dtype=int)
    candidate_prediction, selected = _oof_select(matrix, y, folds, xp)
    gr_prediction, selected_gr = _oof_select(gr_matrix, y, folds, xp)
    fixed_stellar = np.argmin(abs(gr_stellar - 0.5) + abs(gr_gas - 1.0))
    fixed_prediction = _to_numpy(gr_matrix[fixed_stellar], xp)
    btfr = _btfr_oof(rows, y_np)
    flexible = _ridge_oof(rows, y_np, float(config["evaluation"]["ridge_alpha"]))
    losses = {
        "candidate": _mse(y_np, candidate_prediction),
        "fixed_GR": _mse(y_np, fixed_prediction),
        "calibrated_GR": _mse(y_np, gr_prediction),
        "baryonic_TF": _mse(y_np, btfr),
        "flexible_nuisance": _mse(y_np, flexible),
    }
    improvements = {
        key: _improvement(losses[key], losses["candidate"]) for key in losses if key != "candidate"
    }
    selected_cells = []
    for fold, index in enumerate(selected):
        selected_cells.append(
            {
                "fold": fold,
                "index": index,
                **{key: float(value[index]) for key, value in arrays.items()},
            }
        )
    mass_slices = _slice_improvements(
        rows, candidate_prediction, gr_prediction, y_np, "mass_stratum"
    )
    gas_slices = _slice_improvements(
        rows, candidate_prediction, gr_prediction, y_np, "gas_fraction_proxy"
    )
    rng = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    residual = y_np - gr_prediction
    observed_gain = improvements["calibrated_GR"]
    null_gains = []
    for _ in range(int(config["evaluation"]["permutation_trials"])):
        permutation = rng.permutation(len(rows))
        y_null_np = gr_prediction + residual[permutation]
        y_null = xp.asarray(y_null_np)
        candidate_null, _ = _oof_select(matrix, y_null, folds, xp)
        gr_null, _ = _oof_select(gr_matrix, y_null, folds, xp)
        null_gains.append(_improvement(_mse(y_null_np, gr_null), _mse(y_null_np, candidate_null)))
    p_value = (1.0 + sum(value >= observed_gain for value in null_gains)) / (1.0 + len(null_gains))
    robustness = {}
    for gas_scale in config["physics"]["gas_scale_robustness_hR"]:
        alt_rows, _ = _load_rows(root, config, float(gas_scale))
        if [row["name"] for row in alt_rows] != [row["name"] for row in rows]:
            raise GravityItem18Error("gas-scale robustness changed quality sample")
        alt_matrix = _prediction_matrix(arrays, alt_rows, xp)
        alt_gr, _, _ = _gr_matrix(config, alt_rows, xp)
        alt_candidate = np.empty(len(rows))
        alt_baseline = np.empty(len(rows))
        for fold in range(5):
            test = folds == fold
            alt_candidate[test] = _to_numpy(alt_matrix[selected[fold], xp.asarray(test)], xp)
            alt_baseline[test] = _to_numpy(alt_gr[selected_gr[fold], xp.asarray(test)], xp)
        robustness[str(gas_scale)] = _improvement(
            _mse(y_np, alt_baseline), _mse(y_np, alt_candidate)
        )
    controls = _synthetic_controls(config, xp)
    a0_logs = np.asarray([math.log10(cell["a0_m_s2"]) for cell in selected_cells])
    median_a0 = float(np.median(a0_logs))
    clustered = int(
        np.sum(
            np.abs(a0_logs - median_a0)
            <= float(config["gates"]["maximum_transition_scale_distance_from_median_dex"])
        )
    )
    power_count = max(Counter(cell["power"] for cell in selected_cells).values())
    gates = {
        "minimum_complete_exploration_objects": quality_pass,
        "confirmation_values_read": int(response_manifest["confirmation_response_values_read"])
        == 0,
        "post_response_candidate_cells": int(candidate_manifest["counts"]["post_response_cells"])
        == 0,
        "improvement_vs_fixed_GR": improvements["fixed_GR"]
        >= float(config["gates"]["minimum_mse_improvement_vs_fixed_GR"]),
        "improvement_vs_calibrated_GR": improvements["calibrated_GR"]
        >= float(config["gates"]["minimum_mse_improvement_vs_calibrated_GR"]),
        "improvement_vs_baryonic_TF": improvements["baryonic_TF"]
        >= float(config["gates"]["minimum_mse_improvement_vs_baryonic_TF"]),
        "improvement_vs_flexible_nuisance": improvements["flexible_nuisance"]
        >= float(config["gates"]["minimum_mse_improvement_vs_flexible_nuisance"]),
        "both_mass_halves": min(mass_slices.values())
        >= float(config["gates"]["minimum_each_mass_half_improvement_vs_calibrated_GR"]),
        "both_gas_fraction_halves": min(gas_slices.values())
        >= float(config["gates"]["minimum_each_gas_fraction_half_improvement_vs_calibrated_GR"]),
        "selection_aware_permutation": p_value
        <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
        "same_power_folds": power_count >= int(config["gates"]["minimum_same_power_folds"]),
        "transition_scale_clustered_folds": clustered
        >= int(config["gates"]["minimum_transition_scale_clustered_folds"]),
        "gas_scale_robustness": min(robustness.values()) >= 0.0,
        "known_GR_control": bool(controls["known_GR"]["pass"]),
        "synthetic_antiscreening_control": bool(controls["antiscreening_injection"]["pass"]),
    }
    decision = (
        "PASS_EXPLORATION"
        if all(gates.values())
        else (
            "INCONCLUSIVE_QUALITY"
            if not quality_pass
            else "REJECT_ITEM18_ANTISCREENING_EXPLORATION"
        )
    )
    if hasattr(xp, "get_default_memory_pool"):
        xp.get_default_memory_pool().free_all_blocks()
    compute = _content_hashed(
        {
            "schema_version": "invariant-gravity-item18-compute-1.0",
            "backend": backend,
            "device": device,
            "raw_parameter_cells": _raw_candidate_count(config),
            "physics_admissible_cells": len(arrays["amplitude"]),
            "null_inclusive_training_residual_evaluations": int(
                (1 + len(null_gains))
                * 5
                * len(arrays["amplitude"])
                * max(1, len(rows) - len(rows) // 5)
            ),
            "elapsed_seconds": time.perf_counter() - start,
            "paid_api_calls": 0,
            "api_spend_usd": 0.0,
        }
    )
    paths = _source_paths(root, config)
    _write_json(paths["compute_manifest"], compute)
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item18-diskmass-antiscreening-result-1.0",
            "item": 18,
            "hypothesis": config["hypothesis"],
            "provenance_and_creativity_label": config["candidate_generator"]["creativity_label"],
            "mathematical_definition": {
                "law": config["physics"]["law"],
                "action": config["physics"]["action"],
            },
            "dimensional_and_symmetry_checks": {
                "gbar_over_a0_dimensionless": True,
                "nu_dimensionless": True,
                "static_parity_even_rotationally_invariant": True,
                "one_AU_filter_applied_before_response": True,
                "full_covariant_stability_proved": False,
            },
            "data_source_receipt": {
                "predictor_manifest": paths["predictor_source_manifest"]
                .relative_to(root)
                .as_posix(),
                "sample_manifest": paths["sample_manifest"].relative_to(root).as_posix(),
                "response_manifest": paths["response_source_manifest"].relative_to(root).as_posix(),
                "exploration_available": int(sample["counts"]["exploration"]),
                "exploration_quality_valid": len(rows),
                "reserved_confirmation": int(sample["counts"]["reserved_confirmation"]),
                "confirmation_opened": 0,
                "prefreeze_contamination_excluded": sample["excluded_identity_disclosure"],
            },
            "frozen_boundary": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "post_response_candidate_cells": 0,
            },
            "candidate_counts": candidate_manifest["counts"],
            "baselines": config["evaluation"],
            "losses": losses,
            "improvements": improvements,
            "selected_cells_by_fold": selected_cells,
            "selected_GR_calibrations_by_fold": [
                {
                    "fold": fold,
                    "stellar_mass_to_light": float(gr_stellar[index]),
                    "gas_mass_scale": float(gr_gas[index]),
                }
                for fold, index in enumerate(selected_gr)
            ],
            "robustness": {
                "mass_halves_improvement_vs_calibrated_GR": mass_slices,
                "gas_fraction_halves_improvement_vs_calibrated_GR": gas_slices,
                "gas_disk_scale_replays_improvement_vs_calibrated_GR": robustness,
            },
            "selection_aware_permutation": {
                "trials": len(null_gains),
                "p_value": p_value,
                "null_gain_quantiles": {
                    str(q): float(np.quantile(null_gains, q)) for q in (0.05, 0.5, 0.95)
                },
            },
            "controls": controls,
            "gates": gates,
            "gates_passed": sum(gates.values()),
            "gates_total": len(gates),
            "counterexamples": [
                row["name"]
                for row, cand, base, obs in zip(
                    rows, candidate_prediction, gr_prediction, y_np, strict=True
                )
                if (cand - obs) ** 2 > (base - obs) ** 2
            ],
            "result": decision,
            "claim_limit": config["scope"]["claim_ceiling"],
            "compute_manifest": paths["compute_manifest"].relative_to(root).as_posix(),
            "compute": compute,
            "exact_next_action": "Advance to Item 19 massive gravitational particles. Preserve this exact finite-antiscreening family as a tested known-family region; do not retune the opened responses or open the eight confirmations.",
        }
    )
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, result)
    return result_path


def validate_result(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    path = root / str(config["paths"]["result"])
    result = _read_json(path)
    _verify_content_hash(result, "Item 18 result")
    if int(result["data_source_receipt"]["confirmation_opened"]) != 0:
        raise GravityItem18Error("result opened confirmation")
    if int(result["frozen_boundary"]["post_response_candidate_cells"]) != 0:
        raise GravityItem18Error("result admits post-response cells")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fetch-responses", "run", "validate"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare":
        paths = prepare_predictors(root)
        print(paths["sample_manifest"])
    elif args.command == "fetch-responses":
        print(fetch_responses(root))
    elif args.command == "run":
        print(run_experiment(root))
    else:
        print(validate_result(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
