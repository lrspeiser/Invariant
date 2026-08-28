"""Frozen Item 27 causal fading-memory search on fresh CALIFA data."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _backend,
    _download,
    _read_tsv,
    _to_numpy,
    _write_tsv,
)
from sigma_theory_compiler.gravity_item24_temporal_lapse import (
    _angular_separation_arcsec,
    _hmac_rank,
)
from sigma_theory_compiler.gravity_item25_time_varying_g import (
    _canonical_bytes,
    _content_hashed,
    _fit_candidate_predictions,
    _improvement,
    _linear_predict,
    _mse,
    _read_json,
    _ridge_predict,
    _select_candidate,
    _sha256_bytes,
    _sha256_file,
    _verify_content_hash,
    _write_json,
)

CONFIG_PATH = Path("configs/gravity_item27_gravitational_memory_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item27_gravitational_memory.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem27Error(RuntimeError):
    """Raised when an Item 27 freeze, leakage, or replay invariant is violated."""


def _git(root: Path, *args: str, text_mode: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=text_mode
    )
    return result.stdout.strip() if text_mode else result.stdout


def _require_ancestor(root: Path, commit: str, label: str) -> None:
    if commit.startswith("TO_BE_BOUND"):
        raise GravityItem27Error(f"{label} has not been bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise GravityItem27Error(f"{label} is not an ancestor of HEAD")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    if (
        config.get("schema_version")
        != "invariant-gravity-item27-gravitational-memory-config-1.0"
        or int(config.get("item", -1)) != 27
    ):
        raise GravityItem27Error("unexpected Item 27 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem27Error("stable gravity goal changed")
    if int(config["candidate_generator"]["raw_candidate_cells"]) != 262144:
        raise GravityItem27Error("raw candidate boundary changed")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem27Error("post-response candidates entered Item 27")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem27Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem27Error("paid calls are outside Item 27")
    policy = config["discovery_policy"]
    if not bool(policy["equal_initial_viability"]):
        raise GravityItem27Error("equal-viability policy changed")
    if not bool(policy["age_or_history_is_not_privileged"]):
        raise GravityItem27Error("age or history was privileged")
    for relative, digest in config["dependency_sha256"].items():
        if _sha256_file(root / str(relative)) != str(digest):
            raise GravityItem27Error(f"scientific dependency changed: {relative}")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    value.pop("implementation_correction_commit", None)
    value.pop("implementation_correction_scope", None)
    value.pop("implementation_correction_history", None)
    value.pop("response_access_incident", None)
    return _sha256_bytes(_canonical_bytes(value))


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen) != _contract_digest(config):
        raise GravityItem27Error("scientific contract differs from frozen commit")
    module_commit = str(config.get("implementation_correction_commit", commit))
    _require_ancestor(root, module_commit, "implementation correction")
    module = _git(
        root, "show", f"{module_commit}:{MODULE_PATH.as_posix()}", text_mode=False
    )
    if not isinstance(module, bytes) or _sha256_bytes(module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem27Error("Item 27 module differs from scientific freeze")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "predictors",
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
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        repo_path = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{repo_path}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem27Error(f"{key} differs from sample freeze")


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    per = int(config["discovery_policy"]["equal_raw_capacity_per_mechanism"])
    count = int(generator["raw_candidate_cells"])
    if count != 4 * per:
        raise GravityItem27Error("mechanism capacity is not equal")
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    arrays: dict[str, np.ndarray] = {"niche": np.repeat(np.arange(4, dtype=np.int8), per)}
    choices = (
        ("amplitude", generator["amplitudes"]),
        ("polarity", generator["polarities"]),
        ("tau1", generator["timescales_gyr"]),
        ("tau2", generator["timescales_gyr"]),
        ("stretch", generator["stretched_powers"]),
        ("mix", generator["mixture_weights"]),
        ("tail", generator["tail_powers"]),
        ("state", list(range(len(generator["state_modes"])))),
        ("a_transition", generator["acceleration_transitions_m_s2"]),
        ("accel_power", generator["acceleration_powers"]),
    )
    for key, values in choices:
        source = np.asarray(values)
        arrays[key] = source[random.integers(0, len(source), size=count)]
    return arrays


def _candidate_values(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    begin: int,
    end: int,
    xp: Any,
) -> dict[str, Any]:
    del config
    return {key: xp.asarray(value[begin:end]) for key, value in arrays.items()}


def _raw_candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.shape.__repr__().encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _kernel(values: Mapping[str, Any], ages_gyr: Any, xp: Any) -> Any:
    ages = xp.asarray(ages_gyr)[None, :]
    niche = values["niche"][:, None]
    tau1 = values["tau1"][:, None]
    tau2 = values["tau2"][:, None]
    stretch = values["stretch"][:, None]
    mix = values["mix"][:, None]
    tail = values["tail"][:, None]
    exponential = xp.exp(-ages / tau1)
    stretched = xp.exp(-xp.power(ages / tau1, stretch))
    mixture = mix * xp.exp(-ages / tau1) + (1.0 - mix) * xp.exp(-ages / tau2)
    scale_free = xp.power(1.0 + ages / tau1, -tail)
    return xp.where(
        niche == 0,
        exponential,
        xp.where(niche == 1, stretched, xp.where(niche == 2, mixture, scale_free)),
    )


def _kernel_l1(values: Mapping[str, Any], xp: Any) -> Any:
    niche = values["niche"]
    tau1 = values["tau1"]
    tau2 = values["tau2"]
    stretch = values["stretch"]
    mix = values["mix"]
    tail = values["tail"]
    exponential = tau1
    stretched = tau1 * xp.asarray([math.gamma(1.0 + 1.0 / float(v)) for v in _to_numpy(stretch, xp)])
    mixture = mix * tau1 + (1.0 - mix) * tau2
    scale_free = tau1 / (tail - 1.0)
    return xp.where(
        niche == 0,
        exponential,
        xp.where(niche == 1, stretched, xp.where(niche == 2, mixture, scale_free)),
    )


def _admissible_candidates(
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    physics = config["physics"]
    batch = int(config["evaluation"]["candidate_batch_size"])
    keep_parts: list[np.ndarray] = []
    local_max = 0.0
    kernel_min = math.inf
    normalization_max = 0.0
    monotonic_failures = 0
    nonintegrable = 0
    degenerate = 0
    domain_min = math.inf
    domain_max = -math.inf
    support = np.asarray([0.0, 0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 6.0, 10.0, 14.0])
    for begin in range(0, len(raw["niche"]), batch):
        end = min(begin + batch, len(raw["niche"]))
        values = _candidate_values(config, raw, begin, end, np)
        kernels = _kernel(values, support, np)
        normalization = np.abs(kernels[:, 0] - 1.0)
        monotone = np.all(np.diff(kernels, axis=1) <= 1e-12, axis=1)
        positive = np.all(np.isfinite(kernels) & (kernels >= 0.0), axis=1)
        l1 = _kernel_l1(values, np)
        integrable = np.isfinite(l1) & (l1 > 0.0)
        mixture_distinct = (values["niche"] != 2) | (
            (values["tau2"] >= 3.0 * values["tau1"])
            & (values["mix"] >= 0.1)
            & (values["mix"] <= 0.9)
        )
        log_mu_extreme = np.abs(values["amplitude"])
        mu_low = np.exp(-log_mu_extreme)
        mu_high = np.exp(log_mu_extreme)
        domain = (mu_low >= float(physics["minimum_mu_on_domain"])) & (
            mu_high <= float(physics["maximum_mu_on_domain"])
        )
        solar_window = 1.0 / (
            1.0
            + np.power(
                float(physics["solar_acceleration_m_s2"]) / values["a_transition"],
                values["accel_power"],
            )
        )
        local_q = (
            float(physics["solar_fractional_source_change_per_year"])
            * 1e9
            * l1
        )
        local_response = np.expm1(
            np.abs(values["amplitude"] * local_q * solar_window)
        )
        local = local_response <= float(physics["maximum_local_fractional_response"])
        keep = (
            (normalization <= 1e-12)
            & monotone
            & positive
            & integrable
            & mixture_distinct
            & domain
            & local
        )
        keep_parts.append(np.where(keep)[0] + begin)
        local_max = max(local_max, float(np.max(local_response[keep], initial=0.0)))
        kernel_min = min(kernel_min, float(np.min(kernels[keep], initial=1.0)))
        normalization_max = max(
            normalization_max, float(np.max(normalization[keep], initial=0.0))
        )
        monotonic_failures += int(np.count_nonzero(~monotone))
        nonintegrable += int(np.count_nonzero(~integrable))
        degenerate += int(np.count_nonzero(~mixture_distinct))
        domain_min = min(domain_min, float(np.min(mu_low[keep], initial=1.0)))
        domain_max = max(domain_max, float(np.max(mu_high[keep], initial=1.0)))
    indices = np.concatenate(keep_parts)
    arrays = {key: value[indices] for key, value in raw.items()}
    counts = Counter(int(value) for value in arrays["niche"])
    audit = {
        "raw_candidates": len(raw["niche"]),
        "raw_niche_counts": {
            str(niche): int(np.count_nonzero(raw["niche"] == niche)) for niche in range(4)
        },
        "admissible_candidates": len(indices),
        "admissible_niche_counts": {str(niche): counts[niche] for niche in range(4)},
        "raw_candidate_digest": _raw_candidate_digest(raw),
        "admissible_candidate_digest": _raw_candidate_digest(arrays),
        "advanced_support_cells": 0,
        "normalization_max_absolute_error": normalization_max,
        "minimum_admitted_kernel_value": kernel_min,
        "monotonicity_failure_cells": monotonic_failures,
        "nonintegrable_cells": nonintegrable,
        "degenerate_mixture_cells": degenerate,
        "maximum_admitted_local_fractional_response": local_max,
        "admitted_domain_mu_range": [domain_min, domain_max],
    }
    return arrays, audit


def _suggest_injections(arrays: Mapping[str, np.ndarray]) -> list[int]:
    output: list[int] = []
    for niche in range(4):
        indices = np.where(arrays["niche"] == niche)[0]
        score = (
            np.abs(np.log10(arrays["tau1"][indices] / 3.0))
            + np.abs(arrays["amplitude"][indices] - 1.0)
            + (arrays["polarity"][indices] != 1.0) * 2.0
            + (arrays["state"][indices] != 0) * 2.0
            + np.abs(np.log10(arrays["a_transition"][indices] / 1e-10))
        )
        output.append(int(indices[int(np.argmin(score))]))
    return output


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    arrays, audit = _admissible_candidates(config)
    injections = [
        int(value)
        for value in config["candidate_generator"]["synthetic_injection_admissible_indices"]
    ]
    if injections and (
        len(injections) != 4
        or [int(arrays["niche"][index]) for index in injections] != [0, 1, 2, 3]
    ):
        raise GravityItem27Error("synthetic injection boundary is invalid")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item27-memory-candidates-1.0",
            "generator": config["candidate_generator"],
            "physics_gates": config["physics"],
            "audit": audit,
            "synthetic_injection_admissible_indices": injections,
            "suggested_target_blind_injections": _suggest_injections(arrays),
            "responses_open_when_generated": False,
            "post_response_candidate_cells": 0,
        }
    )


def _normal_identity(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _parse_sexagesimal(value: str, *, hours: bool) -> float:
    text = value.strip()
    sign = -1.0 if text.startswith("-") else 1.0
    pieces = [float(piece) for piece in text.lstrip("+-").split(":")]
    while len(pieces) < 3:
        pieces.append(0.0)
    result = pieces[0] + pieces[1] / 60.0 + pieces[2] / 3600.0
    return result * 15.0 if hours else sign * result


def _csv_rows(body: bytes) -> list[list[str]]:
    text = body.decode("utf-8")
    return [
        [value.strip() for value in row]
        for row in csv.reader(io.StringIO(text))
        if row and not row[0].lstrip().startswith("#")
    ]


def _predecessor_exclusions(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    names: set[str] = set()
    coordinates: list[tuple[float, float, str]] = []
    files = 0
    for path in sorted(root.glob(str(config["sources"]["predecessor_sample_glob"]))):
        if path.parent.name.startswith("item-27-"):
            continue
        try:
            manifest = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        files += 1
        for row in manifest.get("objects", []):
            if not isinstance(row, Mapping):
                continue
            for key in ("name", "other_name", "normalized_identity"):
                if row.get(key):
                    names.add(_normal_identity(str(row[key])))
            identity = row.get("identity")
            if isinstance(identity, str) and not identity.isdigit():
                names.add(_normal_identity(identity))
            for prefix, key in (("UGC", "ugc"), ("NGC", "ngc")):
                if row.get(key) not in (None, ""):
                    try:
                        names.add(_normal_identity(f"{prefix}{int(row[key])}"))
                    except (TypeError, ValueError):
                        pass
            for ra_key, dec_key in (
                ("ra_deg", "dec_deg"),
                ("ra", "dec"),
                ("catalog_ra_deg", "catalog_dec_deg"),
            ):
                if row.get(ra_key) is not None and row.get(dec_key) is not None:
                    try:
                        coordinates.append(
                            (float(row[ra_key]), float(row[dec_key]), path.parent.name)
                        )
                    except (TypeError, ValueError):
                        pass
    return {"names": names, "coordinates": coordinates, "files": files}


def _download_source(url: str) -> tuple[bytes, dict[str, Any]]:
    body, headers = _download(url)
    return body, {
        "url": url,
        "sha256": _sha256_bytes(body),
        "bytes": len(body),
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
    }


def _history_summary(
    body: bytes,
    disk_scale_arcsec: float,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    from astropy.io import fits

    with fits.open(io.BytesIO(gzip.decompress(body)), memmap=False) as hdus:
        data = np.asarray(hdus[0].data[156:195], dtype=float)
        header = hdus[0].header
    ages = np.asarray(
        [
            float(
                re.search(r"age\s+0*([0-9.]+)", str(header[f"DESC_{index}"])).group(1)
            )
            for index in range(156, 195)
        ]
    )
    total = np.sum(data, axis=0)
    quality = config["predictor_quality"]
    valid = (
        np.all(np.isfinite(data), axis=0)
        & (total >= float(quality["minimum_fraction_sum"]))
        & (total <= float(quality["maximum_fraction_sum"]))
    )
    if np.count_nonzero(valid) < int(quality["minimum_valid_history_spaxels"]):
        return None
    yy, xx = np.indices(valid.shape)
    x0 = float(np.median(xx[valid]))
    y0 = float(np.median(yy[valid]))
    radius = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    inner = valid & (radius <= disk_scale_arcsec)
    outer = valid & (radius > disk_scale_arcsec) & (radius <= 2.0 * disk_scale_arcsec)
    if np.count_nonzero(inner) < int(quality["minimum_inner_history_spaxels"]):
        return None
    if np.count_nonzero(outer) < int(quality["minimum_outer_history_spaxels"]):
        return None

    def profile(mask: np.ndarray) -> list[float]:
        fractions = np.mean(data[:, mask] / total[mask], axis=1)
        fractions = np.maximum(fractions, 0.0)
        fractions /= np.sum(fractions)
        return [float(value) for value in fractions]

    global_profile = profile(valid)
    return {
        "ages_gyr": [float(value) for value in ages],
        "global": global_profile,
        "inner": profile(inner),
        "outer": profile(outer),
        "valid_spaxels": int(np.count_nonzero(valid)),
        "inner_spaxels": int(np.count_nonzero(inner)),
        "outer_spaxels": int(np.count_nonzero(outer)),
        "fossil_mean_age_gyr": float(np.sum(ages * np.asarray(global_profile))),
        "fossil_recent_1gyr_fraction": float(
            np.sum(np.asarray(global_profile)[ages <= 1.0])
        ),
    }


def _predictor_rows(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = config["sources"]
    base = str(sources["base_url"])
    source_receipts: list[dict[str, Any]] = []
    tables: dict[str, list[list[str]]] = {}
    for key in (
        "pipe3d_object_table",
        "pipe3d_mean_table",
        "pipe3d_re_table",
        "photometric_table",
    ):
        body, receipt = _download_source(base + str(sources[key]))
        source_receipts.append(receipt)
        tables[key] = _csv_rows(body)
    for key in ("pipe3d_object_table", "pipe3d_mean_table", "pipe3d_re_table"):
        if len(tables[key]) != int(sources["expected_pipe3d_rows"]):
            raise GravityItem27Error(f"{key} row count changed")
    index_body, index_receipt = _download_source(base + str(sources["kinematic_index"]))
    source_receipts.append(index_receipt)
    advertised = {
        _normal_identity(match.group(1)): match.group(1)
        for match in re.finditer(
            r'href="([^"/]+)\.CALIFA\.V1200\.stekin\.fits"',
            index_body.decode("utf-8"),
        )
    }
    if len(advertised) != int(sources["expected_kinematic_identities"]):
        raise GravityItem27Error("advertised kinematic identity count changed")
    object_rows = {_normal_identity(row[0]): row for row in tables["pipe3d_object_table"]}
    mean_rows = {_normal_identity(row[0]): row for row in tables["pipe3d_mean_table"]}
    re_rows = {_normal_identity(row[0]): row for row in tables["pipe3d_re_table"]}
    photo_rows: dict[str, list[str]] = {}
    for row in tables["photometric_table"]:
        key = _normal_identity(row[1])
        previous = photo_rows.get(key)
        if previous is None or int(float(row[6])) > int(float(previous[6])):
            photo_rows[key] = row
    intersection = sorted(set(object_rows) & set(mean_rows) & set(re_rows) & set(photo_rows) & set(advertised))
    if len(intersection) != int(sources["expected_predictor_kinematic_intersection"]):
        raise GravityItem27Error("predictor/kinematic identity intersection changed")
    exclusions = _predecessor_exclusions(root, config)
    cache = root / "work/gravity/item27-califa-sfh-cache"
    cache.mkdir(parents=True, exist_ok=True)

    def acquire_history(key: str) -> tuple[str, bytes, dict[str, Any]]:
        name = object_rows[key][0]
        url = base + str(sources["sfh_template"]).format(name=name)
        path = cache / f"{name}.SFH.cube.fits.gz"
        if path.exists():
            body = path.read_bytes()
            receipt = {
                "url": url,
                "sha256": _sha256_bytes(body),
                "bytes": len(body),
                "cache_reused": True,
            }
        else:
            body, receipt = _download_source(url)
            path.write_bytes(body)
            receipt["cache_reused"] = False
        return key, body, receipt

    histories: dict[str, tuple[bytes, dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for key, body, receipt in pool.map(acquire_history, intersection):
            histories[key] = (body, receipt)
    quality = config["predictor_quality"]
    output: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    name_overlaps: list[str] = []
    coordinate_overlaps: list[dict[str, Any]] = []
    history_receipts: list[dict[str, Any]] = []
    for key in intersection:
        obj = object_rows[key]
        mean = mean_rows[key]
        photo = photo_rows[key]
        name = obj[0]
        history_body, history_receipt = histories[key]
        history_receipts.append({"name": name, **history_receipt})
        ra = _parse_sexagesimal(obj[2], hours=True)
        dec = _parse_sexagesimal(obj[3], hours=False)
        if key in exclusions["names"]:
            name_overlaps.append(name)
            continue
        matches = [
            (sep, label)
            for prior_ra, prior_dec, label in exclusions["coordinates"]
            if (
                sep := _angular_separation_arcsec(ra, dec, prior_ra, prior_dec)
            )
            < float(sources["predecessor_coordinate_veto_arcsec"])
        ]
        if matches:
            nearest = min(matches)
            coordinate_overlaps.append(
                {"name": name, "separation_arcsec": nearest[0], "source": nearest[1]}
            )
            continue
        log_mass = float(obj[5])
        log_sfr = float(obj[7])
        redshift = float(obj[4])
        disk = int(float(photo[6]))
        bulge = int(float(photo[5]))
        disk_scale = float(photo[81])
        axis_ratio = float(photo[83])
        position_angle = float(photo[85])
        bulge_fraction = float(photo[78])
        disk_fraction = float(photo[91])
        checks = {
            "mass": float(quality["minimum_log_stellar_mass"])
            <= log_mass
            <= float(quality["maximum_log_stellar_mass"]),
            "redshift": float(quality["minimum_redshift"])
            <= redshift
            <= float(quality["maximum_redshift"]),
            "disk_component": disk == 1,
            "disk_scale": float(quality["minimum_disk_scale_arcsec"])
            <= disk_scale
            <= float(quality["maximum_disk_scale_arcsec"]),
            "axis_ratio": float(quality["minimum_disk_axis_ratio"])
            <= axis_ratio
            <= float(quality["maximum_disk_axis_ratio"]),
            "position_angle": 0.0 <= position_angle <= 180.0,
            "bulge_fraction": 0.0 <= bulge_fraction <= 1.0,
            "disk_fraction": 0.0 <= disk_fraction <= 1.0,
        }
        failed = [label for label, passed in checks.items() if not passed]
        if failed:
            failures.update(failed)
            continue
        history = _history_summary(history_body, disk_scale, config)
        if history is None:
            failures["history_coverage"] += 1
            continue
        distance_mpc = float(config["physics"]["c_km_s"]) * redshift / 70.0
        scale_kpc_arcsec = distance_mpc * 1000.0 / 206265.0
        output.append(
            {
                "name": name,
                "normalized_identity": key,
                "califa_id": int(obj[1]),
                "ra_deg": ra,
                "dec_deg": dec,
                "redshift": redshift,
                "distance_Mpc_proxy": distance_mpc,
                "log_stellar_mass": log_mass,
                "log_SFR": log_sfr,
                "mean_log_age_year": float(mean[1]),
                "mean_metallicity_ZH": float(mean[3]),
                "disk_scale_arcsec": disk_scale,
                "disk_scale_kpc": disk_scale * scale_kpc_arcsec,
                "disk_axis_ratio": axis_ratio,
                "disk_position_angle_deg": position_angle,
                "bulge_component": bulge,
                "bulge_fraction_r": bulge_fraction,
                "disk_fraction_r": disk_fraction,
                "history_valid_spaxels": history["valid_spaxels"],
                "history_inner_spaxels": history["inner_spaxels"],
                "history_outer_spaxels": history["outer_spaxels"],
                "fossil_mean_age_gyr": history["fossil_mean_age_gyr"],
                "fossil_recent_1gyr_fraction": history["fossil_recent_1gyr_fraction"],
                "history_ages_gyr": json.dumps(history["ages_gyr"], separators=(",", ":")),
                "history_global": json.dumps(history["global"], separators=(",", ":")),
                "history_inner": json.dumps(history["inner"], separators=(",", ":")),
                "history_outer": json.dumps(history["outer"], separators=(",", ":")),
                "kinematic_filename": f"{advertised[key]}.CALIFA.V1200.stekin.fits",
            }
        )
    audit = {
        "source_receipts": source_receipts,
        "history_source_receipts": history_receipts,
        "pipe3d_rows": len(object_rows),
        "photometric_identities": len(photo_rows),
        "advertised_kinematic_identities": len(advertised),
        "predictor_kinematic_intersection": len(intersection),
        "predecessor_manifest_files": int(exclusions["files"]),
        "predecessor_names": len(exclusions["names"]),
        "predecessor_coordinates": len(exclusions["coordinates"]),
        "name_overlaps": name_overlaps,
        "coordinate_overlaps": coordinate_overlaps,
        "quality_failures": dict(sorted(failures.items())),
        "safe_predictor_eligible": len(output),
        "response_columns_read": [],
        "kinematic_files_opened": 0,
    }
    return sorted(output, key=lambda row: str(row["name"])), audit


def _build_sample(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    sample = config["sample"]
    groups = np.array_split(
        np.asarray(
            sorted(rows, key=lambda row: (float(row["log_stellar_mass"]), str(row["name"]))),
            dtype=object,
        ),
        int(sample["mass_strata"]),
    )
    objects: list[dict[str, Any]] = []
    for stratum, values in enumerate(groups):
        group = [dict(value) for value in values.tolist()]
        ranked = sorted(
            group,
            key=lambda row: _hmac_rank(str(sample["role_key"]), f"califa:{row['normalized_identity']}"),
        )
        confirmations = {
            str(row["normalized_identity"])
            for row in ranked[: int(sample["confirmation_per_stratum"])]
        }
        exploration = sorted(
            [row for row in ranked if str(row["normalized_identity"]) not in confirmations],
            key=lambda row: _hmac_rank(str(sample["fold_key"]), f"califa:{row['normalized_identity']}"),
        )
        folds = {
            str(row["normalized_identity"]): int((index + stratum) % int(sample["outer_folds"]))
            for index, row in enumerate(exploration)
        }
        for row in group:
            identity = str(row["normalized_identity"])
            role = "confirmation" if identity in confirmations else "exploration"
            objects.append(
                {
                    "identity": identity,
                    "name": row["name"],
                    "califa_id": int(row["califa_id"]),
                    "role": role,
                    "mass_stratum": stratum,
                    "outer_fold": None if role == "confirmation" else folds[identity],
                    "ra_deg": float(row["ra_deg"]),
                    "dec_deg": float(row["dec_deg"]),
                    "kinematic_filename": row["kinematic_filename"],
                    "role_rank_sha256": _hmac_rank(
                        str(sample["role_key"]), f"califa:{identity}"
                    ),
                }
            )
    roles = Counter(str(row["role"]) for row in objects)
    fold_counts = Counter(
        int(row["outer_fold"]) for row in objects if row["role"] == "exploration"
    )
    if len(objects) != int(sample["expected_safe_predictor_eligible"]):
        raise GravityItem27Error(f"selected {len(objects)} predictor rows")
    if roles["exploration"] != int(sample["expected_exploration"]):
        raise GravityItem27Error("unexpected exploration count")
    if roles["confirmation"] != int(sample["expected_confirmation"]):
        raise GravityItem27Error("unexpected confirmation count")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item27-memory-sample-1.0",
            "selection_rule": sample["rule"],
            "response_columns_read": [],
            "confirmation_response_values_read": 0,
            "objects": sorted(objects, key=lambda row: str(row["identity"])),
            "role_counts": dict(sorted(roles.items())),
            "fold_counts": {str(key): value for key, value in sorted(fold_counts.items())},
        }
    )


def audit_predictors(root: Path) -> dict[str, Any]:
    config = load_config(root)
    rows, audit = _predictor_rows(root, config)
    return {"eligible": len(rows), "audit": audit}


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["predictors"].parent.mkdir(parents=True, exist_ok=True)
    rows, audit = _predictor_rows(root, config)
    if len(rows) != int(config["sample"]["expected_safe_predictor_eligible"]):
        raise GravityItem27Error(f"safe predictor count changed: {len(rows)}")
    columns = list(rows[0])
    _write_tsv(paths["predictors"], rows, columns)
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item27-memory-predictors-1.0",
            "audit": audit,
            "predictor_file": {
                "path": paths["predictors"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["predictors"]),
                "rows": len(rows),
            },
        }
    )
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], _build_sample(rows, config))
    _write_json(paths["candidate_manifest"], _candidate_manifest(config))
    return paths


def _response_summary(
    body: bytes,
    predictor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from astropy.io import fits

    with fits.open(io.BytesIO(body), memmap=False) as hdus:
        table = hdus[1].data
        columns = {str(name).upper(): str(name) for name in table.names}
        names = set(columns)
        required = {
            "BIN_ID",
            "XBIN",
            "YBIN",
            "SNR_BIN",
            "VP",
            "DVP",
            "SP",
            "DSP",
            "QC",
        }
        if not required.issubset(names):
            raise GravityItem27Error(f"V1200 schema changed: {sorted(names)}")
        bin_id = np.asarray(table[columns["BIN_ID"]], dtype=int)
        _, unique = np.unique(bin_id, return_index=True)
        x = np.asarray(table[columns["XBIN"]][unique], dtype=float)
        y = np.asarray(table[columns["YBIN"]][unique], dtype=float)
        snr = np.asarray(table[columns["SNR_BIN"]][unique], dtype=float)
        velocity = np.asarray(table[columns["VP"]][unique], dtype=float)
        velocity_error = np.asarray(table[columns["DVP"]][unique], dtype=float)
        dispersion = np.asarray(table[columns["SP"]][unique], dtype=float)
        dispersion_error = np.asarray(table[columns["DSP"]][unique], dtype=float)
        qc = np.asarray(table[columns["QC"]][unique], dtype=float)
    extraction = config["response_extraction"]
    q = float(predictor["disk_axis_ratio"])
    q0 = float(extraction["inclination_intrinsic_axis_ratio"])
    cos2 = np.clip((q * q - q0 * q0) / (1.0 - q0 * q0), 0.0, 1.0)
    sin_i = float(np.sqrt(1.0 - cos2))
    angle = math.radians(float(predictor["disk_position_angle_deg"]))
    major = x * math.cos(angle) + y * math.sin(angle)
    minor = -x * math.sin(angle) + y * math.cos(angle)
    radius = np.sqrt(major**2 + (minor / max(q, 0.1)) ** 2) / float(
        predictor["disk_scale_arcsec"]
    )
    base_quality = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(snr)
        & np.isfinite(velocity)
        & np.isfinite(velocity_error)
        & np.isfinite(dispersion)
        & np.isfinite(dispersion_error)
        & np.isfinite(qc)
        & (snr >= float(extraction["minimum_bin_snr"]))
        & (velocity_error <= float(extraction["maximum_velocity_error_km_s"]))
        & (dispersion_error <= float(extraction["maximum_dispersion_error_km_s"]))
        & (dispersion >= float(extraction["minimum_dispersion_km_s"]))
        & (dispersion <= float(extraction["maximum_dispersion_km_s"]))
    )
    if np.count_nonzero(base_quality) < int(extraction["minimum_unique_bins_primary"]):
        return None, {"failure": "insufficient_total_quality_bins", "quality_bins": int(np.count_nonzero(base_quality))}
    systemic = float(np.median(velocity[base_quality]))
    low_percentile, high_percentile = [
        float(value) for value in extraction["rotation_percentiles"]
    ]

    def annulus(label: str, bounds: Sequence[float], minimum: int) -> dict[str, Any] | None:
        selected = base_quality & (radius >= float(bounds[0])) & (radius < float(bounds[1]))
        if np.count_nonzero(selected) < minimum:
            return None
        line_velocity = velocity[selected] - systemic
        low, high = np.percentile(line_velocity, [low_percentile, high_percentile])
        rotation = float((high - low) / (2.0 * max(sin_i, 0.2)))
        sigma = float(np.median(dispersion[selected]))
        vrms = float(math.sqrt(rotation**2 + sigma**2))
        if not (
            rotation >= float(extraction["minimum_rotation_km_s"])
            and vrms <= float(extraction["maximum_vrms_km_s"])
        ):
            return None
        return {
            "label": label,
            "bins": int(np.count_nonzero(selected)),
            "rotation_km_s": rotation,
            "dispersion_km_s": sigma,
            "vrms_km_s": vrms,
            "median_qc": float(np.median(qc[selected])),
        }

    primary = annulus(
        "primary",
        extraction["primary_annulus_disk_scale"],
        int(extraction["minimum_unique_bins_primary"]),
    )
    inner = annulus(
        "inner",
        extraction["inner_replay_annulus_disk_scale"],
        int(extraction["minimum_unique_bins_replay"]),
    )
    outer = annulus(
        "outer",
        extraction["outer_replay_annulus_disk_scale"],
        int(extraction["minimum_unique_bins_replay"]),
    )
    missing = [
        label
        for label, value in (("primary", primary), ("inner", inner), ("outer", outer))
        if value is None
    ]
    if primary is None:
        return None, {
            "failure": "annulus_quality",
            "missing_annuli": missing,
            "quality_bins": int(np.count_nonzero(base_quality)),
        }
    assert primary is not None
    return {
        "primary_vrms_km_s": primary["vrms_km_s"],
        "inner_vrms_km_s": float("nan") if inner is None else inner["vrms_km_s"],
        "outer_vrms_km_s": float("nan") if outer is None else outer["vrms_km_s"],
        "primary_rotation_km_s": primary["rotation_km_s"],
        "primary_dispersion_km_s": primary["dispersion_km_s"],
        "primary_bins": primary["bins"],
        "inner_bins": 0 if inner is None else inner["bins"],
        "outer_bins": 0 if outer is None else outer["bins"],
        "median_primary_qc": primary["median_qc"],
    }, {
        "failure": None,
        "missing_annuli": missing,
        "quality_bins": int(np.count_nonzero(base_quality)),
    }


def acquire_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    predictors = {str(row["normalized_identity"]): row for row in _read_tsv(paths["predictors"])}
    sample = _read_json(paths["sample_manifest"])
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    if any(row["role"] == "confirmation" for row in exploration):
        raise GravityItem27Error("confirmation entered response acquisition")
    base = str(config["sources"]["base_url"])
    template = str(config["sources"]["kinematic_template"])
    cache = root / "work/gravity/item27-califa-response-cache"
    cache.mkdir(parents=True, exist_ok=True)

    def acquire(row: Mapping[str, Any]) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
        identity = str(row["identity"])
        predictor = predictors[identity]
        name = str(predictor["name"])
        url = base + template.format(name=name)
        body, receipt = _download_source(url)
        cache_path = cache / str(predictor["kinematic_filename"])
        cache_path.write_bytes(body)
        return dict(row), body, receipt

    records: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    all_annuli = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        acquired = list(pool.map(acquire, exploration))
    for sample_row, body, receipt in acquired:
        identity = str(sample_row["identity"])
        predictor = predictors[identity]
        summary, quality = _response_summary(body, predictor, config)
        receipts.append(
            {
                "identity": identity,
                "name": predictor["name"],
                **receipt,
                "quality": quality,
            }
        )
        if summary is None:
            failures[str(quality["failure"])] += 1
            continue
        if not quality.get("missing_annuli"):
            all_annuli += 1
        records.append(
            {
                "identity": identity,
                "name": predictor["name"],
                "fold": int(sample_row["outer_fold"]),
                "mass_stratum": int(sample_row["mass_stratum"]),
                **summary,
            }
        )
    columns = list(records[0]) if records else ["identity", "name"]
    _write_tsv(paths["exploration_responses"], records, columns)
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item27-memory-responses-1.0",
            "exact_exploration_files_queried": len(exploration),
            "valid_primary": len(records),
            "valid_all_annuli": all_annuli,
            "quality_failures": dict(sorted(failures.items())),
            "source_receipts": receipts,
            "response_columns_read": [
                "BIN_ID",
                "XBIN",
                "YBIN",
                "SNR_BIN",
                "Vp",
                "DVp",
                "Sp",
                "DSp",
                "QC",
            ],
            "confirmation_files_queried": 0,
            "confirmation_values_read": 0,
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
                "rows": len(records),
            },
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["exploration_responses"]


def _history_arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    ages = np.asarray(json.loads(str(rows[0]["history_ages_gyr"])), dtype=float)
    modes = []
    for key in ("history_global", "history_inner", "history_outer"):
        values = np.asarray([json.loads(str(row[key])) for row in rows], dtype=float)
        modes.append(values)
    modes.append(modes[2] - modes[1])
    return ages, np.stack(modes, axis=0)


def _radius_factor(config: Mapping[str, Any], label: str) -> float:
    key = {
        "primary": "primary_annulus_disk_scale",
        "inner": "inner_replay_annulus_disk_scale",
        "outer": "outer_replay_annulus_disk_scale",
    }[label]
    low, high = [float(value) for value in config["response_extraction"][key]]
    return 0.5 * (low + high)


def _row_physics(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], label: str
) -> tuple[np.ndarray, np.ndarray]:
    radius_factor = _radius_factor(config, label)
    mass = np.power(10.0, np.asarray([float(row["log_stellar_mass"]) for row in rows]))
    scale = np.asarray([float(row["disk_scale_kpc"]) for row in rows])
    radius = radius_factor * scale
    enclosed = 1.0 - np.exp(-radius_factor) * (1.0 + radius_factor)
    velocity = np.sqrt(
        float(config["physics"]["G_kpc_km2_s2_Msun"]) * mass * enclosed / radius
    )
    acceleration = (
        float(config["physics"]["G_kpc_km2_s2_Msun"])
        * mass
        * enclosed
        / radius**2
        * 1e6
        / float(config["physics"]["meters_per_kpc"])
    )
    return velocity, acceleration


def _build_term_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> np.ndarray:
    ages, histories = _history_arrays(rows)
    _, acceleration = _row_physics(rows, config, label)
    batch = int(config["evaluation"]["candidate_batch_size"])
    pieces: list[np.ndarray] = []
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        values = _candidate_values(config, arrays, begin, end, np)
        kernel = _kernel(values, ages, np)
        state_values = np.stack([kernel @ histories[index].T for index in range(4)], axis=0)
        row_index = np.arange(end - begin)
        q_memory = state_values[values["state"].astype(int), row_index]
        window = 1.0 / (
            1.0
            + np.power(
                acceleration[None, :] / values["a_transition"][:, None],
                values["accel_power"][:, None],
            )
        )
        log_mu = (
            values["polarity"][:, None]
            * values["amplitude"][:, None]
            * q_memory
            * window
        )
        pieces.append(0.5 * log_mu / math.log(10.0))
    return np.concatenate(pieces, axis=0)


def _base_design(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], label: str
) -> np.ndarray:
    velocity, _ = _row_physics(rows, config, label)
    return np.column_stack([np.ones(len(rows)), np.log10(velocity) - 2.0])


def _flex_design(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], label: str
) -> np.ndarray:
    base = _base_design(rows, config, label)[:, 1]
    mass = np.asarray([float(row["log_stellar_mass"]) for row in rows]) - 10.5
    size = np.log10(np.asarray([float(row["disk_scale_kpc"]) for row in rows]))
    surface = mass - 2.0 * size - 9.0
    sfr = np.asarray([float(row["log_SFR"]) for row in rows])
    age = np.asarray([float(row["mean_log_age_year"]) for row in rows]) - 9.5
    metallicity = np.asarray([float(row["mean_metallicity_ZH"]) for row in rows])
    axis = np.asarray([float(row["disk_axis_ratio"]) for row in rows])
    inclination = np.sqrt(
        np.clip(1.0 - (axis**2 - 0.04) / 0.96, 0.0, 1.0)
    )
    bulge = np.asarray([float(row["bulge_fraction_r"]) for row in rows])
    disk = np.asarray([float(row["disk_fraction_r"]) for row in rows])
    redshift = np.asarray([float(row["redshift"]) for row in rows]) * 50.0
    return np.column_stack(
        [
            base,
            mass,
            size,
            surface,
            sfr,
            age,
            metallicity,
            axis,
            inclination,
            bulge,
            disk,
            redshift,
            mass * age,
            surface * age,
            sfr * age,
            bulge * axis,
        ]
    )


def _target(rows: Sequence[Mapping[str, Any]], label: str) -> np.ndarray:
    return np.log10(np.asarray([float(row[f"{label}_vrms_km_s"]) for row in rows]))


def _oof_search(
    xp: Any,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    folds: np.ndarray,
    term_matrix: np.ndarray,
    label: str,
) -> dict[str, Any]:
    base = _base_design(rows, config, label)
    flexible = _flex_design(rows, config, label)
    candidate_prediction = np.full(len(rows), np.nan)
    base_prediction = np.full(len(rows), np.nan)
    flexible_prediction = np.full(len(rows), np.nan)
    selected: list[int] = []
    residual_evaluations = 0
    for outer in sorted({int(value) for value in folds}):
        test = np.where(folds == outer)[0]
        train = np.where(folds != outer)[0]
        index, evaluations = _select_candidate(
            xp, config, base, target, folds, outer, term_matrix
        )
        selected.append(index)
        residual_evaluations += evaluations
        terms = xp.asarray(term_matrix[index : index + 1])
        prediction = _fit_candidate_predictions(xp, base, target, terms, train, test)
        candidate_prediction[test] = _to_numpy(prediction[0], xp)
        base_prediction[test] = _linear_predict(base, target, train, test)
        flexible_prediction[test] = _ridge_predict(
            flexible,
            target,
            train,
            test,
            float(config["evaluation"]["ridge_alpha"]),
        )
    return {
        "candidate": candidate_prediction,
        "base": base_prediction,
        "flexible": flexible_prediction,
        "selected": selected,
        "residual_evaluations": residual_evaluations,
    }


def _fixed_oof(
    xp: Any,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    folds: np.ndarray,
    term_matrix: np.ndarray,
    selected: Sequence[int],
    label: str,
) -> dict[str, np.ndarray]:
    base = _base_design(rows, config, label)
    flexible = _flex_design(rows, config, label)
    output = {key: np.full(len(rows), np.nan) for key in ("candidate", "base", "flexible")}
    for index, outer in zip(selected, sorted({int(value) for value in folds}), strict=True):
        test = np.where(folds == outer)[0]
        train = np.where(folds != outer)[0]
        prediction = _fit_candidate_predictions(
            xp, base, target, xp.asarray(term_matrix[index : index + 1]), train, test
        )
        output["candidate"][test] = _to_numpy(prediction[0], xp)
        output["base"][test] = _linear_predict(base, target, train, test)
        output["flexible"][test] = _ridge_predict(
            flexible,
            target,
            train,
            test,
            float(config["evaluation"]["ridge_alpha"]),
        )
    return output


def _candidate_record(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], index: int
) -> dict[str, Any]:
    values = _candidate_values(config, arrays, index, index + 1, np)
    niche = int(values["niche"][0])
    record: dict[str, Any] = {
        "index": index,
        "niche": niche,
        "niche_id": config["candidate_generator"]["niches"][niche]["id"],
        "state_mode": config["candidate_generator"]["state_modes"][int(values["state"][0])],
    }
    for key in (
        "amplitude",
        "polarity",
        "tau1",
        "tau2",
        "stretch",
        "mix",
        "tail",
        "a_transition",
        "accel_power",
    ):
        record[key] = float(values[key][0])
    return record


def _load_joined_rows(
    paths: Mapping[str, Path], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictors = {str(row["normalized_identity"]): row for row in _read_tsv(paths["predictors"])}
    responses = _read_tsv(paths["exploration_responses"])
    sample = _read_json(paths["sample_manifest"])
    response_manifest = _read_json(paths["response_source_manifest"])
    roles = {str(row["identity"]): row for row in sample["objects"]}
    rows: list[dict[str, Any]] = []
    for response in responses:
        identity = str(response["identity"])
        role = roles[identity]
        if role["role"] != "exploration":
            raise GravityItem27Error("confirmation response entered evaluation")
        rows.append({**predictors[identity], **response, "fold": int(role["outer_fold"]), "mass_stratum": int(role["mass_stratum"])})
    quality = {
        "frozen_exploration": int(config["sample"]["expected_exploration"]),
        "valid_primary": len(rows),
        "valid_all_annuli": int(response_manifest["valid_all_annuli"]),
        "minimum_valid_exploration": int(config["sample"]["minimum_valid_exploration"]),
        "formal_quality_pass": int(response_manifest["valid_all_annuli"])
        >= int(config["sample"]["minimum_valid_exploration"]),
        "fold_counts": dict(sorted(Counter(int(row["fold"]) for row in rows).items())),
    }
    return rows, quality


def _evaluate(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    formal_quality_pass: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    xp, backend, device = _backend()
    started = time.perf_counter()
    arrays, candidate_audit = _admissible_candidates(config)
    term_matrix = _build_term_matrix(config, arrays, rows, "primary")
    folds = np.asarray([int(row["fold"]) for row in rows])
    target = _target(rows, "primary")
    observed = _oof_search(
        xp, config, rows, target, folds, term_matrix, "primary"
    )
    residual_evaluations = int(observed["residual_evaluations"])
    candidate_mse = _mse(target, observed["candidate"])
    base_mse = _mse(target, observed["base"])
    flexible_mse = _mse(target, observed["flexible"])
    observed_improvement = _improvement(base_mse, candidate_mse)
    base_full = _base_design(rows, config, "primary")
    base_coefficient = np.linalg.lstsq(base_full, target, rcond=None)[0]
    base_full_prediction = base_full @ base_coefficient
    residual = target - base_full_prediction
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    strata = np.asarray([int(row["mass_stratum"]) for row in rows])
    null_improvements: list[float] = []
    for trial in range(int(config["evaluation"]["permutation_trials"])):
        permuted = residual.copy()
        for stratum in sorted({int(value) for value in strata}):
            indices = np.where(strata == stratum)[0]
            permuted[indices] = random.permutation(permuted[indices])
        null_target = base_full_prediction + permuted
        null = _oof_search(
            xp,
            config,
            rows,
            null_target,
            folds,
            term_matrix,
            "primary",
        )
        null_improvements.append(
            _improvement(
                _mse(null_target, null["base"]),
                _mse(null_target, null["candidate"]),
            )
        )
        residual_evaluations += int(null["residual_evaluations"])
        if (trial + 1) % 10 == 0:
            print(
                f"Item 27 selection-aware nulls {trial + 1}/{config['evaluation']['permutation_trials']}",
                flush=True,
            )
    permutation_p = (
        1.0 + sum(value >= observed_improvement for value in null_improvements)
    ) / (len(null_improvements) + 1.0)

    injections = [
        int(value)
        for value in config["candidate_generator"]["synthetic_injection_admissible_indices"]
    ]
    if len(injections) != 4:
        raise GravityItem27Error("frozen injection boundary changed")
    synthetic: list[dict[str, Any]] = []
    for niche, injection in enumerate(injections):
        if int(arrays["niche"][injection]) != niche:
            raise GravityItem27Error("injection niche changed")
        injected_target = base_full_prediction + term_matrix[injection]
        replay = _oof_search(
            xp,
            config,
            rows,
            injected_target,
            folds,
            term_matrix,
            "primary",
        )
        selected_niches = [int(arrays["niche"][index]) for index in replay["selected"]]
        recovered = int(np.count_nonzero(np.asarray(selected_niches) == niche))
        synthetic.append(
            {
                "injected_niche": niche,
                "injected_index": injection,
                "selected_niches": selected_niches,
                "selected_niche_folds": recovered,
                "pass": recovered >= int(config["gates"]["minimum_same_niche_folds"]),
            }
        )
        residual_evaluations += int(replay["residual_evaluations"])

    no_memory_target = base_full_prediction
    no_memory = _oof_search(
        xp,
        config,
        rows,
        no_memory_target,
        folds,
        term_matrix,
        "primary",
    )
    no_memory_improvement = _improvement(
        _mse(no_memory_target, no_memory["base"]),
        _mse(no_memory_target, no_memory["candidate"]),
    )
    no_memory_pass = no_memory_improvement <= float(
        config["gates"]["known_no_memory_control_maximum_material_improvement"]
    )
    residual_evaluations += int(no_memory["residual_evaluations"])

    replays: dict[str, Any] = {}
    selected_by_fold = {
        outer: index
        for outer, index in zip(
            sorted({int(value) for value in folds}), observed["selected"], strict=True
        )
    }
    for label in ("inner", "outer"):
        replay_rows = [
            row for row in rows if math.isfinite(float(row[f"{label}_vrms_km_s"]))
        ]
        replay_folds = np.asarray([int(row["fold"]) for row in replay_rows])
        if len(replay_rows) < 20 or len(set(replay_folds.tolist())) < int(
            config["sample"]["outer_folds"]
        ):
            replays[label] = {
                "status": "INCONCLUSIVE_QUALITY",
                "objects": len(replay_rows),
                "folds_present": sorted(set(replay_folds.tolist())),
                "candidate_mse": None,
                "instantaneous_baryonic_mse": None,
                "flexible_nuisance_mse": None,
                "improvement_vs_instantaneous_baryonic": None,
                "improvement_vs_flexible_nuisance": None,
            }
            continue
        replay_target = _target(replay_rows, label)
        replay_terms = _build_term_matrix(config, arrays, replay_rows, label)
        replay_selected = [selected_by_fold[int(value)] for value in sorted(set(replay_folds.tolist()))]
        prediction = _fixed_oof(
            xp,
            config,
            replay_rows,
            replay_target,
            replay_folds,
            replay_terms,
            replay_selected,
            label,
        )
        replay_candidate = _mse(replay_target, prediction["candidate"])
        replay_base = _mse(replay_target, prediction["base"])
        replay_flexible = _mse(replay_target, prediction["flexible"])
        replays[label] = {
            "status": "DIAGNOSTIC",
            "objects": len(replay_rows),
            "folds_present": sorted(set(replay_folds.tolist())),
            "candidate_mse": replay_candidate,
            "instantaneous_baryonic_mse": replay_base,
            "flexible_nuisance_mse": replay_flexible,
            "improvement_vs_instantaneous_baryonic": _improvement(replay_base, replay_candidate),
            "improvement_vs_flexible_nuisance": _improvement(replay_flexible, replay_candidate),
        }

    mass = np.asarray([float(row["log_stellar_mass"]) for row in rows])
    history_age = np.asarray([float(row["fossil_mean_age_gyr"]) for row in rows])
    slices = {
        "low_stellar_mass": np.where(mass <= np.median(mass))[0],
        "high_stellar_mass": np.where(mass > np.median(mass))[0],
        "younger_fossil_history": np.where(history_age <= np.median(history_age))[0],
        "older_fossil_history": np.where(history_age > np.median(history_age))[0],
    }
    slice_metrics: dict[str, Any] = {}
    for label, indices in slices.items():
        value_candidate = _mse(target, observed["candidate"], indices)
        value_base = _mse(target, observed["base"], indices)
        value_flexible = _mse(target, observed["flexible"], indices)
        slice_metrics[label] = {
            "objects": len(indices),
            "candidate_mse": value_candidate,
            "instantaneous_baryonic_mse": value_base,
            "flexible_nuisance_mse": value_flexible,
            "improvement_vs_instantaneous_baryonic": _improvement(value_base, value_candidate),
            "improvement_vs_flexible_nuisance": _improvement(value_flexible, value_candidate),
        }
    selected_records = [
        _candidate_record(config, arrays, int(index)) for index in observed["selected"]
    ]
    selected_niches = [int(record["niche"]) for record in selected_records]
    niche_counts = Counter(selected_niches)
    same_niche_folds = max(niche_counts.values())
    counterexamples = int(
        np.count_nonzero(
            (target - observed["candidate"]) ** 2
            > (target - observed["flexible"]) ** 2
        )
    )
    mass_labels = ("low_stellar_mass", "high_stellar_mass")
    history_labels = ("younger_fossil_history", "older_fossil_history")
    universal_pass = all(
        [
            observed_improvement >= float(config["gates"]["minimum_improvement_vs_instantaneous_baryonic"]),
            _improvement(flexible_mse, candidate_mse) >= float(config["gates"]["minimum_improvement_vs_flexible_nuisance"]),
            all(
                replays[label]["improvement_vs_instantaneous_baryonic"] is not None
                and replays[label]["improvement_vs_instantaneous_baryonic"]
                >= float(config["gates"]["minimum_each_radial_replay_improvement_vs_instantaneous"])
                for label in replays
            ),
            all(
                slice_metrics[label]["improvement_vs_instantaneous_baryonic"]
                >= float(config["gates"]["minimum_each_mass_half_improvement_vs_instantaneous"])
                for label in mass_labels
            ),
            all(
                slice_metrics[label]["improvement_vs_instantaneous_baryonic"]
                >= float(config["gates"]["minimum_each_history_half_improvement_vs_instantaneous"])
                for label in history_labels
            ),
            permutation_p <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
            same_niche_folds >= int(config["gates"]["minimum_same_niche_folds"]),
            all(value["pass"] for value in synthetic),
            no_memory_pass,
            formal_quality_pass,
        ]
    )
    phenomenon_pass = all(
        [
            _improvement(flexible_mse, candidate_mse)
            >= float(config["gates"]["phenomenon_minimum_improvement_vs_flexible"]),
            permutation_p <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
            same_niche_folds >= int(config["gates"]["minimum_same_niche_folds"]),
            all(
                slice_metrics[label]["improvement_vs_flexible_nuisance"] >= 0.0
                for label in mass_labels
            ),
            formal_quality_pass,
        ]
    )
    cpu_terms = term_matrix[np.asarray(observed["selected"])]
    gpu_terms = _to_numpy(xp.asarray(cpu_terms), xp)
    cpu_gpu_max = float(np.max(np.abs(cpu_terms - gpu_terms)))
    scientific = {
        "valid_objects": len(rows),
        "formal_quality_pass": formal_quality_pass,
        "candidate_audit": candidate_audit,
        "metrics": {
            "candidate_mse": candidate_mse,
            "instantaneous_baryonic_mse": base_mse,
            "flexible_nuisance_mse": flexible_mse,
            "improvement_vs_instantaneous_baryonic": observed_improvement,
            "improvement_vs_flexible_nuisance": _improvement(flexible_mse, candidate_mse),
            "selection_aware_permutation_p": permutation_p,
            "null_improvement_minimum": float(np.min(null_improvements)),
            "null_improvement_median": float(np.median(null_improvements)),
            "null_improvement_maximum": float(np.max(null_improvements)),
            "individual_counterexamples_vs_flexible": counterexamples,
        },
        "fixed_radial_replays": replays,
        "slice_metrics": slice_metrics,
        "selected_folds": selected_records,
        "selected_niche_counts": {str(key): value for key, value in sorted(niche_counts.items())},
        "same_niche_folds": same_niche_folds,
        "controls": {
            "synthetic_niche_recovery": synthetic,
            "synthetic_all_pass": all(value["pass"] for value in synthetic),
            "no_memory_control_improvement": no_memory_improvement,
            "no_memory_control_pass": no_memory_pass,
            "cpu_gpu_max_absolute_difference": cpu_gpu_max,
            "cpu_gpu_pass": cpu_gpu_max <= 1e-12,
        },
        "universal_gravity_track_pass": universal_pass,
        "phenomenon_publication_track_pass": phenomenon_pass,
        "paper_claim_allowed": False,
        "formal_status": (
            "INCONCLUSIVE_QUALITY"
            if not formal_quality_pass
            else "PASS_EXPLORATION_BOTH_TRACKS"
            if universal_pass and phenomenon_pass
            else "PASS_EXPLORATION_UNIVERSAL_ONLY"
            if universal_pass
            else "PASS_EXPLORATION_PHENOMENON_LEAD"
            if phenomenon_pass
            else "SCOPED_REJECT_BOTH_TRACKS"
        ),
    }
    compute = {
        "schema_version": "invariant-gravity-item27-memory-compute-1.0",
        "backend": backend,
        "device": device,
        "admissible_candidates": len(arrays["niche"]),
        "training_residual_evaluations": residual_evaluations,
        "permutation_trials": len(null_improvements),
        "synthetic_full_searches": 4,
        "no_memory_control_full_searches": 1,
        "wall_seconds": time.perf_counter() - started,
        "paid_model_calls": 0,
        "paid_api_spend_usd": 0.0,
    }
    return scientific, compute


def _build_receipt(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    response_manifest: Mapping[str, Any],
    quality: Mapping[str, Any],
    scientific: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _source_paths(root, config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item27-memory-result-1.0",
            "item": 27,
            "title": config["title"],
            "hypothesis": config["hypothesis"],
            "discovery_policy": config["discovery_policy"],
            "theory_and_equivalence_audit": config["theory"],
            "observable_lineage": config["sources"]["observable_lineage"],
            "frozen_boundary": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "stable_goal_sha256": config["stable_goal_sha256"],
                "implementation_correction_commit": config["implementation_correction_commit"],
                "implementation_correction_scope": config["implementation_correction_scope"],
                "implementation_correction_history": config["implementation_correction_history"],
                "response_access_incident": config["response_access_incident"],
                "confirmation_opened": False,
                "confirmation_response_values_read": int(response_manifest["confirmation_values_read"]),
                "post_response_formula_generation": False,
                "advanced_support_used": False,
            },
            "sample": {
                "quality_audit": quality,
                "valid_exploration_identities": [str(row["identity"]) for row in rows],
                "confirmation_identities_remain_sealed": int(config["sample"]["expected_confirmation"]),
            },
            "baselines": {
                "instantaneous_baryonic": config["evaluation"]["baseline_instantaneous_baryonic"],
                "flexible_nuisance": config["evaluation"]["baseline_flexible_nuisance"],
            },
            "scientific_result": scientific,
            "compute_and_api_cost": compute,
            "counterexamples_and_limitations": config["theory"]["claim_limits"],
            "exact_next_action": "Preserve every Item 27 result under the equal-viability two-track policy, preregister unchanged cross-source replication for any phenomenon lead, and advance the numbered roadmap to Item 28 periodic gravity without privileging age/history.",
            "reproducibility": {
                "config_path": CONFIG_PATH.as_posix(),
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "module_path": MODULE_PATH.as_posix(),
                "module_sha256": _sha256_file(root / MODULE_PATH),
                "predictor_manifest_path": paths["predictor_source_manifest"].relative_to(root).as_posix(),
                "response_manifest_path": paths["response_source_manifest"].relative_to(root).as_posix(),
                "compute_manifest_path": paths["compute_manifest"].relative_to(root).as_posix(),
            },
        }
    )


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    for key in (
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "exploration_responses",
        "response_source_manifest",
    ):
        if not paths[key].exists():
            raise GravityItem27Error(f"missing frozen input: {key}")
    rows, quality = _load_joined_rows(paths, config)
    minimum_diagnostic = 20
    if len(rows) < minimum_diagnostic:
        raise GravityItem27Error(f"only {len(rows)} valid responses; below diagnostic minimum {minimum_diagnostic}")
    scientific, compute = _evaluate(
        config, rows, bool(quality["formal_quality_pass"])
    )
    _write_json(paths["compute_manifest"], _content_hashed(compute))
    response_manifest = _read_json(paths["response_source_manifest"])
    receipt = _build_receipt(
        root, config, rows, response_manifest, quality, scientific, compute
    )
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, receipt)
    return result_path


def validate_result(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    for key in (
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "response_source_manifest",
        "compute_manifest",
    ):
        _verify_content_hash(_read_json(paths[key]), key)
    response_manifest = _read_json(paths["response_source_manifest"])
    if int(response_manifest["confirmation_values_read"]) != 0:
        raise GravityItem27Error("confirmation boundary was opened")
    result = root / str(config["paths"]["result"])
    payload = _read_json(result)
    _verify_content_hash(payload, "result")
    if int(payload["frozen_boundary"]["confirmation_response_values_read"]) != 0:
        raise GravityItem27Error("result contains confirmation response values")
    if int(payload["compute_and_api_cost"]["paid_model_calls"]) != 0:
        raise GravityItem27Error("paid model calls entered Item 27")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-predictors")
    sub.add_parser("prepare-predictors")
    sub.add_parser("acquire-responses")
    sub.add_parser("run")
    sub.add_parser("validate")
    sub.add_parser("show-candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path.cwd()
    if args.command == "audit-predictors":
        print(json.dumps(audit_predictors(root), indent=2, sort_keys=True))
    elif args.command == "prepare-predictors":
        print(prepare_predictors(root)["sample_manifest"].as_posix())
    elif args.command == "acquire-responses":
        print(acquire_responses(root).as_posix())
    elif args.command == "run":
        print(run_experiment(root).as_posix())
    elif args.command == "validate":
        print(validate_result(root).as_posix())
    else:
        print(json.dumps(_candidate_manifest(load_config(root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
