"""Audit and reconstruct development-only X-COP pressure covariance."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

CONFIG_PATH = Path("configs/gravity_cluster_development_covariance_reconstruction_v1.json")
OUTPUT_PATH = Path(
    "runs/gravity/publication-readiness/development-covariance-reconstruction-v1.json"
)
CONFIG_SCHEMA = "invariant-gravity-cluster-development-covariance-reconstruction-1.0"
RECEIPT_SCHEMA = (
    "invariant-gravity-cluster-development-covariance-reconstruction-receipt-1.0"
)
DEVELOPMENT_CLUSTERS = (
    "A1644",
    "A1795",
    "A2142",
    "A2255",
    "A2319",
    "A3266",
    "A85",
    "ZW1215",
)
EXCLUDED_CLUSTERS = ("A2029", "A3158", "A644", "RXC1825")
SOURCE_IDS = (
    "XCOP_ARCHIVE",
    "ITEM59_SOURCE_RECEIPT",
    "ITEM59_PREFLIGHT",
    "ITEM59_CONFIG",
)
PUBLIC_SOURCE_IDS = ("XCOP_OFFICIAL_DATA_RELEASE", "XCOP_THERMODYNAMIC_PRIMARY_PAPER")
COMPONENT_IDS = ("CP5.1", "CP5.2", "CP5.3", "CP5.4", "CP5.5", "CP5.6")
PRESSURE_COLUMNS = ("RW_SZ", "P_SZ", "eP_SZ")


class GravityClusterDevelopmentCovarianceError(RuntimeError):
    """Raised when the development covariance provenance or boundary changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterDevelopmentCovarianceError(f"{label} keys changed")


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise GravityClusterDevelopmentCovarianceError(f"{label} escaped root") from error
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterDevelopmentCovarianceError(f"expected object: {path}")
    return value


def _validate_content_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    expected = body.pop("content_sha256", None)
    actual = _sha(body)
    if expected != actual:
        raise GravityClusterDevelopmentCovarianceError("bound source content hash changed")
    return actual


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
            "contract_id",
            "audit_cutoff",
            "purpose",
            "implementation_binding",
            "public_metadata",
            "local_source_bindings",
            "population_boundary",
            "pressure_covariance_reconstruction",
            "auxiliary_inventory",
            "component_dispositions",
            "claim_boundary",
            "output_path",
        },
        "development covariance contract",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "append_only_development_audit"
        or config["contract_id"]
        != "gravity-cluster-development-covariance-reconstruction-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterDevelopmentCovarianceError("contract identity changed")

    implementation = config["implementation_binding"]
    if implementation != {
        "path": "src/sigma_theory_compiler/gravity_cluster_development_covariance_reconstruction.py",
        "file_sha256": implementation.get("file_sha256"),
    } or len(str(implementation.get("file_sha256", ""))) != 64:
        raise GravityClusterDevelopmentCovarianceError("implementation binding changed")

    public = config["public_metadata"]
    if tuple(row.get("source_id") for row in public) != PUBLIC_SOURCE_IDS:
        raise GravityClusterDevelopmentCovarianceError("public metadata inventory changed")
    for row in public:
        _strict(row, {"source_id", "url", "audited_fact"}, "public metadata")
        if not str(row["url"]).startswith("https://") or not row["audited_fact"]:
            raise GravityClusterDevelopmentCovarianceError("public metadata weakened")

    bindings = config["local_source_bindings"]
    if tuple(row.get("source_id") for row in bindings) != SOURCE_IDS:
        raise GravityClusterDevelopmentCovarianceError("local source inventory changed")
    for row in bindings:
        _strict(
            row,
            {"source_id", "path", "file_sha256", "content_sha256"},
            "local source binding",
        )
        if len(str(row["file_sha256"])) != 64 or (
            row["content_sha256"] is not None
            and len(str(row["content_sha256"])) != 64
        ):
            raise GravityClusterDevelopmentCovarianceError("local source hash changed")

    boundary = config["population_boundary"]
    if boundary != {
        "development_clusters": list(DEVELOPMENT_CLUSTERS),
        "excluded_same_release_confirmation_clusters": list(EXCLUDED_CLUSTERS),
        "only_development_covariance_members_may_be_opened": True,
        "same_release_confirmation_members_opened_by_this_contract": 0,
        "independent_target_rows_allowed": 0,
        "independent_target_rows_opened": 0,
        "paid_model_calls": 0,
        "network_payload_reads": 0,
    }:
        raise GravityClusterDevelopmentCovarianceError("population or access boundary changed")

    pressure = config["pressure_covariance_reconstruction"]
    _strict(
        pressure,
        {
            "covariance_hdu",
            "covariance_radius_field",
            "covariance_profile_field",
            "covariance_error_field",
            "covariance_matrix_field",
            "standalone_pressure_hdu",
            "standalone_radius_field",
            "standalone_profile_field",
            "standalone_error_field",
            "standalone_scale_header",
            "mapping_rule",
            "mapping_reason",
            "radius_ratio_relative_tolerance",
            "profile_ratio_relative_tolerance",
            "symmetry_absolute_tolerance",
            "minimum_correlation_eigenvalue",
            "member_bindings",
        },
        "pressure reconstruction",
    )
    if (
        pressure["covariance_hdu"] != 4
        or pressure["standalone_pressure_hdu"] != 2
        or not pressure["mapping_rule"]
        or not pressure["mapping_reason"]
    ):
        raise GravityClusterDevelopmentCovarianceError("pressure mapping changed")
    members = pressure["member_bindings"]
    if tuple(row.get("cluster") for row in members) != DEVELOPMENT_CLUSTERS:
        raise GravityClusterDevelopmentCovarianceError("pressure member population changed")
    for row in members:
        _strict(
            row,
            {
                "cluster",
                "covariance_member",
                "covariance_member_sha256",
                "standalone_pressure_member",
                "standalone_pressure_sha256",
            },
            "pressure member binding",
        )
        cluster = str(row["cluster"])
        if (
            not str(row["covariance_member"]).startswith(f"{cluster}/")
            or str(row["standalone_pressure_member"]) != f"{cluster}/{cluster}_pressure.fits"
            or len(str(row["covariance_member_sha256"])) != 64
            or len(str(row["standalone_pressure_sha256"])) != 64
        ):
            raise GravityClusterDevelopmentCovarianceError("pressure member binding changed")

    components = config["component_dispositions"]
    if tuple(components) != COMPONENT_IDS:
        raise GravityClusterDevelopmentCovarianceError("CP5 component order changed")
    for component_id, row in components.items():
        _strict(
            row,
            {"status", "reconstructible", "blocker", "smallest_next_action"},
            f"{component_id} disposition",
        )
        if (
            not row["reconstructible"]
            or not row["blocker"]
            or not row["smallest_next_action"]
            or row["status"] in {"PASS", "COMPLETE"}
        ):
            raise GravityClusterDevelopmentCovarianceError("CP5 claim boundary weakened")
    if components["CP5.1"]["status"] != (
        "DEVELOPMENT_RECONSTRUCTION_PILOT_READY_NOT_GATE_COMPLETE"
    ) or any(components[key]["status"] != "BLOCKED" for key in COMPONENT_IDS[1:]):
        raise GravityClusterDevelopmentCovarianceError("CP5 disposition changed")

    claims = config["claim_boundary"]
    if claims != {
        "development_pressure_covariance_reconstruction_supported": True,
        "pressure_covariance_scored": False,
        "temperature_covariance_reconstructed": False,
        "density_covariance_reconstructed": False,
        "shared_calibration_covariance_reconstructed": False,
        "background_beam_psf_covariance_reconstructed": False,
        "cross_instrument_covariance_reconstructed": False,
        "CP5_1_through_CP5_6_complete": False,
        "full_source_covariance_complete": False,
        "independent_replication": False,
        "scientific_result_emitted": False,
    }:
        raise GravityClusterDevelopmentCovarianceError("claim boundary changed")


def _load_local_sources(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    implementation = config["implementation_binding"]
    implementation_path = _under(root, str(implementation["path"]), "implementation")
    if (
        not implementation_path.is_file()
        or _file_sha(implementation_path) != implementation["file_sha256"]
    ):
        raise GravityClusterDevelopmentCovarianceError("bound implementation changed")
    result: dict[str, Any] = {}
    for binding in config["local_source_bindings"]:
        path = _under(root, str(binding["path"]), "local source")
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise GravityClusterDevelopmentCovarianceError(
                f"bound source file changed: {binding['path']}"
            )
        if binding["content_sha256"] is None:
            result[str(binding["source_id"])] = path
        else:
            value = _read_json(path)
            if _validate_content_hash(value) != binding["content_sha256"]:
                raise GravityClusterDevelopmentCovarianceError(
                    f"bound source content changed: {binding['path']}"
                )
            result[str(binding["source_id"])] = value

    item59 = _read_json(result["ITEM59_CONFIG"])
    receipt = result["ITEM59_SOURCE_RECEIPT"]
    preflight = result["ITEM59_PREFLIGHT"]
    archive = result["XCOP_ARCHIVE"]
    if (
        item59["source"]["archive_sha256"] != _file_sha(archive)
        or tuple(item59["population"]["development_clusters_already_exposed"])
        != DEVELOPMENT_CLUSTERS
        or tuple(item59["population"]["independent_confirmation_clusters_sealed_until_freeze"])
        != EXCLUDED_CLUSTERS
        or receipt["archive_sha256"] != item59["source"]["archive_sha256"]
        or preflight["development_clusters"] != 8
        or preflight["confirmation_response_rows_read"] != 0
        or preflight["inferred_total_mass_rows_read"] != 0
    ):
        raise GravityClusterDevelopmentCovarianceError("Item 59 source lineage changed")
    return result


def _tar_bytes(handle: tarfile.TarFile, member: str) -> bytes:
    try:
        info = handle.getmember(member)
    except KeyError as error:
        raise GravityClusterDevelopmentCovarianceError(
            f"covariance member missing: {member}"
        ) from error
    stream = handle.extractfile(info)
    if stream is None:
        raise GravityClusterDevelopmentCovarianceError(f"member is not a file: {member}")
    return stream.read()


def reconstruct_pressure_covariances(
    root: Path,
) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    config = load_config(root)
    sources = _load_local_sources(root, config)
    pressure = config["pressure_covariance_reconstruction"]
    source_receipt = sources["ITEM59_SOURCE_RECEIPT"]
    receipt_hashes = {
        (str(row["cluster"]), str(row["role"])): str(row["sha256"])
        for row in source_receipt["files"]
    }
    raw_root = root / "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw"
    result: dict[str, dict[str, Any]] = {}
    with tarfile.open(sources["XCOP_ARCHIVE"], "r:gz") as archive:
        for binding in pressure["member_bindings"]:
            cluster = str(binding["cluster"])
            if cluster not in DEVELOPMENT_CLUSTERS or cluster in EXCLUDED_CLUSTERS:
                raise GravityClusterDevelopmentCovarianceError(
                    "attempted non-development covariance access"
                )
            member = str(binding["covariance_member"])
            blob = _tar_bytes(archive, member)
            if _bytes_sha(blob) != binding["covariance_member_sha256"]:
                raise GravityClusterDevelopmentCovarianceError(
                    f"covariance member changed: {cluster}"
                )
            with fits.open(io.BytesIO(blob), memmap=False) as covariance_hdus:
                covariance_hdu = covariance_hdus[int(pressure["covariance_hdu"])]
                row = covariance_hdu.data[0]
                names = set(covariance_hdu.columns.names or ())
                required = {
                    pressure["covariance_radius_field"],
                    pressure["covariance_profile_field"],
                    pressure["covariance_error_field"],
                    pressure["covariance_matrix_field"],
                }
                if not required <= names:
                    raise GravityClusterDevelopmentCovarianceError(
                        f"covariance schema changed: {cluster}"
                    )
                radii = np.asarray(row[pressure["covariance_radius_field"]], dtype=float)
                profile = np.asarray(row[pressure["covariance_profile_field"]], dtype=float)
                source_error = np.asarray(row[pressure["covariance_error_field"]], dtype=float)
                covariance = np.asarray(
                    row[pressure["covariance_matrix_field"]], dtype=float
                ).reshape((len(radii), len(radii)))
                metadata = covariance_hdus[1].data[0]

            pressure_path = raw_root / str(binding["standalone_pressure_member"])
            if (
                not pressure_path.is_file()
                or _file_sha(pressure_path) != binding["standalone_pressure_sha256"]
                or receipt_hashes.get((cluster, "pressure"))
                != binding["standalone_pressure_sha256"]
            ):
                raise GravityClusterDevelopmentCovarianceError(
                    f"standalone pressure source changed: {cluster}"
                )
            with fits.open(pressure_path, memmap=False) as pressure_hdus:
                standalone_hdu = pressure_hdus[int(pressure["standalone_pressure_hdu"])]
                if tuple(standalone_hdu.columns.names or ()) != PRESSURE_COLUMNS:
                    raise GravityClusterDevelopmentCovarianceError(
                        f"standalone pressure schema changed: {cluster}"
                    )
                standalone = standalone_hdu.data
                standalone_radii = np.asarray(
                    standalone[pressure["standalone_radius_field"]], dtype=float
                )
                scale = float(standalone_hdu.header[pressure["standalone_scale_header"]])
                standalone_profile = (
                    np.asarray(standalone[pressure["standalone_profile_field"]], dtype=float)
                    * scale
                )
                standalone_error = (
                    np.asarray(standalone[pressure["standalone_error_field"]], dtype=float)
                    * scale
                )

            arrays = (
                radii,
                profile,
                source_error,
                covariance,
                standalone_radii,
                standalone_profile,
                standalone_error,
            )
            if (
                any(not np.all(np.isfinite(value)) for value in arrays)
                or len(radii) != len(standalone_radii)
                or len(radii) < 2
                or np.any(radii <= 0)
                or np.any(profile <= 0)
                or np.any(source_error <= 0)
                or np.any(standalone_radii <= 0)
                or np.any(standalone_profile <= 0)
                or np.any(standalone_error <= 0)
                or np.any(np.diag(covariance) <= 0)
            ):
                raise GravityClusterDevelopmentCovarianceError(
                    f"invalid covariance values: {cluster}"
                )
            symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
            if symmetry_error > float(pressure["symmetry_absolute_tolerance"]):
                raise GravityClusterDevelopmentCovarianceError(
                    f"covariance is not symmetric: {cluster}"
                )
            source_sigma = np.sqrt(np.diag(covariance))
            if not np.allclose(source_sigma, source_error, rtol=1e-10, atol=1e-18):
                raise GravityClusterDevelopmentCovarianceError(
                    f"covariance diagonal changed: {cluster}"
                )
            radius_ratio = radii / standalone_radii
            profile_ratio = profile / standalone_profile
            radius_error = float(
                np.max(np.abs(radius_ratio / np.median(radius_ratio) - 1.0))
            )
            profile_error = float(
                np.max(np.abs(profile_ratio / np.median(profile_ratio) - 1.0))
            )
            if (
                radius_error > float(pressure["radius_ratio_relative_tolerance"])
                or profile_error > float(pressure["profile_ratio_relative_tolerance"])
            ):
                raise GravityClusterDevelopmentCovarianceError(
                    f"covariance/profile bin mapping changed: {cluster}"
                )
            correlation = covariance / np.outer(source_sigma, source_sigma)
            eigenvalues = np.linalg.eigvalsh((correlation + correlation.T) / 2.0)
            if float(eigenvalues[0]) < float(pressure["minimum_correlation_eigenvalue"]):
                raise GravityClusterDevelopmentCovarianceError(
                    f"pressure correlation is not positive semidefinite: {cluster}"
                )
            reconstructed = correlation * np.outer(standalone_error, standalone_error)
            if not np.allclose(
                np.diag(reconstructed), standalone_error**2, rtol=1e-12, atol=0.0
            ):
                raise GravityClusterDevelopmentCovarianceError(
                    f"reconstructed diagonal changed: {cluster}"
                )
            result[cluster] = {
                "matrix": reconstructed,
                "correlation": correlation,
                "covariance_member": member,
                "covariance_member_sha256": _bytes_sha(blob),
                "standalone_pressure_path": pressure_path.relative_to(root).as_posix(),
                "standalone_pressure_sha256": _file_sha(pressure_path),
                "bins": len(radii),
                "source_simulations": int(metadata["NSIM"]),
                "source_monte_carlo_draws": int(metadata["NMC"]),
                "source_beam_arcmin": float(metadata["BEAM"]),
                "radius_ratio_maximum_relative_deviation": radius_error,
                "profile_ratio_maximum_relative_deviation": profile_error,
                "correlation_minimum_eigenvalue": float(eigenvalues[0]),
                "maximum_absolute_offdiagonal_correlation": float(
                    np.max(np.abs(correlation - np.eye(len(correlation))))
                ),
                "correlation_sha256": _array_sha(correlation),
                "reconstructed_covariance_sha256": _array_sha(reconstructed),
            }
    if tuple(result) != DEVELOPMENT_CLUSTERS:
        raise GravityClusterDevelopmentCovarianceError("development reconstruction changed")
    return result


def _auxiliary_archive_audit(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    sources = _load_local_sources(root, config)
    auxiliary = config["auxiliary_inventory"]
    with tarfile.open(sources["XCOP_ARCHIVE"], "r:gz") as archive:
        names = tuple(archive.getnames())
    lower_names = tuple(name.lower() for name in names)
    absent = {
        token: sum(token.lower() in name for name in lower_names)
        for token in auxiliary["absent_archive_filename_tokens"]
    }
    if any(absent.values()):
        raise GravityClusterDevelopmentCovarianceError(
            "previously absent response/calibration asset appeared"
        )
    counts = {
        "background_mosaics": 0,
        "exposure_mosaics": 0,
        "science_mosaics": 0,
        "spectral_fit_summaries": 0,
    }
    for cluster in DEVELOPMENT_CLUSTERS:
        cluster_names = tuple(name for name in names if name.startswith(f"{cluster}/"))
        matches = {
            "background_mosaics": [name for name in cluster_names if "_bkg.fits.gz" in name],
            "exposure_mosaics": [name for name in cluster_names if "_expo.fits.gz" in name],
            "science_mosaics": [
                name
                for name in cluster_names
                if "/mosaic_" in name
                and name.endswith(".fits.gz")
                and "_bkg." not in name
                and "_expo." not in name
                and "_asmooth." not in name
            ],
            "spectral_fit_summaries": [
                name for name in cluster_names if "/spectral_results_" in name
            ],
        }
        if any(len(value) != 1 for value in matches.values()):
            raise GravityClusterDevelopmentCovarianceError(
                f"auxiliary archive inventory changed: {cluster}"
            )
        for key in counts:
            counts[key] += 1
    return {
        "available_counts": counts,
        "absent_filename_token_counts": absent,
        "interpretation": auxiliary["interpretation"],
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    reconstructions = reconstruct_pressure_covariances(root)
    auxiliary = _auxiliary_archive_audit(root, config)
    summaries = []
    for cluster, row in reconstructions.items():
        summaries.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"matrix", "correlation"}
            }
            | {"cluster": cluster, "unit": "(keV cm^-3)^2"}
        )
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "contract_id": config["contract_id"],
        "decision": "DEVELOPMENT_PRESSURE_COVARIANCE_PILOT_RECONSTRUCTIBLE_CP5_REMAINS_PARTIAL",
        "config_binding": {"path": CONFIG_PATH.as_posix(), "content_sha256": _sha(config)},
        "implementation_binding": config["implementation_binding"],
        "public_metadata": config["public_metadata"],
        "local_source_bindings": config["local_source_bindings"],
        "population_boundary": config["population_boundary"],
        "pressure_reconstructions": summaries,
        "auxiliary_archive_audit": auxiliary,
        "advanced_goal_evidence": {
            "CP5.1": "released_planck_pressure_correlation_reconstructible_for_8_already_exposed_development_clusters"
        },
        "completed_goal_evidence": {},
        "component_dispositions": config["component_dispositions"],
        "counts": {
            "development_clusters_reconstructed": len(reconstructions),
            "pressure_covariance_matrices": len(reconstructions),
            "pressure_covariance_bins": sum(row["bins"] for row in reconstructions.values()),
            "same_release_confirmation_members_opened": 0,
            "independent_target_rows_opened": 0,
            "temperature_covariance_matrices": 0,
            "density_covariance_matrices": 0,
            "shared_or_cross_instrument_covariance_matrices": 0,
            "paid_model_calls": 0,
        },
        "claims": config["claim_boundary"],
        "pilot_plan": {
            "population": list(DEVELOPMENT_CLUSTERS),
            "candidate_and_nuisances": "frozen_Item59_values_no_refit",
            "comparison": "existing_diagonal_pressure_score_vs_reconstructed_pressure_covariance_score",
            "reporting": "publish_every_cluster_score_delta_and_matrix_condition_diagnostic",
            "prohibited": [
                "formula_or_nuisance_refit",
                "confirmation_or_independent_member_access",
                "temperature_or_density_offdiagonal_covariance_invention",
                "scientific_promotion_from_this_pilot",
            ],
        },
        "next_action": "Run the no-refit eight-cluster pressure-only scoring pilot, while separately acquiring one complete X-ray response/background/deprojection packet before attempting CP5.2-CP5.6.",
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterDevelopmentCovarianceError("covariance receipt changed")


def write_receipt(root: Path) -> Path:
    path = root.resolve() / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(build_receipt(root)))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        output: Any = str(write_receipt(root))
    elif args.command == "check":
        receipt = _read_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
    else:
        receipt = build_receipt(root)
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
