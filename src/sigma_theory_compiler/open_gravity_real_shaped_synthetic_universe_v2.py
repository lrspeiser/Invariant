"""Source-anchored population contract for the real-shaped synthetic universe v2."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import (
    Availability,
    DataElement,
    DataElementCatalogue,
    DataRole,
    ExperimentRole,
    UncertaintyKind,
    catalogue_from_elements,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

CONFIG_PATH = Path("configs/open_gravity_real_shaped_synthetic_universe_v2.json")
_ROOT = Path(__file__).resolve().parents[2]
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DOMAINS = ("galaxy", "cluster", "solar", "lensing", "clock", "gw", "void", "simulation")
_PREDICTION_FEATURES = {"prediction.vector.acceleration": ("galaxy.synthetic.v2",)}


@dataclass(frozen=True, slots=True)
class SourceAnchor:
    anchor_id: str
    domain: str
    path: str
    sha256: str
    calibration_tier: str

    def __post_init__(self) -> None:
        _identifier(self.anchor_id, "anchor_id")
        _identifier(self.domain, "anchor domain")
        _hash(self.sha256, "anchor sha256")
        if self.calibration_tier != "PUBLIC_SOURCE_ONLY":
            raise SchemaViolation("foundation anchors must be response blind")
        parsed = PurePosixPath(self.path)
        if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
            raise SchemaViolation("anchor path escaped repository")
        if parsed.as_posix() != self.path:
            raise SchemaViolation("anchor path is not canonical POSIX form")


@dataclass(frozen=True, slots=True)
class DomainPopulationSpec:
    domain: str
    experiment_id: str
    required_features: tuple[str, ...]
    response_features: tuple[str, ...]
    population_axes: tuple[str, ...]
    observation_transforms: tuple[str, ...]
    conventional_controls: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.domain, "population domain")
        _identifier(self.experiment_id, "experiment_id")
        for label, values in (
            ("required features", self.required_features),
            ("response features", self.response_features),
            ("population axes", self.population_axes),
            ("observation transforms", self.observation_transforms),
            ("conventional controls", self.conventional_controls),
        ):
            if not values or values != tuple(sorted(set(values))):
                raise SchemaViolation(f"{label} must be nonempty, unique, and sorted")
            for value in values:
                _identifier(value, label)
        if set(self.required_features) & set(self.response_features):
            raise SchemaViolation("formula inputs and scoring responses overlap")


_FEATURE_METADATA: dict[str, tuple[str, int, tuple[int, ...], str, str, str]] = {
    "detector.beam.kernel": (
        "beam response kernel",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "detector",
        "pixels",
    ),
    "detector.psf.kernel": (
        "point-spread kernel",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "detector",
        "pixels",
    ),
    "detector.scalar.calibration": (
        "detector calibration",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "detector",
        "frequency bins",
    ),
    "detector.scalar.clock-cadence": (
        "clock cadence",
        0,
        (0, 0, 1, 0, 0, 0, 0),
        "s",
        "clock",
        "epochs",
    ),
    "detector.scalar.data-quality-mask": (
        "data quality mask",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "detector",
        "epochs",
    ),
    "detector.scalar.psd": (
        "strain power spectral density",
        0,
        (0, 0, 1, 0, 0, 0, 0),
        "Hz^-1",
        "detector",
        "frequency bins",
    ),
    "detector.scalar.station-clock": (
        "station clock bias",
        0,
        (0, 0, 1, 0, 0, 0, 0),
        "s",
        "barycentric",
        "epochs",
    ),
    "detector.tensor.antenna-response": (
        "detector antenna response",
        2,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "celestial",
        "detectors",
    ),
    "environment.scalar.flow-velocity": (
        "peculiar flow velocity",
        0,
        (0, 1, -1, 0, 0, 0, 0),
        "m s^-1",
        "cmb",
        "objects",
    ),
    "environment.scalar.nonthermal-pressure-fraction": (
        "nonthermal pressure fraction",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "cluster",
        "voxels",
    ),
    "environment.scalar.plasma-delay": (
        "plasma propagation delay",
        0,
        (0, 0, 1, 0, 0, 0, 0),
        "s",
        "barycentric",
        "epochs",
    ),
    "environment.scalar.temperature": (
        "environment temperature",
        0,
        (0, 0, 0, 0, 1, 0, 0),
        "K",
        "instrument",
        "epochs",
    ),
    "environment.tensor.line-of-sight-structure": (
        "line-of-sight tidal structure",
        2,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "observer",
        "sky pixels",
    ),
    "environment.vector.external-acceleration": (
        "external acceleration",
        1,
        (0, 1, -2, 0, 0, 0, 0),
        "m s^-2",
        "source",
        "voxels",
    ),
    "geometry.scalar.distance": ("distance", 0, (0, 1, 0, 0, 0, 0, 0), "m", "observer", "objects"),
    "geometry.scalar.void-path-length": (
        "void path length",
        0,
        (0, 1, 0, 0, 0, 0, 0),
        "m",
        "observer",
        "objects",
    ),
    "geometry.vector.disk-normal": (
        "disk normal",
        1,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "source",
        "objects",
    ),
    "geometry.vector.sky-position": (
        "sky direction",
        1,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "icrs",
        "objects",
    ),
    "history.scalar.cooling-rate": (
        "specific cooling rate",
        0,
        (0, 2, -3, 0, 0, 0, 0),
        "m^2 s^-3",
        "simulation",
        "subhalos x epochs",
    ),
    "history.scalar.energy-dissipation": (
        "specific energy dissipation rate",
        0,
        (0, 2, -3, 0, 0, 0, 0),
        "m^2 s^-3",
        "simulation",
        "subhalos x epochs",
    ),
    "history.scalar.shock-mach": (
        "shock Mach number",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "simulation",
        "subhalos x epochs",
    ),
    "history.tensor.source-waveform": (
        "source gravitational waveform",
        2,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "source",
        "time",
    ),
    "history.tree.merger": ("merger tree", 0, (0, 0, 0, 0, 0, 0, 0), "1", "simulation", "nodes"),
    "history.vector.body-position": (
        "body position",
        1,
        (0, 1, 0, 0, 0, 0, 0),
        "m",
        "barycentric",
        "bodies x epochs",
    ),
    "history.vector.body-velocity": (
        "body velocity",
        1,
        (0, 1, -1, 0, 0, 0, 0),
        "m s^-1",
        "barycentric",
        "bodies x epochs",
    ),
    "history.vector.orbit-state": (
        "orbit state",
        1,
        (0, 1, -1, 0, 0, 0, 0),
        "mixed position-velocity",
        "barycentric",
        "epochs",
    ),
    "photon.scalar.frequency": (
        "photon frequency",
        0,
        (0, 0, -1, 0, 0, 0, 0),
        "Hz",
        "observer",
        "channels",
    ),
    "photon.vector.ray-path": (
        "photon ray path",
        1,
        (0, 1, 0, 0, 0, 0, 0),
        "m",
        "observer",
        "path samples",
    ),
    "response.scalar.arc-brightness": (
        "arc surface brightness",
        0,
        (1, 0, -3, 0, 0, 0, 0),
        "W m^-2 sr^-1",
        "detector",
        "pixels",
    ),
    "response.scalar.capture-state": (
        "capture indicator",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "simulation",
        "mergers",
    ),
    "response.scalar.circular-speed": (
        "circular speed",
        0,
        (0, 1, -1, 0, 0, 0, 0),
        "m s^-1",
        "source",
        "radial bins",
    ),
    "response.scalar.clock-residual": (
        "clock residual",
        0,
        (0, 0, 1, 0, 0, 0, 0),
        "s",
        "clock",
        "epochs",
    ),
    "response.scalar.doppler": (
        "Doppler velocity",
        0,
        (0, 1, -1, 0, 0, 0, 0),
        "m s^-1",
        "observer",
        "epochs",
    ),
    "response.scalar.image-delay": (
        "image time delay",
        0,
        (0, 0, 1, 0, 0, 0, 0),
        "s",
        "observer",
        "images",
    ),
    "response.scalar.lensing-shear": (
        "lensing shear",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "observer",
        "sky bins",
    ),
    "response.scalar.range": (
        "ranging distance",
        0,
        (0, 1, 0, 0, 0, 0, 0),
        "m",
        "observer",
        "epochs",
    ),
    "response.scalar.redshift": ("redshift", 0, (0, 0, 0, 0, 0, 0, 0), "1", "observer", "images"),
    "response.scalar.redshift-residual": (
        "redshift residual",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "cmb",
        "objects",
    ),
    "response.tensor.strain": (
        "detector strain",
        2,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "detector",
        "time",
    ),
    "response.vector.astrometry": (
        "astrometric position",
        1,
        (0, 0, 0, 0, 0, 0, 0),
        "rad",
        "icrs",
        "epochs",
    ),
    "response.vector.hydrostatic-acceleration": (
        "hydrostatic acceleration",
        1,
        (0, 1, -2, 0, 0, 0, 0),
        "m s^-2",
        "cluster",
        "radial bins",
    ),
    "response.vector.velocity-field": (
        "velocity field",
        1,
        (0, 1, -1, 0, 0, 0, 0),
        "m s^-1",
        "source",
        "pixels",
    ),
    "selection.scalar.halo-match": (
        "halo match indicator",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "simulation",
        "subhalos",
    ),
    "selection.scalar.sky-mask": (
        "sky selection mask",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "icrs",
        "sky pixels",
    ),
    "source.scalar.atomic-gas-density": (
        "atomic gas density",
        0,
        (1, -3, 0, 0, 0, 0, 0),
        "kg m^-3",
        "source",
        "voxels",
    ),
    "source.scalar.body-mass": (
        "body mass",
        0,
        (1, 0, 0, 0, 0, 0, 0),
        "kg",
        "barycentric",
        "bodies",
    ),
    "source.scalar.density-contrast": (
        "density contrast",
        0,
        (0, 0, 0, 0, 0, 0, 0),
        "1",
        "comoving",
        "voxels",
    ),
    "source.scalar.gas-density": (
        "gas density",
        0,
        (1, -3, 0, 0, 0, 0, 0),
        "kg m^-3",
        "cluster",
        "voxels",
    ),
    "source.scalar.lens-density": (
        "lens mass density",
        0,
        (1, -3, 0, 0, 0, 0, 0),
        "kg m^-3",
        "lens",
        "voxels",
    ),
    "source.scalar.molecular-gas-density": (
        "molecular gas density",
        0,
        (1, -3, 0, 0, 0, 0, 0),
        "kg m^-3",
        "source",
        "voxels",
    ),
    "source.scalar.potential": (
        "gravitational potential",
        0,
        (0, 2, -2, 0, 0, 0, 0),
        "m^2 s^-2",
        "source",
        "epochs",
    ),
    "source.scalar.pressure": (
        "gas pressure",
        0,
        (1, -1, -2, 0, 0, 0, 0),
        "Pa",
        "cluster",
        "voxels",
    ),
    "source.scalar.radius": (
        "galactocentric radius",
        0,
        (0, 1, 0, 0, 0, 0, 0),
        "m",
        "source",
        "radial bins",
    ),
    "source.scalar.stellar-density": (
        "stellar density",
        0,
        (1, -3, 0, 0, 0, 0, 0),
        "kg m^-3",
        "source",
        "voxels",
    ),
    "source.scalar.temperature": (
        "gas temperature",
        0,
        (0, 0, 0, 0, 1, 0, 0),
        "K",
        "cluster",
        "voxels",
    ),
    "source.tensor.quadrupole": (
        "mass quadrupole",
        2,
        (1, 2, 0, 0, 0, 0, 0),
        "kg m^2",
        "barycentric",
        "epochs",
    ),
    "source.tensor.shape": (
        "source second-moment tensor",
        2,
        (0, 2, 0, 0, 0, 0, 0),
        "m^2",
        "source",
        "objects",
    ),
    "source.vector.acceleration": (
        "baryonic acceleration",
        1,
        (0, 1, -2, 0, 0, 0, 0),
        "m s^-2",
        "source",
        "radial bins",
    ),
    "source.vector.stellar-kinematics": (
        "stellar kinematics",
        1,
        (0, 1, -1, 0, 0, 0, 0),
        "m s^-1",
        "source",
        "spatial bins",
    ),
    "prediction.vector.acceleration": (
        "predicted acceleration",
        1,
        (0, 1, -2, 0, 0, 0, 0),
        "m s^-2",
        "source",
        "radial bins",
    ),
}

# Ordered array axes are scientific metadata, not display labels.  Keep them
# explicit so a population generator cannot silently transpose semantically
# different dimensions while preserving shape and byte count.
_FEATURE_AXES: dict[str, tuple[str, ...]] = {
    "detector.beam.kernel": ("pixel_y", "pixel_x"),
    "detector.psf.kernel": ("pixel_y", "pixel_x"),
    "detector.scalar.calibration": ("frequency",),
    "detector.scalar.clock-cadence": ("epoch",),
    "detector.scalar.data-quality-mask": ("epoch",),
    "detector.scalar.psd": ("frequency",),
    "detector.scalar.station-clock": ("epoch",),
    "detector.tensor.antenna-response": ("detector", "tensor_i", "tensor_j"),
    "environment.scalar.flow-velocity": ("object",),
    "environment.scalar.nonthermal-pressure-fraction": ("x", "y", "z"),
    "environment.scalar.plasma-delay": ("epoch",),
    "environment.scalar.temperature": ("epoch",),
    "environment.tensor.line-of-sight-structure": ("sky_pixel", "tensor_i", "tensor_j"),
    "environment.vector.external-acceleration": ("x", "y", "z", "component"),
    "geometry.scalar.distance": ("object",),
    "geometry.scalar.void-path-length": ("object",),
    "geometry.vector.disk-normal": ("object", "component"),
    "geometry.vector.sky-position": ("object", "component"),
    "history.scalar.cooling-rate": ("subhalo", "epoch"),
    "history.scalar.energy-dissipation": ("subhalo", "epoch"),
    "history.scalar.shock-mach": ("subhalo", "epoch"),
    "history.tensor.source-waveform": ("time", "tensor_i", "tensor_j"),
    "history.tree.merger": ("node",),
    "history.vector.body-position": ("body", "epoch", "component"),
    "history.vector.body-velocity": ("body", "epoch", "component"),
    "history.vector.orbit-state": ("epoch", "component"),
    "photon.scalar.frequency": ("channel",),
    "photon.vector.ray-path": ("path_sample", "component"),
    "response.scalar.arc-brightness": ("pixel_y", "pixel_x"),
    "response.scalar.capture-state": ("merger",),
    "response.scalar.circular-speed": ("radial_bin",),
    "response.scalar.clock-residual": ("epoch",),
    "response.scalar.doppler": ("epoch",),
    "response.scalar.image-delay": ("image",),
    "response.scalar.lensing-shear": ("sky_bin",),
    "response.scalar.range": ("epoch",),
    "response.scalar.redshift": ("image",),
    "response.scalar.redshift-residual": ("object",),
    "response.tensor.strain": ("time", "tensor_i", "tensor_j"),
    "response.vector.astrometry": ("epoch", "component"),
    "response.vector.hydrostatic-acceleration": ("radial_bin", "component"),
    "response.vector.velocity-field": ("pixel_y", "pixel_x", "component"),
    "selection.scalar.halo-match": ("subhalo",),
    "selection.scalar.sky-mask": ("sky_pixel",),
    "source.scalar.atomic-gas-density": ("x", "y", "z"),
    "source.scalar.body-mass": ("body",),
    "source.scalar.density-contrast": ("x", "y", "z"),
    "source.scalar.gas-density": ("x", "y", "z"),
    "source.scalar.lens-density": ("x", "y", "z"),
    "source.scalar.molecular-gas-density": ("x", "y", "z"),
    "source.scalar.potential": ("epoch",),
    "source.scalar.pressure": ("x", "y", "z"),
    "source.scalar.radius": ("radial_bin",),
    "source.scalar.stellar-density": ("x", "y", "z"),
    "source.scalar.temperature": ("x", "y", "z"),
    "source.tensor.quadrupole": ("epoch", "tensor_i", "tensor_j"),
    "source.tensor.shape": ("object", "tensor_i", "tensor_j"),
    "source.vector.acceleration": ("radial_bin", "component"),
    "source.vector.stellar-kinematics": ("spatial_bin", "component"),
    "prediction.vector.acceleration": ("radial_bin", "component"),
}


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchemaViolation(f"{label} is not a canonical identifier")
    return value


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SchemaViolation(f"{label} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sorted_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SchemaViolation(f"{label} must be an array")
    result = tuple(str(item) for item in value)
    if result != tuple(sorted(set(result))):
        raise SchemaViolation(f"{label} must be unique and sorted")
    return result


def load_config() -> dict[str, Any]:
    path = (_ROOT / CONFIG_PATH).resolve()
    if not path.is_relative_to(_ROOT):
        raise SchemaViolation("synthetic universe config escaped repository")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaViolation("cannot read synthetic universe config") from exc
    validate_config(config, verify_anchors=True)
    return config


def validate_config(config: Mapping[str, Any], *, verify_anchors: bool) -> None:
    expected = {
        "schema",
        "suite_id",
        "version",
        "status",
        "purpose",
        "claim_class",
        "anchor_policy",
        "source_anchors",
        "domains",
        "output_policy",
    }
    if not isinstance(config, Mapping) or set(config) != expected:
        raise SchemaViolation("synthetic universe config keys changed")
    if config["schema"] != "open-gravity-real-shaped-synthetic-universe-2.0":
        raise SchemaViolation("synthetic universe schema changed")
    if config["claim_class"] != "SYNTHETIC_DIRECTIONAL_SIGNAL":
        raise SchemaViolation("synthetic universe claim class changed")
    policy = config["anchor_policy"]
    if not isinstance(policy, Mapping) or set(policy) != {
        "source_only_calibration_allowed",
        "response_calibration_requires_development_only_label",
        "response_calibrated_dataset_forbidden_for_confirmation",
        "latent_truth_visible_to_formula",
        "scoring_response_visible_to_formula",
    }:
        raise SchemaViolation("anchor policy changed")
    if policy["latent_truth_visible_to_formula"] or policy["scoring_response_visible_to_formula"]:
        raise SchemaViolation("truth or scoring response leakage enabled")
    anchors = tuple(SourceAnchor(**row) for row in config["source_anchors"])
    if tuple(anchor.domain for anchor in anchors) != _DOMAINS:
        raise SchemaViolation("one ordered source anchor is required per domain")
    if verify_anchors:
        for anchor in anchors:
            path = (_ROOT / anchor.path).resolve()
            if not path.is_relative_to(_ROOT) or not path.is_file():
                raise SchemaViolation(f"source anchor missing: {anchor.anchor_id}")
            if _file_sha256(path) != anchor.sha256:
                raise SchemaViolation(f"source anchor changed: {anchor.anchor_id}")
    domains = tuple(_population_from_row(row) for row in config["domains"])
    if tuple(spec.domain for spec in domains) != _DOMAINS:
        raise SchemaViolation("domain population inventory changed")
    all_features = {
        feature
        for spec in domains
        for feature in (*spec.required_features, *spec.response_features)
    }
    registered_inputs_and_responses = set(_FEATURE_METADATA) - set(_PREDICTION_FEATURES)
    if all_features != registered_inputs_and_responses:
        missing = sorted(all_features - registered_inputs_and_responses)
        unused = sorted(registered_inputs_and_responses - all_features)
        raise SchemaViolation(f"feature metadata mismatch; missing={missing}; unused={unused}")
    if set(_FEATURE_AXES) != set(_FEATURE_METADATA):
        raise SchemaViolation("ordered feature-axis metadata does not cover the catalogue")


def _population_from_row(row: Mapping[str, Any]) -> DomainPopulationSpec:
    if set(row) != {
        "domain",
        "experiment_id",
        "required_features",
        "response_features",
        "population_axes",
        "observation_transforms",
        "conventional_controls",
    }:
        raise SchemaViolation("domain population keys changed")
    return DomainPopulationSpec(
        domain=str(row["domain"]),
        experiment_id=str(row["experiment_id"]),
        required_features=_sorted_tuple(row["required_features"], "required features"),
        response_features=_sorted_tuple(row["response_features"], "response features"),
        population_axes=_sorted_tuple(row["population_axes"], "population axes"),
        observation_transforms=_sorted_tuple(
            row["observation_transforms"], "observation transforms"
        ),
        conventional_controls=_sorted_tuple(row["conventional_controls"], "conventional controls"),
    )


def build_catalogue(config: Mapping[str, Any]) -> DataElementCatalogue:
    validate_config(config, verify_anchors=False)
    domains = tuple(_population_from_row(row) for row in config["domains"])
    roles: defaultdict[str, dict[str, DataRole]] = defaultdict(dict)
    provenance: dict[str, set[str]] = defaultdict(set)
    anchor_by_domain = {row["domain"]: row["sha256"] for row in config["source_anchors"]}
    for spec in domains:
        for feature in spec.required_features:
            if feature.startswith("detector."):
                role = DataRole.INSTRUMENT
            elif feature.startswith("selection."):
                role = DataRole.SELECTION_MASK
            else:
                role = DataRole.FORMULA_INPUT
            roles[feature][spec.experiment_id] = role
            provenance[feature].add(anchor_by_domain[spec.domain])
        for feature in spec.response_features:
            roles[feature][spec.experiment_id] = DataRole.SCORING_ONLY_RESPONSE
            provenance[feature].add(anchor_by_domain[spec.domain])
    for feature, experiments in _PREDICTION_FEATURES.items():
        for experiment_id in experiments:
            roles[feature][experiment_id] = DataRole.DERIVED
        provenance[feature].add(canonical_sha256({"prediction_contract": feature}))
    elements: list[DataElement] = []
    for feature in sorted(roles):
        quantity, rank, dimension, unit, frame, support = _FEATURE_METADATA[feature]
        namespace = feature.rsplit(".", 1)[0]
        elements.append(
            DataElement(
                element_id=feature,
                namespace=namespace,
                physical_quantity=quantity,
                tensor_rank=rank,
                si_dimension=dimension,  # type: ignore[arg-type]
                canonical_unit=unit,
                frame=frame,
                support=support,
                axes=_FEATURE_AXES[feature],
                component="total",
                derivation_parents=(),
                uncertainty=(
                    UncertaintyKind.COVARIANCE
                    if feature.startswith(("response.", "detector."))
                    else UncertaintyKind.MIXED
                ),
                availability=(
                    Availability.SYNTHETIC_ONLY
                    if feature.startswith("prediction.")
                    else (
                        Availability.PUBLIC_RESPONSE
                        if feature.startswith("response.")
                        else Availability.PUBLIC_SOURCE
                    )
                ),
                experiment_roles=tuple(
                    ExperimentRole(experiment, role)
                    for experiment, role in sorted(roles[feature].items())
                ),
                provenance_sha256=canonical_sha256(sorted(provenance[feature])),
            )
        )
    for truth_id, quantity in (
        ("truth.scalar.injection-id", "hidden injected formula identity"),
        ("truth.scalar.parameter-vector", "hidden injected parameters"),
    ):
        elements.append(
            DataElement(
                element_id=truth_id,
                namespace="truth.scalar",
                physical_quantity=quantity,
                tensor_rank=0,
                si_dimension=(0, 0, 0, 0, 0, 0, 0),
                canonical_unit="typed hidden value",
                frame="latent",
                support="world",
                axes=("object",) if truth_id.endswith("injection-id") else ("parameter",),
                component="total",
                derivation_parents=(),
                uncertainty=UncertaintyKind.NONE,
                availability=Availability.SYNTHETIC_ONLY,
                experiment_roles=tuple(
                    ExperimentRole(experiment_id, DataRole.LATENT_SYNTHETIC_TRUTH)
                    for experiment_id in sorted(spec.experiment_id for spec in domains)
                ),
                provenance_sha256=canonical_sha256(config["source_anchors"]),
            )
        )
    return catalogue_from_elements(
        "open-gravity-real-shaped-synthetic-elements", "v2.0.0-foundation", elements
    )


def build_foundation_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config, verify_anchors=True)
    catalogue = build_catalogue(config)
    populations = tuple(_population_from_row(row) for row in config["domains"])
    return {
        "schema": "open-gravity-real-shaped-synthetic-foundation-receipt-2.0",
        "suite_id": config["suite_id"],
        "version": config["version"],
        "status": config["status"],
        "claim_class": config["claim_class"],
        "config_sha256": canonical_sha256(config),
        "catalogue_sha256": catalogue.content_sha256,
        "domain_count": len(populations),
        "data_element_count": len(catalogue.elements),
        "source_anchor_count": len(config["source_anchors"]),
        "domain_feature_counts": {
            spec.domain: {
                "formula_inputs": len(spec.required_features),
                "scoring_responses": len(spec.response_features),
                "population_axes": len(spec.population_axes),
                "conventional_controls": len(spec.conventional_controls),
            }
            for spec in populations
        },
        "generators_implemented": False,
        "observation_operator_library_implemented": True,
        "formula_replays_executed": 0,
        "scientific_response_rows_opened": 0,
    }


__all__ = [
    "CONFIG_PATH",
    "DomainPopulationSpec",
    "SourceAnchor",
    "build_catalogue",
    "build_foundation_receipt",
    "load_config",
    "validate_config",
]
