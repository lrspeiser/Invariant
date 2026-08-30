"""No-data proof of concept for an extended-source matter-clock response."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

from .gpu_baryonic_lensing_cluster_screen import GATE_CONFIG, build_lensing_grid

CONFIG_PATH = Path("configs/gravity_extended_source_clock_response_v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_extended_source_clock_response.py")
TEST_PATH = Path("tests/test_gravity_extended_source_clock_response.py")
OUTPUT_PATH = Path("runs/gravity/theory/extended-source-clock-response-v1.json")
CONFIG_SCHEMA = "invariant-gravity-extended-source-clock-response-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-extended-source-clock-response-receipt-1.0"
STATUS = "synthetic_proof_of_concept_no_real_rows"
DECISION = (
    "EXTENDED_SOURCE_CLOCK_RESPONSE_PASSES_FROZEN_SYNTHETIC_GALAXY_LENSING_AND_"
    "CLUSTER_CONTROLS_REAL_DATA_AND_COVARIANT_LIGHT_GRAVITY_COMPLETION_NOT_RUN"
)
EXPECTED_CONFIG_FILE_SHA256 = "5105db6ea5e52e658e71d73f20d6075b05b306a1bbcfc291f0221601b002d19a"
EXPECTED_CONFIG_CONTENT_SHA256 = "47b2d712a9d1d6f7c4756bd5e1b906b445fe6402d1d537e6f15a66f09616b829"
EXPECTED_SECTION_SHA256 = {
    "predecessor_binding": "16d6ab1b20df0d17c1d0232975364295cfa2e7d1854f9b9cfb65533c55fbf27f",
    "clock_hypothesis": "ee2e8cde43a3076d8be4c1f50e18c90fa19789e57a737d4ccef81ebc59ebbcdb",
    "source_geometry_contract": "6d5cb228f961071f7ef31902f230e65209aa745cf47b4a93e5518002d06ed9de",
    "synthetic_control_contract": "d730686d98020575d3f50c67a4435664b5cb9f131da8743b6a897a57fde2fedc",
    "light_and_gravity_contract": "088bf5359f5bb6f1a2df6b722e930754e0e9963e1fef25ff818530928d755ebb",
    "adjudication": "84258cb2ea3ffddedcc60e799c3974df9f7fa7cbe76c6ee3d0192b2cb3e45db2",
    "claim_boundary": "65178ac2a29e786ad993372fff74e9ada409600e6d4fb11057281f92ed443c44",
    "next_test_contract": "56b3cfa357800683c0a39cd7537748bc804378e3eea122769d70665748fe5dd7",
    "zero_access_and_compute": "45c038622428a68e18e9214e30e4fab4463fd150e85e61618257a63b5d2d380d",
}


class ExtendedSourceClockError(RuntimeError):
    """Raised when the frozen clock-response proof of concept changes."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _content_sha(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExtendedSourceClockError(f"could not read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ExtendedSourceClockError("JSON root is not an object")
    return value


def load_config(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    return _load_json(repo / CONFIG_PATH)


def validate_config(config: dict[str, Any], root: Path | None = None) -> None:
    repo = _repo_root() if root is None else root.resolve()
    required = {
        "schema_version",
        "analysis_id",
        "status",
        "purpose",
        "predecessor_binding",
        "clock_hypothesis",
        "source_geometry_contract",
        "synthetic_control_contract",
        "light_and_gravity_contract",
        "adjudication",
        "claim_boundary",
        "next_test_contract",
        "zero_access_and_compute",
        "output_path",
    }
    if set(config) != required:
        raise ExtendedSourceClockError("config keys changed")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["analysis_id"] != "gravity-extended-source-clock-response-v1"
        or config["status"] != "frozen_synthetic_proof_of_concept_no_real_rows"
        or config["output_path"] != OUTPUT_PATH.as_posix()
        or config["adjudication"]["overall_decision"] != DECISION
    ):
        raise ExtendedSourceClockError("config identity changed")
    if _file_sha(repo / CONFIG_PATH) != EXPECTED_CONFIG_FILE_SHA256:
        raise ExtendedSourceClockError("config file hash changed")
    if _content_sha(config) != EXPECTED_CONFIG_CONTENT_SHA256:
        raise ExtendedSourceClockError("config content changed")
    for section, expected in EXPECTED_SECTION_SHA256.items():
        if _content_sha(config[section]) != expected:
            raise ExtendedSourceClockError(f"config section changed: {section}")
    if any(config["zero_access_and_compute"].values()):
        raise ExtendedSourceClockError("zero-access ledger changed")
    expected_truth = {
        "acceleration_only_clock_is_empirical_rar_rewrite": True,
        "extended_source_variable_is_target_independent_in_declared_profiles": True,
        "solar_compact_exterior_limit_passed": True,
        "synthetic_outer_galaxy_gate_passed": True,
        "synthetic_lensing_gate_passed_under_inherited_assumption": True,
        "synthetic_cluster_gate_passed": True,
        "cluster_pass_is_post_hoc_real_data_evidence": False,
        "real_cluster_score_executed": False,
        "real_galaxy_score_executed": False,
        "covariant_clock_theory_derived": False,
        "lensing_theory_derived": False,
        "scientific_claim_allowed": False,
    }
    if any(config["adjudication"][key] is not value for key, value in expected_truth.items()):
        raise ExtendedSourceClockError("adjudication truth values changed")


def validate_predecessor(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    binding = config["predecessor_binding"]
    commit = binding["git_commit"]
    try:
        object_type = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExtendedSourceClockError("predecessor Git object is unavailable") from error
    if object_type != "commit":
        raise ExtendedSourceClockError("predecessor is not a commit")
    for artifact in binding["artifacts"]:
        path = Path(artifact["path"])
        expected = artifact["file_sha256"]
        if _file_sha(repo / path) != expected:
            raise ExtendedSourceClockError("predecessor working bytes changed")
        try:
            committed = subprocess.run(
                ["git", "show", f"{commit}:{path.as_posix()}"],
                cwd=repo,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise ExtendedSourceClockError("predecessor commit path is unavailable") from error
        if _sha_bytes(committed) != expected:
            raise ExtendedSourceClockError("predecessor commit bytes changed")
    receipt = _load_json(repo / Path(binding["receipt_path"]))
    if (
        receipt.get("schema_version") != binding["receipt_schema_version"]
        or receipt.get("decision") != binding["receipt_decision"]
        or receipt.get("content_sha256") != binding["receipt_content_sha256"]
    ):
        raise ExtendedSourceClockError("predecessor receipt identity changed")
    payload = dict(receipt)
    content = payload.pop("content_sha256", None)
    if content != _content_sha(payload):
        raise ExtendedSourceClockError("predecessor receipt content hash is invalid")
    return {
        "git_commit": commit,
        "artifact_count": len(binding["artifacts"]),
        "receipt_content_sha256": content,
        "valid": True,
    }


def _fraction(text: str) -> mp.mpf:
    value = Fraction(text)
    return mp.mpf(value.numerator) / value.denominator


def nu_rar(y: mp.mpf) -> mp.mpf:
    if y <= 0:
        raise ExtendedSourceClockError("y must be positive")
    return 1 / (1 - mp.exp(-mp.sqrt(y)))


def nu_clock(y: mp.mpf, eta: mp.mpf) -> mp.mpf:
    return max(nu_rar(y), 2 * max(mp.mpf(0), eta))


def clock_ratio(y: mp.mpf, eta: mp.mpf) -> mp.mpf:
    return 1 / mp.sqrt(nu_clock(y, eta))


def _disk_shape_gbar(radius: mp.mpf) -> mp.mpf:
    y = radius / 2
    bessel = mp.besseli(0, y) * mp.besselk(0, y) - mp.besseli(1, y) * mp.besselk(1, y)
    return y * bessel


def _build_disk_grid() -> list[dict[str, Any]]:
    masses = ("1/250", "8/125", "128/125")
    radii = (8, 10, 12, 16, 20)
    disks = []
    for mass_text in masses:
        mass = _fraction(mass_text)
        disks.append(
            {
                "mass": mass,
                "mass_text": mass_text,
                "points": [
                    {"radius": radius, "gbar": mass * _disk_shape_gbar(mp.mpf(radius))}
                    for radius in radii
                ],
            }
        )
    return disks


def _disk_equivalent_eta(radius: mp.mpf) -> mp.mpf:
    return 2 + radius * mp.diff(lambda value: mp.log(_disk_shape_gbar(value)), radius)


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    simplified = sp.simplify(residual)
    if simplified != 0:
        raise ExtendedSourceClockError(f"symbolic check failed: {check_id}: {simplified}")
    return {
        "check_id": check_id,
        "statement": statement,
        "residual": str(simplified),
        "passed": True,
    }


def symbolic_checks() -> list[dict[str, Any]]:
    r, rho, m, mprime, y, nu = sp.symbols("r rho M Mprime y nu", positive=True)
    eta = r * mprime / m
    rhobar = 3 * m / (4 * sp.pi * r**3)
    eta_density = 3 * rho / rhobar
    beta_mass = 9 * (r - sp.atan(r))
    beta_rho = 9 / (4 * sp.pi * (1 + r**2))
    beta_eta = r**3 / ((1 + r**2) * (r - sp.atan(r)))
    beta_gbar = beta_mass / r**2
    beta_gdyn = 18 * r / (1 + r**2)
    clock = nu ** sp.Rational(-1, 2)
    hernquist_mass = m * r**2 / (r + 1) ** 2
    hernquist_eta = sp.simplify(r * sp.diff(hernquist_mass, r) / hernquist_mass)
    return [
        _check("S01_MASS_SLOPE_DEFINITION", eta - r * mprime / m, "eta=d ln M/d ln r"),
        _check(
            "S02_DENSITY_CONTRAST_IDENTITY",
            eta_density.subs(rho, mprime / (4 * sp.pi * r**2)) - eta,
            "eta=3 rho/rho_bar",
        ),
        _check(
            "S03_BETA_MASS_DERIVATIVE",
            sp.diff(beta_mass, r) - 4 * sp.pi * r**2 * beta_rho,
            "beta mass derivative",
        ),
        _check(
            "S04_BETA_ETA", r * sp.diff(beta_mass, r) / beta_mass - beta_eta, "beta eta closed form"
        ),
        _check("S05_BETA_GBAR", beta_mass / r**2 - beta_gbar, "beta baryonic field"),
        _check(
            "S06_BETA_RATIO", beta_gdyn / beta_gbar - 2 * beta_eta, "hydrostatic ratio equals 2 eta"
        ),
        _check("S07_CLOCK_MAPPING", nu * clock**2 - 1, "g_pred=g_b/C^2"),
        _check("S08_HERNQUIST_ETA", hernquist_eta - 2 / (r + 1), "Hernquist mass slope"),
        _check(
            "S09_COMPACT_EXTERIOR_ETA",
            (r * sp.diff(m, r) / m),
            "constant exterior mass has eta zero",
        ),
        _check(
            "S10_RAR_REWRITE",
            y / clock.subs(nu, nu) ** 2 - y * nu,
            "clock response is acceleration multiplier",
        ),
    ]


def evaluate_synthetic_controls() -> dict[str, Any]:
    mp.mp.dps = 50
    disks = _build_disk_grid()
    disk_spreads: list[mp.mpf] = []
    disk_means: list[mp.mpf] = []
    disk_eta_rows: list[dict[str, str]] = []
    for disk in disks:
        speeds: list[mp.mpf] = []
        for point in disk["points"]:
            radius = mp.mpf(point["radius"])
            eta = _disk_equivalent_eta(radius)
            y = mp.mpf(point["gbar"])
            if eta >= 0:
                raise ExtendedSourceClockError("outer disk eta no longer selects the RAR channel")
            speeds.append(mp.sqrt(y * nu_clock(y, eta) * radius))
            disk_eta_rows.append(
                {"radius": str(point["radius"]), "eta": mp.nstr(eta, 30), "selected_channel": "RAR"}
            )
        mean = sum(speeds) / len(speeds)
        disk_means.append(mean)
        disk_spreads.append((max(speeds) - min(speeds)) / mean)
    btfr = mp.log(disks[-1]["mass"] / disks[0]["mass"]) / mp.log(disk_means[-1] / disk_means[0])
    frozen_disk_thresholds = {"flatness": mp.mpf("0.06"), "btfr_slope": mp.mpf("0.3")}
    disk_pass = (
        max(disk_spreads) <= frozen_disk_thresholds["flatness"]
        and abs(btfr - 4) <= frozen_disk_thresholds["btfr_slope"]
    )

    lensing_grid = build_lensing_grid()
    vflat: list[mp.mpf] = []
    for disk in disks:
        speeds = [
            mp.sqrt(mp.mpf(point["gbar"]) * nu_rar(mp.mpf(point["gbar"])) * point["radius"])
            for point in disk["points"]
        ]
        vflat.append(sum(speeds) / len(speeds))
    by_mass: dict[int, list[mp.mpf]] = {}
    max_hernquist_channel_ratio = mp.mpf(0)
    for integral in lensing_grid["integrals"]:
        mass = _fraction(integral["mass_text"])
        alpha = mp.mpf(0)
        for node in integral["nodes"]:
            y = mp.mpf(node["y"])
            radius = mp.sqrt(mass / y) - 1
            eta = 2 / (radius + 1)
            if 2 * eta >= nu_rar(y):
                raise ExtendedSourceClockError("Hernquist lensing node no longer selects RAR")
            max_hernquist_channel_ratio = max(max_hernquist_channel_ratio, 2 * eta / nu_rar(y))
            alpha += mp.mpf(node["weight"]) * nu_clock(y, eta)
        by_mass.setdefault(integral["mass_index"], []).append(alpha)
    worst_flatness = mp.mpf(0)
    worst_consistency = mp.mpf(0)
    for mass_index, alphas in by_mass.items():
        mean = sum(alphas) / len(alphas)
        worst_flatness = max(worst_flatness, (max(alphas) - min(alphas)) / mean)
        expected = 2 * mp.pi * vflat[mass_index] ** 2
        worst_consistency = max(worst_consistency, *(abs(alpha / expected - 1) for alpha in alphas))
    lens_thresholds = GATE_CONFIG["lensing"]["fp64_thresholds"]
    lensing_pass = worst_flatness <= mp.mpf(
        lens_thresholds["flatness"]
    ) and worst_consistency <= mp.mpf(lens_thresholds["consistency"])

    cluster = GATE_CONFIG["cluster"]
    cluster_rows: list[dict[str, Any]] = []
    cluster_deviations: list[mp.mpf] = []
    for radius_text, y_text, gdyn_text in zip(
        cluster["probe_radii"], cluster["gbar_50dps"], cluster["gdyn_exact"], strict=True
    ):
        radius = _fraction(radius_text)
        y = mp.mpf(y_text)
        gdyn = _fraction(gdyn_text)
        eta = radius**3 / ((1 + radius**2) * (radius - mp.atan(radius)))
        rar = nu_rar(y)
        extended = 2 * eta
        if extended <= rar:
            raise ExtendedSourceClockError("cluster probe no longer selects extended channel")
        ratio = y * nu_clock(y, eta) / gdyn
        deviation = abs(ratio - 1)
        cluster_deviations.append(deviation)
        cluster_rows.append(
            {
                "radius": radius_text,
                "y": y_text,
                "eta": mp.nstr(eta, 30),
                "nu_rar": mp.nstr(rar, 30),
                "nu_extended": mp.nstr(extended, 30),
                "g_pred_over_gdyn": mp.nstr(ratio, 30),
                "selected_channel": "extended_source",
            }
        )
    cluster_tolerance = mp.mpf(cluster["fp64_thresholds"]["consistency"])
    cluster_pass = max(cluster_deviations) <= cluster_tolerance

    solar_rows = []
    for y in (mp.mpf("1e4"), mp.mpf("1e8")):
        nu_value = nu_clock(y, mp.mpf(0))
        solar_rows.append(
            {
                "y": mp.nstr(y, 10),
                "eta": "0",
                "nu_minus_one": mp.nstr(nu_value - 1, 30),
                "clock_ratio_minus_one": mp.nstr(clock_ratio(y, mp.mpf(0)) - 1, 30),
            }
        )
    solar_pass = all(abs(mp.mpf(row["nu_minus_one"])) < mp.mpf("1e-20") for row in solar_rows)
    if not all((disk_pass, lensing_pass, cluster_pass, solar_pass)):
        raise ExtendedSourceClockError("one or more frozen synthetic controls failed")
    return {
        "solar_compact_exterior": {"passes": solar_pass, "rows": solar_rows},
        "outer_galaxy": {
            "passes": disk_pass,
            "max_speed_fractional_spread": mp.nstr(max(disk_spreads), 30),
            "btfr_slope": mp.nstr(btfr, 30),
            "eta_rows": disk_eta_rows,
            "selected_channel": "RAR",
        },
        "lensing": {
            "passes": lensing_pass,
            "assumption_only": True,
            "worst_flatness": mp.nstr(worst_flatness, 30),
            "worst_consistency": mp.nstr(worst_consistency, 30),
            "max_extended_to_rar_channel_ratio": mp.nstr(max_hernquist_channel_ratio, 30),
            "selected_channel": "RAR",
        },
        "cluster": {
            "passes": cluster_pass,
            "max_fractional_deviation": mp.nstr(max(cluster_deviations), 30),
            "rows": cluster_rows,
            "identity": "g_dyn/g_b=2 eta on the frozen beta-model control",
        },
    }


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    config = load_config(repo)
    validate_config(config, repo)
    predecessor = validate_predecessor(config, repo)
    symbolic = symbolic_checks()
    controls = evaluate_synthetic_controls()
    result = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": STATUS,
        "decision": DECISION,
        "bindings": {
            "config_path": CONFIG_PATH.as_posix(),
            "config_file_sha256": _file_sha(repo / CONFIG_PATH),
            "config_content_sha256": _content_sha(config),
            "implementation_path": SOURCE_PATH.as_posix(),
            "implementation_file_sha256": _file_sha(repo / SOURCE_PATH),
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": _file_sha(repo / TEST_PATH),
            "predecessor": predecessor,
        },
        "clock_hypothesis": config["clock_hypothesis"],
        "symbolic_checks": symbolic,
        "synthetic_controls": controls,
        "counts": {
            "symbolic_checks": len(symbolic),
            "solar_probes": len(controls["solar_compact_exterior"]["rows"]),
            "outer_galaxy_probe_rows": len(controls["outer_galaxy"]["eta_rows"]),
            "lensing_profile_masses": len(GATE_CONFIG["lensing"]["masses"]),
            "cluster_probes": len(controls["cluster"]["rows"]),
            "real_rows": 0,
        },
        "light_and_gravity_contract": config["light_and_gravity_contract"],
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "next_test_contract": config["next_test_contract"],
        "zero_access_and_compute": config["zero_access_and_compute"],
    }
    result["content_sha256"] = _content_sha(result)
    return result


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise ExtendedSourceClockError("refusing to overwrite existing different bytes")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() == payload:
                return "EXISTING_IDENTICAL"
            raise ExtendedSourceClockError("concurrent writer published different bytes") from None
        return "CREATED"
    finally:
        temp.unlink(missing_ok=True)


def write_receipt(root: Path | None = None) -> str:
    repo = _repo_root() if root is None else root.resolve()
    payload = _canonical(build_receipt(repo))
    return _atomic_no_clobber(repo / OUTPUT_PATH, payload)


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    stored = _load_json(repo / OUTPUT_PATH)
    rebuilt = build_receipt(repo)
    if stored != rebuilt:
        raise ExtendedSourceClockError("stored receipt differs from exact rebuild")
    payload = dict(stored)
    content = payload.pop("content_sha256", None)
    if content != _content_sha(payload):
        raise ExtendedSourceClockError("stored receipt content hash is invalid")
    return stored


def status(root: Path | None = None) -> dict[str, Any]:
    receipt = check_receipt(root)
    controls = receipt["synthetic_controls"]
    return {
        "valid": True,
        "decision": receipt["decision"],
        "solar_limit": controls["solar_compact_exterior"]["passes"],
        "outer_galaxy": controls["outer_galaxy"]["passes"],
        "lensing_assumption_only": controls["lensing"]["passes"],
        "cluster_synthetic": controls["cluster"]["passes"],
        "real_rows": receipt["counts"]["real_rows"],
        "real_cluster_score": receipt["adjudication"]["real_cluster_score_executed"],
        "covariant_theory": receipt["adjudication"]["covariant_clock_theory_derived"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args()
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        print(json.dumps(check_receipt(), sort_keys=True))
    else:
        print(json.dumps(status(), sort_keys=True))


if __name__ == "__main__":
    main()
