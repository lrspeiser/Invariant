"""Bind one observed galaxy source to an isolated, conditional axisymmetric field."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import simpson
from scipy.special import iv, kv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from invariant_gravity_extensions.isolated_axisymmetric import (
    MassComponent,
    MultipoleGrid,
    solve_isolated,
)
from invariant_gravity_extensions.reconstructed_axisymmetric import (
    ReconstructedNewtonianSource,
    SurfaceDensityDisk,
    multipole_fields,
)
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def controls(config):
    probes = np.array([.5, 1, 2, 4.])
    records = []
    for a, b in [(0, 1), (1, .3)]:
        exact = MassComponent("analytic", 1, a, b)
        grid = MultipoleGrid(1e-4, 1e3, 1025, 160, 48, plane_scale=b)
        source = ReconstructedNewtonianSource.build("numerical", lambda R, z, c=exact: c.fields(R, z)["laplacian"], grid)
        actual, truth = multipole_fields(source.potential, probes, .2*probes), exact.fields(probes, .2*probes)
        spec = SaturatedActionSpec("qumond", shape=1)
        predicted = solve_isolated((source,), spec, .1, grid).evaluate(probes, .2*probes)["acceleration"]
        reference = solve_isolated((exact,), spec, .1, grid).evaluate(probes, .2*probes)["acceleration"]
        records.append({"a": a, "b": b,
                        "force_relative_RMS_error": float(np.linalg.norm(actual["gradient"]-truth["gradient"])/np.linalg.norm(truth["gradient"])),
                        "hessian_relative_RMS_error": float(np.linalg.norm(actual["hessian"]-truth["hessian"])/np.linalg.norm(truth["hessian"])),
                        "qumond_reconstructed_vs_analytic_source_relative_RMS": float(np.linalg.norm(predicted-reference)/np.linalg.norm(reference))})
    r = np.array([.5, 1, 2, 3, 5.])
    y = r/2
    expected = np.pi*r*(iv(0, y)*kv(0, y)-iv(1, y)*kv(1, y))
    thin = []
    for height in [.1, .02]:
        grid = MultipoleGrid(1e-4, 1e3, 1537, 384, 192, plane_scale=height)
        disk = SurfaceDensityDisk(np.geomspace(1e-5, 30, 3001), np.exp(-np.geomspace(1e-5, 30, 3001)), height, 30, 1)
        source = ReconstructedNewtonianSource.build("exponential", lambda R, z, d=disk: 4*np.pi*d.density(R, z), grid)
        force = source.fields(r, 0)["gradient"][0]
        thin.append({"height_over_Rd": height, "force": force, "relative_error_to_zero_thickness": force/expected-1})
    gates = config["independent_controls"]
    passed = (all(row["force_relative_RMS_error"] < gates["force_relative_RMS_max"] and
                  row["hessian_relative_RMS_error"] < gates["hessian_relative_RMS_max"] and
                  row["qumond_reconstructed_vs_analytic_source_relative_RMS"] < gates["field_source_refinement_relative_force_max"]
                  for row in records) and np.max(abs(thin[-1]["relative_error_to_zero_thickness"])) < .03 and
              np.all(thin[-1]["force"] > thin[0]["force"]))
    return {"analytic_density_controls": records, "Freeman_radii_over_Rd": r, "Freeman_thin_force": expected,
            "finite_thickness_controls": thin, "passes": bool(passed)}


def source_maps(config, checkout, builder):
    if config["object_id"] != "NGC3198":
        raise PermissionError("only the registered NGC3198 development source is admitted")
    split = json.loads((ROOT/config["split_contract"]).read_bytes())
    photometry = json.loads((ROOT/config["photometry_exploration_contract"]).read_bytes())
    if split["assignment"].get("NGC3198") != "train" or not any(g["galaxy"] == "NGC3198" for g in photometry["galaxies"]):
        raise PermissionError("development membership changed")
    geometry = next(g for g in json.loads((ROOT/config["geometry_contract"]).read_bytes())["objects"] if g["object_id"] == "NGC3198")
    for key in ["distance_mpc", "ra_deg", "dec_deg", "outer_ellipticity"]:
        if config["geometry"][key] != geometry[key]:
            raise ValueError("photometric geometry binding changed")
    if config["geometry"]["position_angle_deg"] != geometry["outer_position_angle_deg"]:
        raise ValueError("photometric position angle binding changed")
    receipt = json.loads((ROOT/config["source_receipt"]).read_bytes())
    records = [r for r in receipt["inventory"]["records"] if r["object_id"] == config["object_id"]]
    if len(records) != 7 or {r["role"] for r in records} != set(config["source_roles"]):
        raise ValueError("exact seven-file source inventory required")
    images, accesses = {}, []
    for record in records:
        path = (ROOT/record["relative_path"]).resolve()
        allowed = (ROOT/"work/private/open-gravity-rg-12gal-source-only-v1").resolve()
        if path.parent != allowed or not path.name.startswith(config["object_id"]+"__"):
            raise ValueError("source cache path outside registered scope")
        if not path.exists():
            original = (checkout/record["relative_path"]).resolve()
            if not original.is_relative_to(checkout.resolve()):
                raise ValueError("source checkout path escaped root")
            payload = original.read_bytes()
            if sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError("source payload hash mismatch")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
        if sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("cache source hash mismatch")
        images[record["role"]] = builder._fits_image(path)
        accesses.append({k: record[k] for k in ["object_id", "relative_path", "role", "sha256", "url", "bunit"]})
    names = {"STELLAR_FLUX": "STELLAR_MASS_MAP", "STELLAR_ICA_MASK": "STELLAR_ICA_MASK", "STELLAR_COLOR": "STELLAR_COLOR_MAP",
             "HI_MOM0_NATURAL_SENSITIVITY": "HI_MOM0_NATURAL", "HI_MOM0_ROBUST_PRIMARY": "HI_MOM0_ROBUST",
             "CO21_BROAD_MOM0": "CO21_MOM0", "CO21_BROAD_EMOM0": "CO21_EMOM0"}
    aliases = {key: images[value] for key, value in names.items()}
    g = config["geometry"]
    metadata = {k: g[k] for k in ["distance_mpc", "ra_deg", "dec_deg", "position_angle_deg"]}
    metadata["inclination_deg"] = math.degrees(math.acos(math.sqrt(((1-g["outer_ellipticity"])**2-g["intrinsic_q0"]**2)/(1-g["intrinsic_q0"]**2))))
    legacy = json.loads((ROOT/config["source_builder_contract"]).read_bytes())
    profiles = []
    for n in [config["map"]["coarse_pixels"], config["map"]["pixels"]]:
        print(f"Observed map projection: {n}x{n}", flush=True)
        maps = builder._surface_maps(legacy, metadata, aliases, n=n, box_kpc=config["map"]["box_kpc"],
                                     beam=config["map"]["primary_hi"], use_sip=False)
        radial = np.hypot(maps["x_pc"], maps["y_pc"])/1000
        width = config["map"]["annulus_width_kpc"]
        bins = np.arange(0, config["map"]["box_kpc"]/2+width/2, width)
        index = np.searchsorted(bins, radial.ravel(), side="right")-1
        inside = (index >= 0) & (index < len(bins)-1)
        ids = index[inside]
        area = np.pi*np.diff(bins**2)*1e6
        rows = {}
        angle = np.arctan2(maps["y_pc"], maps["x_pc"]).ravel()[inside]
        for key in ["stellar_fixed", "hi", "co"]:
            values = maps[key].ravel()[inside]
            mass = np.bincount(ids, weights=values*maps["dx_pc"]**2, minlength=len(area))
            modes = []
            for order in [1, 2, 3, 4]:
                c = np.bincount(ids, weights=values*np.cos(order*angle), minlength=len(area))
                s = np.bincount(ids, weights=values*np.sin(order*angle), minlength=len(area))
                total = np.bincount(ids, weights=values, minlength=len(area))
                modes.append(np.divide(np.hypot(c, s), total, out=np.zeros_like(total), where=total > 0))
            rows[key] = {"surface_density_msun_pc2": mass/area, "annular_mass_msun": mass, "relative_azimuthal_modes_m1_m4": modes}
        _, _, ra, dec, _ = builder._disk_grid(metadata, n, config["map"]["box_kpc"])
        coverage = {}
        for key in ["STELLAR_MASS_MAP", "HI_MOM0_ROBUST", "CO21_MOM0"]:
            raw, header = images[key]
            sampled = builder._sample_image(np.isfinite(raw).astype(float), header, ra, dec, use_sip=False, order=0)
            valid = (np.isfinite(sampled) & (sampled > .5)).ravel()[inside]
            numerator = np.bincount(ids, weights=valid, minlength=len(area))
            denominator = np.bincount(ids, minlength=len(area))
            coverage[key] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        half = builder._half_mass_radius_pc(maps["stellar_fixed"], maps["x_pc"], maps["y_pc"], maps["dx_pc"])
        profiles.append({"pixels": n, "radius_kpc": (bins[:-1]+bins[1:])/2, "components": rows,
                         "original_finite_map_coverage_by_annulus": coverage, "stellar_half_mass_radius_kpc": half/1000,
                         "HI_selected_major_beam_kpc": maps["target_fwhm_pc"]/1000})
    return {"accesses": accesses, "metadata": metadata, "profiles": profiles,
            "new_velocity_values_used": 0, "source_selected_after_velocity_residual": False,
            "historical_response_container_inspected_for_provenance_this_turn": True}


def disks_from_profile(profile, config, outer):
    hstar = profile["stellar_half_mass_radius_kpc"]/(1.678*7.3)
    return tuple(SurfaceDensityDisk(np.asarray(profile["radius_kpc"]), np.asarray(row["surface_density_msun_pc2"])*1e6,
                                   hstar if key == "stellar_fixed" else .2, outer, config["map"]["outer_taper_width_kpc"])
                 for key, row in profile["components"].items())


def field_bridge(config, maps):
    radii = np.asarray(config["probe_radii_kpc"])
    G = config["units"]["G_kpc_kms2_msun"]
    p = maps["profiles"][-1]
    primary_disks = disks_from_profile(p, config, config["map"]["source_radius_kpc"])
    rows = []
    source_variants = [("primary", primary_disks),
                       ("coarse_map", disks_from_profile(maps["profiles"][0], config, config["map"]["source_radius_kpc"]))]
    source_variants += [(f"aperture_{outer:g}", disks_from_profile(p, config, outer))
                       for outer in config["map"]["source_radius_sensitivities_kpc"]]
    for name, disks in source_variants:
        definitions = config["multipole_grid"]["refinement"] if name == "primary" else config["multipole_grid"]["refinement"][-1:]
        for definition in definitions:
            grid = MultipoleGrid(config["multipole_grid"]["r_min"], config["multipole_grid"]["r_max"],
                                 **definition, plane_scale=min(d.height for d in disks))
            print(f"Numerical Newtonian map bridge: {name}, l_max={grid.l_max}", flush=True)
            source = ReconstructedNewtonianSource.build(name, lambda R, z, ds=disks: 4*np.pi*G*sum(d.density(R, z) for d in ds), grid)
            force = source.fields(radii, 0)["gradient"][0]
            mass_nodes = np.linspace(0, disks[0].outer_radius, 10001)
            mass = [float(2*np.pi*simpson(d.surface(mass_nodes)*mass_nodes, x=mass_nodes)) for d in disks]
            rows.append({"source_variant": name, "grid": asdict(grid), "radii_kpc": radii, "inward_force_kms2_kpc": force,
                         "component_mass_msun": mass, "vertical_heights_kpc": [d.height for d in disks]})
    primary = [r for r in rows if r["source_variant"] == "primary"]
    reference = primary[-1]["inward_force_kms2_kpc"]
    refinement = float(np.max(abs(primary[0]["inward_force_kms2_kpc"]/reference-1)))
    differences = [{"source_variant": r["source_variant"], "relative_force_change": r["inward_force_kms2_kpc"]/reference-1}
                   for r in rows if r["source_variant"] != "primary"]
    return {"rows": rows, "maximum_numerical_relative_force_change": refinement, "source_sensitivities": differences,
            "numerical_target_met": refinement < config["independent_controls"]["field_source_refinement_relative_force_max"],
            "galaxy_response_scored": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = ROOT/"configs/gravity_map_axisymmetric_source_v1.json"
    config = json.loads(config_path.read_bytes())
    paths = [Path(__file__), config_path, *(ROOT/config[k] for k in ["source_receipt", "geometry_contract", "source_builder_contract", "source_builder_module", "split_contract", "photometry_exploration_contract"]),
             *sorted((ROOT/"src/invariant_gravity_extensions").glob("*.py"))]
    before = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in paths:
        snapshot = args.output/"input-snapshots"/path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as handle:
            handle.write(path.read_bytes())

    def write(name, value):
        with (args.output/name).open("x", encoding="utf8", newline="\n") as handle:
            json.dump(serial(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    provenance = {"input_hashes": before, "started_utc": datetime.now(UTC).isoformat(), "python": platform.python_version(),
                  "numpy": np.__version__, "scipy": scipy.__version__,
                  "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "exact_runtime_input_snapshots_preserved": True}
    write("started.json", {"config": config, **provenance})
    try:
        print("Independent analytic and published disk controls", flush=True)
        checks = controls(config)
        write("controls.json", checks)
        module_spec = importlib.util.spec_from_file_location("legacy_map_builder", ROOT/config["source_builder_module"])
        builder = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(builder)
        maps = source_maps(config, args.source_checkout, builder)
        write("source_profiles.json", maps)
        fields = field_bridge(config, maps)
        if before != {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}:
            raise RuntimeError("input changed during run")
        result = {**provenance, "config": config, "controls": checks, "field_bridge": fields,
                  "source_assumptions_resolved": False, "discovery_claim": False, "rotation_responses_scored": 0,
                  "status": "SOURCE_AND_NUMERICAL_BRIDGE_RETAINED" if checks["passes"] and fields["numerical_target_met"] else "NUMERICAL_BRIDGE_UNRESOLVED_RETAINED"}
        write("result.json", result)
        write("receipt.json", {"status": result["status"], "result_sha256": sha256((args.output/"result.json").read_bytes()).hexdigest()})
        print(json.dumps({"controls_pass": checks["passes"], "maximum_numerical_relative_force_change": fields["maximum_numerical_relative_force_change"], "status": result["status"]}))
    except Exception as exc:
        write("failure.json", {"status": "EXECUTION_FAILURE_NOT_PHYSICAL_REJECTION", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
