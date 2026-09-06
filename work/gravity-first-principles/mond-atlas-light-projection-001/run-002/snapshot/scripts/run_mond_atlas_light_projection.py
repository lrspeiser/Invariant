"""Offline, synthetic-only reproducible audit for the light-projection operator."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import roots_legendre

from mond_atlas_light_projection import (
    AngularDistances, C_SI, DISPOSITION, G_SI, KPC_M, MPC_M, MSUN_KG,
    ScalarMetric, SphericalComponent, deflection, lens_jacobian, lens_map,
    manufactured_metric, signed_magnification,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mond_atlas_light_projection_v1.json"
OUTPUT = ROOT / "work/gravity-first-principles/mond-atlas-light-projection-001"
OWNED = [
    "scripts/mond_atlas_light_projection.py",
    "scripts/run_mond_atlas_light_projection.py",
    "configs/mond_atlas_light_projection_v1.json",
    "tests/test_mond_atlas_light_projection.py",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def analytic_deflection(components, xy_m, *, eta=1.0, half_depth_m=None, ell_origin_m=0.0):
    """Independent closed-form integration of analytic component gradients."""
    value = np.zeros(2)
    for comp in components:
        delta = np.asarray(xy_m) - np.asarray(comp.center_m[:2])
        d2 = float(delta @ delta + comp.scale_m**2)
        if d2 == 0:
            raise ValueError("singular analytic point ray")
        endpoint = 2.0
        if half_depth_m is not None:
            lo = ell_origin_m-half_depth_m-comp.center_m[2]
            hi = ell_origin_m+half_depth_m-comp.center_m[2]
            endpoint = hi/np.sqrt(d2+hi*hi) - lo/np.sqrt(d2+lo*lo)
        value += (1+eta)*G_SI*comp.mass_kg/C_SI**2 * delta/d2 * endpoint
    return value


def surface_deflection_kpc(component_specs, ray_kpc, radial_order, angular_order):
    """Independent 2-D Sigma integral; no potential or LOS routine is called.

    Polar coordinates about the evaluation ray cancel the integrable 1/r
    kernel against the area measure. Integrate r=tan(t) kpc, 0<t<pi/2.
    Uniform angular trapezoid and Gauss-Legendre radius form a separate route.
    """
    nodes, weights = roots_legendre(radial_order)
    t = (nodes+1)*np.pi/4
    radius = np.tan(t)
    dr = weights*np.pi/4/np.cos(t)**2
    angle = np.arange(angular_order)*2*np.pi/angular_order
    directions = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    positions = np.asarray(ray_kpc)[None, None, :] + radius[:, None, None]*directions[None, :, :]
    sigma = np.zeros(positions.shape[:2])  # Msun/kpc^2
    for spec in component_specs:
        a2 = spec["scale_kpc"]**2
        if a2 <= 0:
            raise ValueError("surface-density reference requires softened components")
        offset = positions-np.asarray(spec["center_kpc"][:2])
        sigma += spec["mass_Msun"]*a2/(np.pi*(np.sum(offset**2, axis=2)+a2)**2)
    integral = -np.einsum("r,ra,ai->i", dr, sigma, directions)*2*np.pi/angular_order
    return 4*G_SI*MSUN_KG/(C_SI**2*KPC_M)*integral


class Checks:
    def __init__(self):
        self.rows = []

    def compare(self, name, actual, expected, tolerance, *, required=True, mode="relative", **metadata):
        actual_array, expected_array = np.asarray(actual), np.asarray(expected)
        error = float(np.linalg.norm(actual_array-expected_array))
        if mode == "relative":
            denominator = float(np.linalg.norm(expected_array))
            if denominator == 0:
                raise ValueError("zero references must use absolute error explicitly")
            error /= denominator
        elif mode != "absolute":
            raise ValueError("unknown comparison mode")
        finite = bool(np.all(np.isfinite(actual_array)) and np.all(np.isfinite(expected_array)))
        row = dict(id=name, required=required, passed=finite and error <= tolerance,
                   error=error, error_mode=mode, tolerance=tolerance,
                   actual=actual_array.tolist(), expected=expected_array.tolist(), **metadata)
        if any(existing["id"] == name for existing in self.rows):
            raise ValueError(f"duplicate check {name}")
        self.rows.append(row)
        return error

    def truth(self, name, value, **metadata):
        self.rows.append(dict(id=name, required=True, passed=bool(value), **metadata))


def run_benchmarks(cfg, checks):
    numerics, fixture, tol = cfg["numerics"], cfg["fixtures"], cfg["tolerances"]
    default = dict(step_m=numerics["gradient_step_kpc"]*KPC_M,
                   order=numerics["quadrature_order"], scale_m=numerics["line_scale_kpc"]*KPC_M,
                   weak_limit=numerics["weak_potential_fraction_limit"],
                   singular_guard_steps=numerics["singular_ray_stencil_guard_steps"])
    masses = [SphericalComponent(fixture["point_mass_Msun"]*MSUN_KG),
              SphericalComponent(fixture["plummer_mass_Msun"]*MSUN_KG, fixture["plummer_scale_kpc"]*KPC_M)]
    point, plummer = masses
    fields = [manufactured_metric([component], eta=1) for component in masses]
    distance = cfg["distances"]
    geom = AngularDistances(*(distance[key]*MPC_M for key in ["D_l_Mpc", "D_s_Mpc", "D_ls_Mpc"]))

    # This check does not provide distances to the operator. It validates the
    # separately supplied Astropy EdS fixture with an independent closed form.
    chi = lambda z: 2*(C_SI/1000)/distance["H0_km_s_Mpc"]*(1-1/np.sqrt(1+z))
    zl, zs = distance["synthetic_redshift_lens"], distance["synthetic_redshift_source"]
    eds = [chi(zl)/(1+zl), chi(zs)/(1+zs), (chi(zs)-chi(zl))/(1+zs)]
    for key, expected in zip(["D_l_Mpc", "D_s_Mpc", "D_ls_Mpc"], eds):
        checks.compare("distance_"+key, distance[key], expected, tol["distance_relative"])
    checks.truth("distance_not_source_minus_lens", abs(geom.D_ls_m-(geom.D_s_m-geom.D_l_m)) > .1*geom.D_ls_m)

    for label, component, field in zip(["point", "plummer"], masses, fields):
        for radius in fixture["radial_impact_kpc"]:
            for angle in fixture["azimuth_radians"]:
                ray = radius*KPC_M*np.array([np.cos(angle), np.sin(angle)])
                result = deflection(field, ray, **default)
                expected = analytic_deflection([component], ray)
                checks.compare(f"{label}_radial_{radius}_angle_{angle}", result, expected, tol["deflection_relative"])
        # Direct numerical projection of rho and enclosed projected mass uses
        # dimensionless adaptive integrals, independent of the LOS operator.
        if component.scale_m:
            for radius in fixture["radial_impact_kpc"]:
                q = radius*KPC_M/component.scale_m
                rho = lambda z, r: 3/(4*np.pi)*(1+r*r+z*z)**(-2.5)
                sigma_num, sigma_err = quad(lambda z: rho(z, q), -np.inf, np.inf, epsabs=1e-12, epsrel=1e-11)
                sigma_ref = 1/(np.pi*(1+q*q)**2)
                checks.compare(f"projected_density_{radius}", sigma_num, sigma_ref,
                               tol["density_projection_relative"], adaptive_error_bound=sigma_err)
                def radial_integrand(r):
                    projected, _ = quad(lambda z: rho(z, r), -np.inf, np.inf, epsabs=1e-12, epsrel=1e-11)
                    return 2*np.pi*r*projected
                fraction, mass_err = quad(radial_integrand, 0, q, epsabs=1e-11, epsrel=1e-11)
                checks.compare(f"projected_mass_{radius}", fraction, q*q/(1+q*q),
                               tol["density_projection_relative"], adaptive_error_bound=mass_err)
                mass_alpha = 4*G_SI*component.mass_kg*fraction/(C_SI**2*radius*KPC_M)
                checks.compare(f"deflection_from_projected_mass_{radius}", deflection(field, [radius*KPC_M, 0], **default)[0],
                               mass_alpha, tol["deflection_relative"])

    # Constant scalar potential offsets, reflection, mass scaling and length
    # scaling at finite support (infinite stencil tails amplify roundoff).
    ray = KPC_M*np.array([.7, .4])
    for label, component, field in zip(["point", "plummer"], masses, fields):
        baseline = deflection(field, ray, **default)
        checks.compare(f"{label}_reflection", deflection(field, -ray, **default), -baseline, tol["symmetry_relative"])
        for eta in cfg["closure"]["fixed_synthetic_ablation_ratios"]:
            ablated = manufactured_metric([component], eta=eta)
            expected = baseline*(1+eta)/2
            checks.compare(f"{label}_eta_{eta}", deflection(ablated, ray, **default), expected,
                           tol["zero_absolute_radian"] if eta == -1 else tol["deflection_relative"],
                           mode="absolute" if eta == -1 else "relative")
        twice = manufactured_metric([SphericalComponent(2*component.mass_kg, component.scale_m)], eta=1)
        checks.compare(f"{label}_mass_linearity", deflection(twice, ray, **default), 2*baseline, tol["symmetry_relative"])
        scaled = manufactured_metric([SphericalComponent(component.mass_kg, 3*component.scale_m)], eta=1)
        scaled_numerics = dict(default, step_m=3*default["step_m"], scale_m=3*default["scale_m"])
        checks.compare(f"{label}_length_scaling", deflection(scaled, 3*ray, **scaled_numerics), baseline/3, tol["symmetry_relative"])
        shifted = ScalarMetric(lambda x,y,z: field.phi(x,y,z)+1e7,
                               lambda x,y,z: field.psi(x,y,z)-3e7, "CONSTANT_OFFSETS_TEST_ONLY", field.point_centers_m)
        checks.compare(f"{label}_constant_offset", deflection(shifted, ray, half_depth_m=8*KPC_M, **default),
                       deflection(field, ray, half_depth_m=8*KPC_M, **default), tol["symmetry_relative"])
        # Independent kpc/Msun expression checks SI conversion and c^-2 units.
        r_kpc = np.linalg.norm(ray)/KPC_M
        a_kpc = component.scale_m/KPC_M
        G_kpc_km2_s2_Msun = G_SI*MSUN_KG/(KPC_M*1e6)
        alpha_units = 4*G_kpc_km2_s2_Msun*(component.mass_kg/MSUN_KG)*r_kpc/((C_SI/1000)**2*(r_kpc*r_kpc+a_kpc*a_kpc))
        checks.compare(f"{label}_unit_conversion", np.linalg.norm(baseline), alpha_units, tol["deflection_relative"])

    zero = manufactured_metric([], eta=1)
    checks.compare("zero_mass_deflection", deflection(zero, ray, **default), [0,0], tol["zero_absolute_radian"], mode="absolute")
    checks.compare("plummer_central_deflection", deflection(fields[1], [0,0], **default), [0,0], tol["zero_absolute_radian"], mode="absolute")

    # Finite-boundary accuracy is checked against the exact SAME finite domain;
    # tail loss is separately retained against infinity. Chosen ray (1,0) kpc.
    # Sweep diagnostics retain all target failures; only the largest depth is
    # a required tail gate. Domain and gates were declared before execution.
    for label, component, field in zip(["point", "plummer"], masses, fields):
        depth_ray = np.array([KPC_M, 0.0])
        infinite = analytic_deflection([component], depth_ray)
        tails = []
        for depth in numerics["finite_depths_kpc"]:
            result = deflection(field, depth_ray, half_depth_m=depth*KPC_M, **default)
            finite = analytic_deflection([component], depth_ray, half_depth_m=depth*KPC_M)
            checks.compare(f"{label}_finite_integral_{depth}", result, finite, tol["finite_integral_relative"])
            tail = checks.compare(f"{label}_finite_tail_{depth}", result, infinite,
                                  tol["finite_tail_largest_depth_relative"],
                                  required=depth == numerics["finite_depths_kpc"][-1],
                                  expected_underresolved=depth != numerics["finite_depths_kpc"][-1])
            tails.append(tail)
        checks.truth(f"{label}_tail_monotone", all(b<a for a,b in zip(tails, tails[1:])), errors=tails)
        tail_order = float(np.log2(tails[-2]/tails[-1]))
        checks.truth(f"{label}_tail_asymptotic_order", tail_order > 1.9, measured_order=tail_order, minimum=1.9)
        fd_errors = []
        for step in numerics["gradient_steps_kpc"]:
            result = deflection(field, ray, **dict(default, step_m=step*KPC_M))
            err = checks.compare(f"{label}_gradient_step_{step}", result, analytic_deflection([component], ray),
                                 tol["finite_difference_finest_relative"],
                                 required=step == numerics["gradient_steps_kpc"][-1],
                                 expected_underresolved=step != numerics["gradient_steps_kpc"][-1])
            fd_errors.append(err)
        measured_orders = [float(np.log2(a/b)) for a,b in zip(fd_errors, fd_errors[1:])]
        checks.truth(f"{label}_fourth_order_gradient_convergence", min(measured_orders) >= tol["minimum_fd_convergence_order"],
                     measured_orders=measured_orders, minimum=tol["minimum_fd_convergence_order"])
        for radius in [.2, 1.0]:
            qr = KPC_M*np.array([radius, 0.0])
            for order in numerics["quadrature_orders"]:
                checks.compare(f"{label}_quadrature_r_{radius}_order_{order}",
                               deflection(field, qr, **dict(default, order=order)),
                               analytic_deflection([component], qr), tol["quadrature_finest_relative"],
                               required=order == numerics["quadrature_orders"][-1],
                               expected_underresolved=order != numerics["quadrature_orders"][-1])

    specs = fixture["asymmetric_components"]
    components = [SphericalComponent(s["mass_Msun"]*MSUN_KG, s["scale_kpc"]*KPC_M,
                                    tuple(np.asarray(s["center_kpc"])*KPC_M)) for s in specs]
    asymmetric = manufactured_metric(components, eta=1)
    angle = fixture["transformation_rotation_radians"]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    translation = np.asarray(fixture["transformation_translation_kpc"])*KPC_M
    rotated_components = [SphericalComponent(c.mass_kg, c.scale_m, tuple(rotation @ np.array(c.center_m[:2]))+(c.center_m[2],)) for c in components]
    translated_components = [SphericalComponent(c.mass_kg, c.scale_m, tuple(np.array(c.center_m)+translation)) for c in components]
    for index, ray_kpc in enumerate(fixture["asymmetric_rays_kpc"]):
        aray = np.array(ray_kpc)*KPC_M
        result = deflection(asymmetric, aray, **default)
        reference = analytic_deflection(components, aray)
        checks.compare(f"asymmetric_{index}_analytic", result, reference, tol["deflection_relative"])
        surface_values = []
        for factor in [1, 2]:
            surface = surface_deflection_kpc(specs, ray_kpc, fixture["reference_surface_radial_order"]*factor,
                                            fixture["reference_surface_angular_order"]*factor)
            surface_values.append(surface)
            checks.compare(f"asymmetric_{index}_independent_surface_x{factor}", surface, result, tol["surface_integral_relative"])
        checks.compare(f"asymmetric_{index}_surface_convergence", surface_values[0], surface_values[1], tol["surface_integral_relative"])
        checks.compare(f"asymmetric_{index}_rotation", deflection(manufactured_metric(rotated_components, eta=1), rotation@aray, **default),
                       rotation@result, tol["symmetry_relative"])
        checks.compare(f"asymmetric_{index}_translation", deflection(manufactured_metric(translated_components, eta=1), aray+translation[:2],
                       ell_origin_m=translation[2], **default), result, tol["symmetry_relative"])
        finite = deflection(asymmetric, aray, half_depth_m=4*KPC_M, **default)
        checks.compare(f"asymmetric_{index}_finite_boundary", finite, analytic_deflection(components, aray, half_depth_m=4*KPC_M), tol["finite_integral_relative"])
        checks.compare(f"asymmetric_{index}_finite_translation", deflection(manufactured_metric(translated_components, eta=1), aray+translation[:2],
                       half_depth_m=4*KPC_M, ell_origin_m=translation[2], **default), finite, tol["symmetry_relative"])
        individual = sum((deflection(manufactured_metric([c], eta=1), aray, **default) for c in components), np.zeros(2))
        checks.compare(f"asymmetric_{index}_superposition", result, individual, tol["symmetry_relative"])
        jac = lens_jacobian(asymmetric, aray/geom.D_l_m, geom,
                            angular_step_rad=.001*KPC_M/geom.D_l_m, **default)
        checks.compare(f"asymmetric_{index}_jacobian_symmetry", jac[0,1], jac[1,0],
                       tol["jacobian_antisymmetry_absolute"], mode="absolute")

    # Solve point-lens image positions numerically in theta/theta_E. All
    # magnifications come from the two-dimensional numerical lens Jacobian.
    theta_e = np.sqrt(4*G_SI*point.mass_kg/C_SI**2 * geom.D_ls_m/(geom.D_l_m*geom.D_s_m))
    angular_step = numerics["jacobian_step_theta_E"]*theta_e
    image_rows = []
    for u in fixture["point_source_beta_over_theta_E"]:
        lens_residual = lambda x: lens_map(fields[0], [x*theta_e, 0], geom, **default)[0]/theta_e-u
        roots = [brentq(lens_residual, max(u,.2), u+2, xtol=1e-12),
                 brentq(lens_residual, -2, -1/(u+2), xtol=1e-12)]
        root_refs = [.5*(u+np.sqrt(u*u+4)), .5*(u-np.sqrt(u*u+4))]
        signed_refs = [.5+(u*u+2)/(2*u*np.sqrt(u*u+4)), .5-(u*u+2)/(2*u*np.sqrt(u*u+4))]
        signed_values = []
        for parity, root, root_ref, mu_ref in zip(["positive", "negative"], roots, root_refs, signed_refs):
            checks.compare(f"point_image_{u}_{parity}", root, root_ref, tol["image_position_relative_theta_E"], mode="absolute")
            jac = lens_jacobian(fields[0], [root*theta_e, 0], geom, angular_step_rad=angular_step, **default)
            mu = signed_magnification(jac)
            signed_values.append(mu)
            checks.compare(f"point_signed_magnification_{u}_{parity}", mu, mu_ref, tol["magnification_relative"])
            checks.truth(f"point_parity_{u}_{parity}", np.sign(mu) == np.sign(mu_ref))
            image_rows.append(dict(u=u, parity=parity, theta_over_theta_E=root, analytic_theta_over_theta_E=root_ref,
                                   signed_magnification=mu, analytic_signed_magnification=mu_ref, jacobian=jac.tolist()))
        checks.compare(f"point_total_flux_magnification_{u}", sum(abs(v) for v in signed_values),
                       (u*u+2)/(u*np.sqrt(u*u+4)), tol["magnification_relative"])
        checks.compare(f"point_signed_magnification_sum_{u}", sum(signed_values), 1, tol["magnification_relative"])
    theta = np.array([1.5*theta_e, .7*theta_e])
    zero_geom = AngularDistances(geom.D_l_m, geom.D_s_m, 0.0)
    checks.compare("zero_distance_efficiency_map", lens_map(fields[0], theta, zero_geom, **default), theta,
                   tol["zero_absolute_radian"], mode="absolute")
    checks.compare("zero_distance_efficiency_jacobian", lens_jacobian(fields[0], theta, zero_geom, angular_step_rad=angular_step, **default),
                   np.eye(2), tol["jacobian_antisymmetry_absolute"], mode="absolute")
    half_geom = AngularDistances(geom.D_l_m, geom.D_s_m, .5*geom.D_ls_m)
    displacement = theta-lens_map(fields[0], theta, geom, **default)
    checks.compare("distance_efficiency_scaling", theta-lens_map(fields[0], theta, half_geom, **default),
                   .5*displacement, tol["deflection_relative"])
    # Expose a concrete wrong-distance negative control without altering the
    # operator. Its discrepancy must exceed the fixed correctness tolerance.
    wrong_geom = AngularDistances(geom.D_l_m, geom.D_s_m, geom.D_s_m-geom.D_l_m)
    wrong_error = np.linalg.norm((theta-lens_map(fields[0], theta, wrong_geom, **default))-displacement)/np.linalg.norm(displacement)
    checks.truth("detect_Ds_minus_Dl_negative_control", wrong_error > tol["deflection_relative"], wrong_distance_relative_error=float(wrong_error))
    return dict(theta_E_rad=float(theta_e), theta_E_arcsec=float(theta_e*180/np.pi*3600), images=image_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="new immutable local run directory, e.g. run-001")
    args = parser.parse_args()
    if not args.run_id.startswith("run-") or not args.run_id[4:].isdigit():
        parser.error("run-id must have form run-NNN; existing directories are never overwritten")
    out = OUTPUT/args.run_id
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    declaration_path = OUTPUT/"preimplementation-declaration.json"
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    checks = Checks()
    metadata = dict(disposition=DISPOSITION, started_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version, executable=sys.executable, platform=platform.platform(),
                    numpy=np.__version__, scipy=scipy.__version__, cpu_only=True,
                    observations_opened=False, observational_likelihoods_admitted=0,
                    prior_pilot_modified=False, sample_exposure_changed=False,
                    config_sha256=sha256(CONFIG), declaration_sha256=sha256(declaration_path),
                    run_id=args.run_id, command=[sys.executable, "-B", "scripts/run_mond_atlas_light_projection.py", "--run-id", args.run_id])
    try:
        # Preserve exact code before even running tests. No old observational
        # pipeline is imported, executed or read by this benchmark.
        hashes = {}
        for relative in OWNED:
            source = ROOT/relative
            target = out/"snapshot"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[relative] = sha256(source)
        for source in cfg["sources"]:
            path = ROOT/source["path"]
            actual_hash = sha256(path)
            hashes[source["path"]] = actual_hash
            checks.truth("primary_source_hash_"+path.name, actual_hash == source["sha256"], bytes=path.stat().st_size)
        metadata["input_sha256"] = hashes
        checks.truth("declared_theory_before_implementation", declaration["disposition"] == DISPOSITION
                     and declaration["config_sha256"] == sha256(CONFIG)
                     and declaration["implementation_files_present"] is False)
        checks.truth("observations_disabled", cfg["observational_access_allowed"] is False and cfg["observational_scoring_allowed"] is False)
        checks.truth("configured_constants_match_operator", cfg["constants"]["G_m3_kg_s2"] == G_SI
                     and cfg["constants"]["c_m_s"] == C_SI and cfg["constants"]["Msun_kg"] == MSUN_KG
                     and cfg["constants"]["kpc_m"] == KPC_M and cfg["constants"]["Mpc_m"] == MPC_M)
        test_command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_mond_atlas_light_projection.py", "-v"]
        test = subprocess.run(test_command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        (out/"unit-tests.txt").write_text(test.stdout+test.stderr, encoding="utf-8")
        checks.truth("unit_tests", test.returncode == 0, returncode=test.returncode, command=test_command)
        metadata["derived_synthetic_results"] = run_benchmarks(cfg, checks)
    except Exception:
        checks.truth("unhandled_benchmark_exception", False, traceback=traceback.format_exc())
    failures = [r["id"] for r in checks.rows if r["required"] and not r["passed"]]
    diagnostic_failures = [r["id"] for r in checks.rows if not r["required"] and not r["passed"]]
    metadata.update(completed_utc=datetime.now(timezone.utc).isoformat(), required_failures=failures,
                    retained_diagnostic_target_failures=diagnostic_failures, required_passed=not failures,
                    required_check_count=sum(r["required"] for r in checks.rows), total_check_count=len(checks.rows),
                    diagnostic_count=sum(not r["required"] for r in checks.rows),
                    scientific_admission="THEORY_BENCHMARK_ONLY; no observational or candidate-theory admission")
    write_json(out/"checks.json", checks.rows)
    write_json(out/"summary.json", metadata)
    manifest = {str(path.relative_to(out)).replace("\\", "/"): sha256(path)
                for path in sorted(out.rglob("*")) if path.is_file()}
    write_json(out/"sha256-manifest.json", manifest)
    print(json.dumps({k:metadata[k] for k in ["run_id", "disposition", "required_passed", "required_check_count", "total_check_count", "required_failures", "retained_diagnostic_target_failures"]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
