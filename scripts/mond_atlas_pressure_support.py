"""THEORY_BENCHMARK_ONLY: restricted steady, axisymmetric scalar-pressure gas.

Units are kpc, km/s and any consistently chosen mass unit. Positive force means
dPhi/dR (inward acceleration has the opposite sign). A SurfaceColumn needs the
density-weighted force through the column, not necessarily its midplane value.
No observation, gravity-law, Poisson, cube or general Jeans solver is provided.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class SurfaceColumn:
    radius: np.ndarray
    sigma: np.ndarray
    integrated_pressure: np.ndarray
    pressure_gradient: np.ndarray


@dataclass(frozen=True)
class VolumeLayer:
    radius: np.ndarray
    rho: np.ndarray
    pressure: np.ndarray
    pressure_gradient: np.ndarray


@dataclass(frozen=True)
class Balance:
    radius: np.ndarray
    support: np.ndarray
    rotation_squared: np.ndarray

    @property
    def feasible(self):
        return self.rotation_squared >= 0

    @property
    def status(self):
        return "STEADY_CIRCULAR_SOLUTION" if np.all(self.feasible) else "NO_STEADY_CIRCULAR_SOLUTION"

    def speed(self):
        if not np.all(self.feasible):
            raise ValueError("Negative vphi^2: no steady circular solution; signed values retained")
        return np.sqrt(self.rotation_squared)

    def recovered_force(self):
        # The admitted regular-center limit is zero. Never evaluate 0/0.
        result = np.zeros_like(self.radius)
        np.divide(self.rotation_squared + self.support, self.radius,
                  out=result, where=self.radius != 0)
        return result


def _balance(radius, density, pressure, gradient, force, radial_flow,
             vertical_flow, center_tolerance):
    r, d, p, dp, g = np.broadcast_arrays(*[
        np.asarray(x, dtype=float) for x in (radius, density, pressure, gradient, force)])
    if r.ndim != 1 or r.size == 0:
        raise ValueError("A nonempty one-dimensional radial sample is required")
    if not all(np.all(np.isfinite(x)) for x in (r, d, p, dp, g)):
        raise ValueError("All profile and force values must be finite")
    if np.any(r < 0) or np.any(d <= 0) or np.any(p < 0):
        raise ValueError("Radius and pressure must be nonnegative; density positive")
    for flow in (radial_flow, vertical_flow):
        if not np.all(np.isfinite(flow)) or np.any(np.asarray(flow) != 0):
            raise ValueError("Only zero mean radial and vertical flow is admitted")
    center = r == 0
    if np.any(np.abs(dp[center]) > center_tolerance) or np.any(np.abs(g[center]) > center_tolerance):
        raise ValueError("Nonregular center: force and scalar pressure gradient must vanish")
    support = -r * dp / d
    return Balance(r.copy(), support, r * g - support)


def surface_balance(column, density_weighted_force, *, radial_flow=0.0,
                    vertical_flow=0.0, center_tolerance=1e-12):
    if not isinstance(column, SurfaceColumn):
        raise TypeError("surface_balance requires SurfaceColumn (Sigma, Pi)")
    return _balance(column.radius, column.sigma, column.integrated_pressure,
                    column.pressure_gradient, density_weighted_force, radial_flow,
                    vertical_flow, center_tolerance)


def volume_balance(layer, local_force, *, radial_flow=0.0, vertical_flow=0.0,
                   center_tolerance=1e-12):
    if not isinstance(layer, VolumeLayer):
        raise TypeError("volume_balance requires VolumeLayer (rho, P)")
    return _balance(layer.radius, layer.rho, layer.pressure, layer.pressure_gradient,
                    local_force, radial_flow, vertical_flow, center_tolerance)


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def gaussian_column(radius, density_scale, dispersion0, dispersion_scale=None,
                    normalization=1.0):
    """Analytic continuation, without a cut at the outer sampling radius."""
    _positive(density_scale, "density_scale")
    _positive(normalization, "normalization")
    if not np.isfinite(dispersion0) or dispersion0 < 0:
        raise ValueError("dispersion0 must be finite and nonnegative")
    if dispersion_scale is not None:
        _positive(dispersion_scale, "dispersion_scale")
    r = np.asarray(radius, dtype=float)
    sigma = normalization * np.exp(-0.5 * (r / density_scale)**2)
    inverse_c_scale2 = 0.0 if dispersion_scale is None else dispersion_scale**-2
    c2 = dispersion0**2 * np.exp(-0.5 * r*r * inverse_c_scale2)
    pi = sigma * c2
    derivative = -r * pi * (density_scale**-2 + inverse_c_scale2)
    return SurfaceColumn(r, sigma, pi, derivative)


def potential_force(radius, kind, amplitude, core=None):
    """Harmonic amplitude is Omega^2; cored_log amplitude is V0^2."""
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Potential amplitude must be finite and nonnegative")
    r = np.asarray(radius, dtype=float)
    if kind == "harmonic":
        return amplitude * r
    if kind == "cored_log":
        _positive(core, "core")
        return amplitude * r / (core*core + r*r)
    raise ValueError(f"Unrecognized potential: {kind}")


def effective_pressure_variance(thermal_variance, turbulent_variances):
    """P/rho: thermal kT/mean mass plus isotropic one-component Reynolds stress.

    Off-diagonal stress is assumed zero. A vector of R, phi, z variances is
    required explicitly; a single observed spectral width is not sufficient.
    """
    values = np.asarray(turbulent_variances, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Supply three finite nonnegative turbulent variances")
    if not np.isfinite(thermal_variance) or thermal_variance < 0:
        raise ValueError("Thermal variance must be finite and nonnegative")
    if not np.allclose(values, values[0], rtol=1e-12, atol=0):
        raise ValueError("Anisotropic stress requires a tensor Jeans/Euler closure")
    return float(thermal_variance + values[0])


def spectral_variance(thermal_variance, mean_to_tracer_mass, turbulent_1d_variance,
                      instrument_variance=0.0, unresolved_variance=0.0):
    values = np.array([thermal_variance, turbulent_1d_variance,
                       instrument_variance, unresolved_variance], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Broadening variances must be finite and nonnegative")
    _positive(mean_to_tracer_mass, "mean_to_tracer_mass")
    return float(thermal_variance * mean_to_tracer_mass + turbulent_1d_variance
                 + instrument_variance + unresolved_variance)


def case_column(radius, case, model="known_pressure"):
    if model not in ("known_pressure", "pressure_blind"):
        raise ValueError(f"Unknown fitted closure: {model}")
    return gaussian_column(radius, case["density_scale_kpc"],
                           case["dispersion0_km_s"] if model == "known_pressure" else 0.0,
                           case["dispersion_scale_kpc"])


def case_balance(radius, case, model="known_pressure", amplitude=None):
    amplitude = case["amplitude"] if amplitude is None else amplitude
    return surface_balance(case_column(radius, case, model),
                           potential_force(radius, case["potential"], amplitude,
                                           case["core_kpc"]))


def independent_truth(radius, case):
    """Separately expressed manufactured oracle; does not call balance/profile.

    This closed-form oracle generates the study truth. Analytic derivatives in
    the production profile/force functions are also checked by finite differences.
    """
    r = np.asarray(radius, dtype=float)
    if case["potential"] == "harmonic":
        vc2 = case["amplitude"] * r**2
    elif case["potential"] == "cored_log":
        denominator = np.full_like(r, np.inf)
        np.divide(case["core_kpc"], r, out=denominator, where=r != 0)
        vc2 = case["amplitude"] / (1 + denominator**2)
    else:
        raise ValueError("Unknown manufactured potential")
    s0 = case["dispersion0_km_s"]
    ld = case["density_scale_kpc"]
    lc = case["dispersion_scale_kpc"]
    if lc is None:
        drift = (s0 * r / ld)**2
    else:
        drift = s0**2 * np.exp(-r*r / (2*lc*lc)) * ((r/ld)**2 + (r/lc)**2)
    return vc2 - drift, vc2, drift


def fit_amplitude(radius, observed, train_indices, case, model, study):
    r = np.asarray(radius, dtype=float)
    y = np.asarray(observed, dtype=float)
    train = np.asarray(train_indices, dtype=int)
    if y.shape != r.shape or not np.all(np.isfinite(y)):
        raise ValueError("Observed synthetic speed shape/values invalid")
    if train.size == 0 or len(np.unique(train)) != len(train) or np.any(train < 0) or np.any(train >= len(r)):
        raise ValueError("Training indices invalid")
    sigma = study["noise_sigma_km_s"]
    _positive(sigma, "noise sigma")
    projection = math.sin(math.radians(study["inclination_deg"]))
    if not 0 < projection <= 1:
        raise ValueError("Fixed inclination must provide a positive projection")
    support = case_balance(r, case, model, amplitude=0).support
    basis = r * potential_force(r, case["potential"], 1.0, case["core_kpc"])
    positive = basis > 0
    physical_floor = float(np.max(support[positive]/basis[positive])) if np.any(positive) else 0.0
    lo, hi = case["amplitude_bounds"]
    lo = max(lo, physical_floor + 1e-9)
    if lo >= hi:
        raise ValueError("No feasible amplitude interval within frozen bounds")

    def prediction(amplitude):
        return (study["systemic_km_s"] + projection
                * case_balance(r, case, model, amplitude=amplitude).speed())

    def objective(amplitude):
        residual = (y[train] - prediction(amplitude)[train]) / sigma
        return float(residual @ residual)

    opt = study["optimizer"]
    result = minimize_scalar(objective, bounds=(lo, hi), method=opt["method"],
                             options={"xatol": opt["xatol"], "maxiter": opt["maxiter"]})
    pred = prediction(float(result.x))
    return {"success": bool(result.success), "message": str(result.message),
            "amplitude": float(result.x), "objective": float(result.fun),
            "nfev": int(result.nfev), "effective_bounds": [float(lo), float(hi)],
            "at_bound": bool(min(result.x-lo, hi-result.x) < 1e-5*(hi-lo))}, pred


def numerical_controls(config):
    """Target-free controls; every result is returned, including any exception."""
    cfg = config["independent_controls"]
    tol = cfg["analytic_relative_tolerance"]
    rows = []

    def check(name, calculate, tolerance=tol):
        try:
            metric, details = calculate()
            metric = float(metric)
            rows.append({"name": name, "passed": bool(np.isfinite(metric) and metric <= tolerance),
                         "metric": metric if np.isfinite(metric) else None,
                         "maximum": tolerance, "details": details})
        except Exception as exc:
            rows.append({"name": name, "passed": False, "error": repr(exc)})

    def rel(actual, expected):
        a, b = np.asarray(actual), np.asarray(expected)
        return float(np.max(np.abs(a-b)) / max(1.0, float(np.max(np.abs(b)))))

    def rejects(call, exception=ValueError):
        try:
            call()
        except exception:
            return 0.0, {"expected_rejection": exception.__name__}
        return 1.0, {"expected_rejection": "NOT_RAISED"}

    r = np.linspace(0, 4, 65)
    base = gaussian_column(r, 1.2, 18)
    g = 625*r
    bal = surface_balance(base, g)
    check("harmonic_gaussian_exact", lambda: (rel(bal.rotation_squared, 400*r*r), {}))
    check("pressureless_limit", lambda: (rel(surface_balance(gaussian_column(r, 1.2, 0), g).rotation_squared, 625*r*r), {}))
    check("force_recovery_and_regular_center", lambda: (rel(bal.recovered_force(), g), {"v2_center": float(bal.rotation_squared[0])}))
    check("declining_pressure_slows", lambda: (float(not np.all((bal.rotation_squared[1:] < r[1:]*g[1:]) & (bal.support[1:] > 0))), {}))
    check("constant_integrated_pressure", lambda: (rel(surface_balance(SurfaceColumn(r, base.sigma, np.ones_like(r), np.zeros_like(r)), g).rotation_squared, r*g), {}))
    check("increasing_pressure_supercircular", lambda: (float(not np.all(surface_balance(SurfaceColumn(r, np.ones_like(r), 1+r*r, 2*r), g).rotation_squared[1:] > (r*g)[1:])), {"support_sign_is_not_clamped": True}))
    check("density_normalization", lambda: (rel(surface_balance(gaussian_column(r, 1.2, 18, normalization=137), g).rotation_squared, bal.rotation_squared), {}))
    # Mass unit unchanged, kpc -> pc: Sigma / 10^6, Pi / 10^6, gradient / 10^9.
    pc = SurfaceColumn(r*1000, base.sigma/1e6, base.integrated_pressure/1e6, base.pressure_gradient/1e9)
    check("kpc_to_pc_units", lambda: (rel(surface_balance(pc, g/1000).rotation_squared, bal.rotation_squared), {}))
    check("nonregular_center_rejected", lambda: rejects(lambda: surface_balance(SurfaceColumn(np.array([0.]), np.array([1.]), np.array([1.]), np.array([-1.])), np.array([0.]))))
    check("nonregular_center_force_rejected", lambda: rejects(lambda: surface_balance(gaussian_column(np.array([0.]), 1, 1), np.array([1.]))))
    check("volume_in_surface_rejected", lambda: rejects(lambda: surface_balance(VolumeLayer(r, base.sigma, base.integrated_pressure, base.pressure_gradient), g), TypeError))
    check("surface_in_volume_rejected", lambda: rejects(lambda: volume_balance(base, g), TypeError))
    impossible = case_balance(r, config["study"]["impossible_case"])
    check("negative_v2_retained", lambda: (float(not np.all(impossible.rotation_squared[1:] < 0)), {"minimum_v2": float(impossible.rotation_squared.min()), "status": impossible.status}))
    check("negative_v2_speed_rejected", lambda: rejects(impossible.speed))
    check("radial_flow_rejected", lambda: rejects(lambda: surface_balance(base, g, radial_flow=1.0)))
    check("vertical_flow_rejected", lambda: rejects(lambda: surface_balance(base, g, vertical_flow=1.0)))
    # For u_R=1, d(R Sigma u_R)/dR = Sigma(1-R^2/L^2), not zero.
    check("constant_radial_flow_not_steady", lambda: (float(np.max(np.abs(base.sigma*(1-r*r/1.2**2))) < 0.1), {"mass_flux_divergence_nonzero": True}))
    check("isotropic_thermal_turbulent_sum", lambda: (abs(effective_pressure_variance(81, [243]*3)-324), {}))
    check("anisotropic_stress_rejected", lambda: rejects(lambda: effective_pressure_variance(81, [243, 100, 243])))
    check("single_line_width_not_tensor", lambda: rejects(lambda: effective_pressure_variance(81, 243)))
    check("thermal_tracer_mass_conversion", lambda: (abs(spectral_variance(100, 0.05, 9, 4, 16)-34), {"pressure_variance": effective_pressure_variance(100, [9]*3), "line_variance": 34}))
    def linewidth_degeneracy():
        a = spectral_variance(0, 1, 324, 36, 0)
        b = spectral_variance(0, 1, 100, 36, 224)
        return abs(a-b), {"same_line_variance": a, "different_pressure_variances": [324, 100]}
    check("same_line_width_different_support", linewidth_degeneracy)

    test = config["study"]["cases"][2]
    points = np.linspace(0.1, 4, 40)
    h = cfg["finite_difference_step_kpc"]
    def scalar_phi(x):
        return 0.5 * test["amplitude"] * np.log1p((x/test["core_kpc"])**2)
    def scalar_pi(x):
        return test["dispersion0_km_s"]**2 * np.exp(-x*x/(2*test["density_scale_kpc"]**2) - x*x/(2*test["dispersion_scale_kpc"]**2))
    check("potential_finite_difference", lambda: (rel((scalar_phi(points+h)-scalar_phi(points-h))/(2*h), potential_force(points, test["potential"], test["amplitude"], test["core_kpc"])), {"step_kpc": h}), cfg["finite_difference_relative_tolerance"])
    check("pressure_finite_difference", lambda: (rel((scalar_pi(points+h)-scalar_pi(points-h))/(2*h), case_column(points, test).pressure_gradient), {"step_kpc": h}), cfg["finite_difference_relative_tolerance"])
    check("dispersion_gradient_included", lambda: (rel(case_balance(points, test).rotation_squared, independent_truth(points, test)[0]), {}))
    errors, edge_errors = [], []
    for n in cfg["resolution_nodes"]:
        grid = np.linspace(0, 4, n)
        derivative = np.gradient(scalar_pi(grid), grid, edge_order=2)
        exact = case_column(grid, test).pressure_gradient
        scale = np.max(np.abs(exact))
        errors.append(float(np.max(np.abs(derivative-exact))/scale))
        edge_errors.append(float(abs(derivative[-1]-exact[-1])/scale))
    check("pressure_gradient_resolution", lambda: (errors[-1]/errors[0], {"nodes": cfg["resolution_nodes"], "scaled_max_errors_including_boundaries": errors}), cfg["resolution_last_to_first_max"])
    check("pressure_gradient_finest_accuracy", lambda: (errors[-1], {}), cfg["resolution_final_scaled_error_max"])
    check("outer_boundary_derivative", lambda: (edge_errors[-1], {"scaled_outer_errors": edge_errors, "boundary": "smooth continuation, one-sided second order"}), cfg["resolution_final_scaled_error_max"])
    check("outer_pressure_not_vacuum_edge", lambda: (float(not scalar_pi(4) > 0), {"Pi_at_Rmax": float(scalar_pi(4))}))

    flare = cfg["manufactured_flare"]
    omega2, c2 = flare["omega_km_s_kpc"]**2, flare["sigma_km_s"]**2
    ls, lf, h0 = flare["density_scale_kpc"], flare["flare_scale_kpc"], flare["height0_kpc"]
    local_errors, vertical_errors, quad_errors, midplane_errors, radial_fd_errors = [], [], [], [], []
    quadrature_receipt = []
    for radius in [0.2, 1.0, 3.0]:
        height = h0*math.exp(radius*radius/(4*lf*lf))
        hp_over_h = radius/(2*lf*lf)
        sig = math.exp(-radius*radius/(2*ls*ls))
        expected = radius*radius*(omega2-c2/ls**2-c2/(2*lf**2))
        def rho(z):
            return sig/(math.sqrt(2*math.pi)*height)*math.exp(-0.5*(z/height)**2)
        def local_g(z):
            return omega2*radius-c2*hp_over_h*(z/height)**2
        def phi_at(rad, z):
            hz = h0*math.exp(rad*rad/(4*lf*lf))
            return 0.5*omega2*rad*rad+0.5*c2*(z/hz)**2
        def density_at(rad, z):
            hz = h0*math.exp(rad*rad/(4*lf*lf))
            return math.exp(-rad*rad/(2*ls*ls))/(math.sqrt(2*math.pi)*hz)*math.exp(-0.5*(z/hz)**2)
        for z_over_h in [-2, -0.5, 0, 0.5, 2]:
            z = z_over_h*height
            density = rho(z)
            dlogrho = -radius/ls**2-hp_over_h+(z/height)**2*hp_over_h
            layer = VolumeLayer(np.array([radius]), np.array([density]), np.array([c2*density]), np.array([c2*density*dlogrho]))
            local_errors.append(rel(volume_balance(layer, np.array([local_g(z)])).rotation_squared, expected))
            dz = 1e-4*height
            dp_dz = c2*(rho(z+dz)-rho(z-dz))/(2*dz)
            dphi_dz = (phi_at(radius,z+dz)-phi_at(radius,z-dz))/(2*dz)
            vertical_errors.append(abs(dp_dz/density+dphi_dz)/max(1.0,c2/height))
            numeric_g = (phi_at(radius+h,z)-phi_at(radius-h,z))/(2*h)
            numeric_dp = c2*(density_at(radius+h,z)-density_at(radius-h,z))/(2*h)
            radial_fd_errors.extend([rel(numeric_g, local_g(z)), rel(numeric_dp/density, c2*dlogrho)])
        for zmax in cfg["vertical_quadrature_heights"]:
            mass = quad(rho, -zmax*height, zmax*height, epsabs=1e-12, epsrel=1e-12)[0]
            integrated_force = quad(lambda z: rho(z)*local_g(z), -zmax*height, zmax*height, epsabs=1e-10, epsrel=1e-12)[0]/sig
            integrated_pi = quad(lambda z: c2*rho(z), -zmax*height, zmax*height, epsabs=1e-12, epsrel=1e-12)[0]
            surf = gaussian_column(np.array([radius]), ls, math.sqrt(c2))
            value = surface_balance(surf, np.array([integrated_force])).rotation_squared[0]
            err = max(rel(value, expected), abs(mass/sig-1), rel(integrated_pi, sig*c2))
            quadrature_receipt.append({"radius": radius, "height_cut_sigma": zmax, "error": err, "mass_fraction": mass/sig})
            if zmax == max(cfg["vertical_quadrature_heights"]):
                quad_errors.append(err)
        wrong = surface_balance(surf, np.array([omega2*radius])).rotation_squared[0]
        midplane_errors.append(rel(wrong-expected, radius*c2*hp_over_h))
    check("flaring_local_radial_Euler", lambda: (max(local_errors), {"samples": len(local_errors)}))
    check("flaring_vertical_Euler", lambda: (max(vertical_errors), {"method": "central differences of density and scalar potential at fixed radius"}), cfg["finite_difference_relative_tolerance"])
    check("flaring_radial_derivatives", lambda: (max(radial_fd_errors), {"method": "central differences at fixed physical height"}), cfg["finite_difference_relative_tolerance"])
    check("flaring_vertical_quadrature", lambda: (max(quad_errors), {"quadrature": quadrature_receipt, "normalization": "full analytic Sigma, no tail renormalization"}), cfg["vertical_quad_relative_tolerance"])
    check("midplane_surface_substitution_error", lambda: (max(midplane_errors), {"wrong_v2_minus_correct_v2": "R*c_eff^2*dln(h)/dR > 0"}))

    study = config["study"]
    radial = np.linspace(study["radius_min_kpc"], study["radius_max_kpc"], study["radius_count"])
    proj = math.sin(math.radians(study["inclination_deg"]))
    for case in study["cases"]:
        check("manufactured_oracle_"+case["id"], lambda case=case: (rel(case_balance(radial, case).rotation_squared, independent_truth(radial, case)[0]), {}))
        def recover(case=case):
            obs = study["systemic_km_s"] + proj*np.sqrt(independent_truth(radial, case)[0])
            fit, prediction = fit_amplitude(radial, obs, study["train_indices"], case, "known_pressure", study)
            err = abs(fit["amplitude"]/case["amplitude"]-1) if fit["success"] else 1.0
            return err, {"fit": fit}
        check("noiseless_force_recovery_"+case["id"], recover, cfg["fit_amplitude_relative_tolerance"])
    harmonic = study["cases"][1]
    def blind_bias():
        obs = proj*20*radial
        fit, pred = fit_amplitude(radial, obs, study["train_indices"], harmonic, "pressure_blind", study)
        return abs(fit["amplitude"]/400-1), {"expected_blind_amplitude": 400, "true_amplitude": 625, "force_bias_fraction": -0.36, "fit": fit}
    check("pressure_blind_exact_wrong_force", blind_bias, cfg["fit_amplitude_relative_tolerance"])
    def ridge():
        alternatives = []
        for dispersion in study["unknown_pressure_harmonic_alternatives_km_s"]:
            q = 400+(dispersion/1.2)**2
            v2 = q*radial**2-(dispersion*radial/1.2)**2
            alternatives.append({"dispersion": dispersion, "force_amplitude": q, "speed_error": rel(v2, 400*radial**2)})
        return max(x["speed_error"] for x in alternatives), {"alternatives": alternatives, "interpretation": "exact force-pressure ridge, not an inferred posterior"}
    check("unknown_pressure_exact_force_ridge", ridge)
    return rows
