"""Spherical hydrostatic development diagnostics with explicit source support.

Only deprojected SZ pressure is scored. Local P_e/n_e is NOT a projected X-ray
temperature. No inferred total mass or object-specific gravity constant enters.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_triangular

from .saturated_actions import SaturatedActionSpec

DEVELOPMENT_CLUSTERS = frozenset({"A1644", "A1795", "A2142", "A2255", "A2319", "A3266", "A85", "ZW1215"})
G = 6.67430e-11
GM_SUN = 1.32712440041279419e20
KPC = 3.085677581491367e19
PROTON_MASS = 1.67262192369e-27
MU, MU_E = .61, 1.148
PRESSURE_SI_PER_KEV_CM3 = 1.602176634e-10


def _positive_array(values, name):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or np.any(~np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"positive finite vector required: {name}")
    return a


@dataclass
class PowerLawDensity:
    """Log-linear density between knots; exact mass integral, flat inner core.

    Adding observation radii cannot change the source mass. No extrapolation
    beyond the last measured source radius is permitted.
    """
    radius: np.ndarray
    density: np.ndarray

    def __post_init__(self):
        self.radius = _positive_array(self.radius, "radius")
        self.density = _positive_array(self.density, "density")
        if len(self.radius) < 2 or self.radius.shape != self.density.shape or np.any(np.diff(self.radius) <= 0):
            raise ValueError("matched strictly increasing source radii required")
        self.slopes = np.diff(np.log(self.density))/np.diff(np.log(self.radius))
        self.mass_at_knots = np.empty_like(self.radius)
        self.mass_at_knots[0] = 4*np.pi*self.density[0]*self.radius[0]**3/3
        increments = self._shell_mass(np.arange(len(self.radius)-1), self.radius[1:])
        self.mass_at_knots[1:] = self.mass_at_knots[0]+np.cumsum(increments)

    def _shell_mass(self, index, radius):
        exponent = self.slopes[index]+3
        log_ratio = np.log(radius/self.radius[index])
        quotient = np.empty_like(log_ratio, dtype=float)
        small = np.abs(exponent) < 1e-12
        np.divide(np.expm1(exponent*log_ratio), exponent, out=quotient, where=~small)
        quotient = np.where(small, log_ratio, quotient)
        return 4*np.pi*self.density[index]*self.radius[index]**3*quotient

    def evaluate(self, radius):
        r = _positive_array(radius, "query radius")
        if np.any(r > self.radius[-1]*(1+1e-12)):
            raise ValueError("outside measured density support")
        index = np.clip(np.searchsorted(self.radius, r, side="right")-1, 0, len(self.radius)-2)
        rho = self.density[index]*np.exp(self.slopes[index]*np.log(r/self.radius[index]))
        mass = self.mass_at_knots[index]+self._shell_mass(index, r)
        inner = r < self.radius[0]
        rho = np.where(inner, self.density[0], rho)
        mass = np.where(inner, 4*np.pi*self.density[0]*r**3/3, mass)
        return rho, mass


def integrate_electron_pressure(radius_m, ne_m3, acceleration, fraction, boundary_pressure_si):
    """Solve d[P_e/(1-f)]/dr=-mu*m_p*n_e*g, f=P_nonthermal/P_total.

    A radially varying pressure fraction is not a fraction of the pressure
    gradient. The measured outer thermal pressure is an explicit boundary.
    """
    r = _positive_array(radius_m, "pressure radius")
    ne = _positive_array(ne_m3, "electron density")
    g = _positive_array(acceleration, "inward acceleration")
    f = np.asarray(fraction, dtype=float)
    if (r.shape != ne.shape or r.shape != g.shape or r.shape != f.shape or
            np.any(np.diff(r) <= 0) or np.any(~np.isfinite(f)) or np.any(f < 0) or np.any(f >= 1) or
            not np.isfinite(boundary_pressure_si) or boundary_pressure_si <= 0):
        raise ValueError("invalid hydrostatic inputs or nonthermal pressure fraction")
    integrand = MU*PROTON_MASS*ne*g
    integral = cumulative_trapezoid(integrand[::-1], x=-r[::-1], initial=0)[::-1]
    coefficient = (1-f)/(1-f[-1])
    pressure = coefficient*boundary_pressure_si+(1-f)*integral
    return pressure, coefficient


def boundary_residual_covariance(covariance, indices, anchor, coefficients):
    """Cov(P_i-k_i P_anchor), including the measured boundary correlations."""
    c = np.asarray(covariance, dtype=float)
    ids = np.asarray(indices, dtype=int)
    k = np.asarray(coefficients, dtype=float)
    if (c.ndim != 2 or c.shape[0] != c.shape[1] or len(ids) != len(k) or
            len(set(ids)) != len(ids) or anchor in ids or np.any(ids < 0) or np.any(ids >= len(c)) or
            not 0 <= anchor < len(c) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(k)) or
            not np.allclose(c, c.T, rtol=1e-10, atol=0)):
        raise ValueError("invalid covariance, anchor or target indices")
    operator = np.eye(len(c))[ids]
    operator[:, anchor] -= k
    result = operator@c@operator.T
    np.linalg.cholesky(result/np.outer(np.sqrt(np.diag(result)), np.sqrt(np.diag(result))))
    return result


def covariance_loss(residual, covariance):
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance/np.outer(scale, scale)
    white = solve_triangular(np.linalg.cholesky(correlation), np.asarray(residual)/scale, lower=True)
    return float(white@white/len(white))


def _verified_path(root: Path, relative: str, expected_hash: str) -> Path:
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("source path or SHA-256 mismatch")
    return path


def load_development_packet(root, cluster, source_contract, covariance_manifest):
    """Read only the named development cluster and required table columns."""
    if cluster not in DEVELOPMENT_CLUSTERS:
        raise PermissionError("reserved or unregistered cluster: no data access")
    records = {r["role"]: r for r in source_contract["input_contract"]["files"] if r["cluster"] == cluster}
    root = Path(root)
    loaded, access = {}, []
    for role in ("density", "pressure", "stellar_mass"):
        if role not in records:
            if role == "stellar_mass":
                continue
            raise ValueError("missing required development source")
        record = records[role]
        relative = source_contract["input_contract"]["raw_root"]+"/"+record["member"]
        path = _verified_path(root, relative, record["sha256"])
        with fits.open(path, memmap=False) as hdus:
            loaded[role] = (hdus[2].data.copy(), dict(hdus[2].header))
        access.append({"path": relative, "sha256": record["sha256"], "hdu": 2, "role": role})
    density, dh = loaded["density"]
    pressure, ph = loaded["pressure"]
    if (list(density.dtype.names) != ["RW_X", "NE", "ERR_NE_LO", "ERR_NE_HI"] or
            list(pressure.dtype.names) != ["RW_SZ", "P_SZ", "eP_SZ"] or
            not np.isclose(dh["R500"], ph["R500"], rtol=1e-12)):
        raise ValueError("density/pressure schema or radial normalization changed")
    r500 = float(dh["R500"])
    packet = {"cluster": cluster, "r500_kpc": r500,
              "density_radius_kpc": np.asarray(density["RW_X"], float)*r500,
              "ne_cm3": np.asarray(density["NE"], float),
              "ne_low_error": np.asarray(density["ERR_NE_LO"], float),
              "ne_high_error": np.asarray(density["ERR_NE_HI"], float),
              "pressure_radius_kpc": np.asarray(pressure["RW_SZ"], float)*r500,
              "pressure": np.asarray(pressure["P_SZ"], float)*float(ph["P500"]),
              "pressure_error": np.asarray(pressure["eP_SZ"], float)*float(ph["P500"]),
              "stellar": None, "access": access}
    for key in ("density_radius_kpc", "ne_cm3", "pressure_radius_kpc", "pressure", "pressure_error"):
        _positive_array(packet[key], key)
    if np.any(np.diff(packet["density_radius_kpc"]) <= 0) or np.any(np.diff(packet["pressure_radius_kpc"]) <= 0):
        raise ValueError("radial ordering changed")
    if "stellar_mass" in loaded:
        stellar, _ = loaded["stellar_mass"]
        packet["stellar"] = {"radius_kpc": np.asarray(stellar["RADIUS"], float),
                             "mass_msun": np.asarray(stellar["MSTAR"], float)}
    record = next(r for r in covariance_manifest["records"] if r["cluster"] == cluster)
    path = _verified_path(root, record["path"], record["sha256"])
    with fits.open(path, memmap=False) as hdus:
        # HDU 1 contains catalogue mass metadata; it is not parsed or used.
        table = hdus[4].data
        cr = np.asarray(table["RW"][0], float)
        cp = np.asarray(table["FLUX"][0], float)
        cov = np.asarray(table["COVMAT"][0], float)
    if cr.shape != packet["pressure_radius_kpc"].shape or not np.allclose(cr, packet["pressure_radius_kpc"], rtol=1e-8):
        raise ValueError("covariance and released pressure radial bins do not match")
    ratio = packet["pressure"]/cp
    mean_scale = float(np.median(ratio))
    if not np.allclose(ratio, mean_scale, rtol=1e-8):
        raise ValueError("pressure products are not related by a constant normalization")
    corr = cov/np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    np.linalg.cholesky(corr)
    if not np.allclose(corr, corr.T, rtol=1e-10, atol=1e-12):
        raise ValueError("asymmetric published covariance")
    packet["covariance"] = corr*np.outer(packet["pressure_error"], packet["pressure_error"])
    packet["native_scaled_covariance"] = cov*mean_scale**2
    packet["covariance_mapping"] = {
        "mean_profile_scale": mean_scale,
        "quoted_error_over_scaled_native_sigma": (packet["pressure_error"]/(mean_scale*np.sqrt(np.diag(cov)))).tolist(),
        "correlation_min_eigenvalue": float(np.linalg.eigvalsh(corr).min()),
        "scope": "transfer native correlation to high-level quoted errors; retained assumption, not identical published covariance",
    }
    packet["access"].append({"path": record["path"], "sha256": record["sha256"],
                              "hdu": 4, "columns": ["RW", "FLUX", "COVMAT"], "role": "pressure_covariance"})
    return packet


def pressure_indices(packet):
    r, density_r = packet["pressure_radius_kpc"], packet["density_radius_kpc"]
    eligible = np.flatnonzero((np.arange(len(r)) >= 3) & (r >= density_r[0]) & (r <= density_r[-1]))
    if len(eligible) < 4:
        raise ValueError("fewer than three pressure targets and one boundary after declared source cuts")
    anchor = int(eligible[-1])
    targets = eligible[:-1]
    dispositions = []
    for i in range(len(r)):
        status = ("EXCLUDED_PUBLISHED_INNER_BEAM_LIMIT" if i < 3 else
                  "OUTSIDE_MEASURED_DENSITY_SUPPORT" if not density_r[0] <= r[i] <= density_r[-1] else
                  "BOUNDARY_UNSCORED" if i == anchor else "SCORED_DEVELOPMENT")
        dispositions.append({"index": i, "radius_kpc": float(r[i]), "status": status})
    return targets, anchor, dispositions


def predict_pressure(packet, model, nuisance, *, nodes=2049):
    ids, anchor, dispositions = pressure_indices(packet)
    distance = nuisance.get("distance_scale", 1.0)
    calibration = nuisance.get("pressure_calibration", 1.0)
    ne = packet["ne_cm3"].copy()
    density_shift = nuisance.get("density_error_shift", 0)
    ne += density_shift*packet["ne_high_error" if density_shift > 0 else "ne_low_error"]
    if np.any(ne <= 0):
        raise ValueError("density uncertainty scenario nonpositive; retained failure, no clipping")
    ne *= distance**(-.5)
    source_r = packet["density_radius_kpc"]*distance*KPC
    profile = PowerLawDensity(source_r, ne*1e6*MU_E*PROTON_MASS)
    anchor_r = packet["pressure_radius_kpc"][anchor]*distance*KPC
    target_r = packet["pressure_radius_kpc"][ids]*distance*KPC
    radius = np.unique(np.concatenate([np.geomspace(source_r[0], anchor_r, nodes),
                                       source_r[source_r <= anchor_r], target_r, [anchor_r]]))
    rho, gas_mass = profile.evaluate(radius)
    if packet["stellar"] is None:
        star_gm = G*gas_mass*nuisance.get("missing_stellar_gas_ratio", .1)*nuisance.get("stellar_scale", 1)
        stellar_adjustment = {"missing_stellar_rule": True, "monotone_corrections": 0}
    else:
        stellar = packet["stellar"]
        sr = _positive_array(stellar["radius_kpc"], "stellar radius")*distance*KPC
        sm = _positive_array(stellar["mass_msun"], "stellar mass")
        if np.any(np.diff(sr) <= 0):
            raise ValueError("stellar radii not ordered")
        monotone = np.maximum.accumulate(sm)
        mass = np.interp(radius, sr, monotone)
        mass = np.where(radius < sr[0], monotone[0]*(radius/sr[0])**3, mass)
        star_gm = GM_SUN*mass*nuisance.get("stellar_scale", 1)*distance**2
        stellar_adjustment = {"missing_stellar_rule": False, "monotone_corrections": int(np.count_nonzero(monotone != sm)),
                              "maximum_monotone_fraction_change": float(np.max(monotone/sm-1)),
                              "outer_constant_mass_node_count": int(np.count_nonzero(radius > sr[-1]))}
    gbar = (G*gas_mass+star_gm)/radius**2
    if model["family"] == "newtonian":
        acceleration = gbar
    elif model["family"] == "rar_comparator":
        acceleration = gbar/(-np.expm1(-np.sqrt(gbar/model["a0"])))
    elif model["family"] == "saturated_qumond":
        spec = SaturatedActionSpec("qumond", shape=model["shape"], epsilon=model["epsilon"])
        acceleration = gbar*(1+spec.delta_nu(gbar/model["a0"]))
    else:
        raise NotImplementedError("no spherical pressure adapter for this family")
    fraction = nuisance["outer_nonthermal_fraction"]*(radius/anchor_r)
    boundary = packet["pressure"][anchor]*calibration/distance*PRESSURE_SI_PER_KEV_CM3
    p, k = integrate_electron_pressure(radius, rho/(MU_E*PROTON_MASS), acceleration, fraction, boundary)
    target_p = np.interp(target_r, radius, p)/PRESSURE_SI_PER_KEV_CM3
    target_k = np.interp(target_r, radius, k)
    return {"indices": ids, "anchor": anchor, "dispositions": dispositions,
            "prediction": target_p, "observed": packet["pressure"][ids]*calibration/distance,
            "boundary_coefficients": target_k, "pressure_scale": calibration/distance,
            "stellar_adjustment": stellar_adjustment,
            "source_acceleration_m_s2": np.interp(target_r, radius, gbar),
            "predicted_acceleration_m_s2": np.interp(target_r, radius, acceleration)}
