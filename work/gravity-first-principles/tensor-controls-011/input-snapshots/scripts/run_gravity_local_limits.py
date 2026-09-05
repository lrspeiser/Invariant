"""Create an append-only isolated-monopole diagnostic from published summaries."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invariant_gravity_extensions.actions import generate_specs
from invariant_gravity_extensions.local_limits import (
    Orbit,
    baseline_delta_nu,
    binet_precession,
    logarithmic_precession,
    mas_per_century,
    perihelion_first_order,
    power_tail,
)


def calculate(config: dict, candidate_config: dict) -> dict:
    rows = []
    century = config["century_s"]
    epsilon = config["epsilon"]
    for planet in config["planets"]:
        orbit = Orbit(planet["a_au"] * config["au_m"], planet["e"], config["gm_sun_m3_s2"])
        center, width = planet["interval_center_mas_cy"], planet["interval_halfwidth_mas_cy"]
        interval = [center - width, center + width]
        if width <= 0 or not interval[0] < 0 < interval[1]:
            raise ValueError("screen expects positive-width intervals containing zero")
        for a0 in config["a0_m_s2"]:
            first = [perihelion_first_order(orbit, a0, lambda y: baseline_delta_nu(y, epsilon),
                                            nodes=n) for n in config["quadrature_nodes"]]
            analytic = logarithmic_precession(orbit, a0)
            direct = binet_precession(orbit, a0)
            rate = mas_per_century(first[-1], orbit, century)
            peri, apo = orbit.semimajor_m * (1 - orbit.eccentricity), orbit.semimajor_m * (1 + orbit.eccentricity)
            max_fraction = float(baseline_delta_nu(orbit.gm_m3_s2 / (a0 * apo**2), epsilon))
            tails = []
            for exponent in config["power_tail_exponents"]:
                angle = perihelion_first_order(
                    orbit, a0, lambda y, p=exponent: power_tail(y, p, config["power_tail_coefficient"]))
                value = mas_per_century(angle, orbit, century)
                tails.append({"exponent": exponent, "mas_per_century": value,
                              "inside_published_interval": interval[0] <= value <= interval[1],
                              "full_local_gravity_pass": False})
            # The first-order unregularized law scales exactly as sqrt(a0).
            # Since its rate is negative, compare to the negative interval edge.
            scale_ceiling = a0 * (interval[0] / mas_per_century(analytic, orbit, century))**2
            rows.append({
                "planet": planet["name"], "a0_m_s2": a0,
                "perihelion_m": peri, "aphelion_m": apo,
                "max_fractional_acceleration_anomaly": max_fraction,
                "first_order_mas_per_century": rate,
                "closed_log_mas_per_century": mas_per_century(analytic, orbit, century),
                "binet_log_mas_per_century": mas_per_century(direct, orbit, century),
                "quadrature_refinement_relative_change": abs(first[-1] / first[0] - 1),
                "analytic_quadrature_relative_difference": abs(first[-1] / analytic - 1),
                "binet_first_order_relative_difference": abs(direct / analytic - 1),
                "published_interval_mas_per_century": interval,
                "inside_published_interval": interval[0] <= rate <= interval[1],
                "magnitude_over_max_absolute_interval_edge": abs(rate) / max(abs(x) for x in interval),
                "unregularized_first_order_a0_ceiling_m_s2": scale_ceiling,
                "tail_diagnostics": tails,
            })
    certificates = []
    for spec in generate_specs(candidate_config["grammar"]):
        supported = spec.family in {"qumond", "trimond_alignment"}
        matches = spec.epsilon == epsilon
        status = ("UNSUPPORTED_LENGTH_SENSITIVE_SPHERICAL_SOLUTION" if not supported else
                  "UNTESTED_REGULARIZER_MISMATCH" if not matches else
                  "EXCEEDS_PUBLISHED_MONOPOLE_SCREEN" if any(not r["inside_published_interval"] for r in rows)
                  else "WITHIN_MONOPOLE_SCREEN_ONLY")
        certificates.append({"card": spec.card(), "status": status,
                             "branch": config["scope"]["spherical_branch"],
                             "full_solar_system_pass": False})
    return {"schema": config["schema"], "claim_ceiling": "ISOLATED_MONOPOLE_DEVELOPMENT_SCREEN_NOT_EPHEMERIS_FIT",
            "rows": rows, "candidate_dispositions": certificates, "configuration": config,
            "raw_observation_files_opened": 0, "sealed_products_opened": 0,
            "new_historical_novelty_claim": False,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gravity_local_limit_audit_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    try:
        config = json.loads(args.config.read_text())
        candidate_path = ROOT / config["candidate_config"]
        paths = [args.config, candidate_path, Path(__file__),
                 ROOT / "src/invariant_gravity_extensions/actions.py",
                 ROOT / "src/invariant_gravity_extensions/local_limits.py"]

        def hashes():
            return {p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p):
                    sha256(p.read_bytes()).hexdigest() for p in paths}

        before = hashes()
        write("started.json", {"inputs_sha256": before, "claim_ceiling": "ISOLATED_MONOPOLE_SCREEN"})
        candidate_config = json.loads(candidate_path.read_text())
        # Detect a configuration race before computation as well as after it.
        if json.loads(args.config.read_text()) != config:
            raise RuntimeError("configuration changed during loading")
        result = calculate(config, candidate_config)
        if hashes() != before:
            raise RuntimeError("dependencies changed during run")
        result["inputs_sha256"] = before
        write("result.json", result)
        digest = sha256((args.output / "result.json").read_bytes()).hexdigest()
        write("receipt.json", {"status": "COMPLETED_AT_DECLARED_SCOPE", "result_sha256": digest,
                               "full_solar_system_pass": False})
        print(json.dumps({"rows": len(result["rows"]), "result_sha256": digest,
                          "outside_interval": sum(not r["inside_published_interval"] for r in result["rows"])}))
        return 0
    except Exception as exc:
        write("failure.json", {"status": "EXECUTION_FAILURE_NOT_THEORY_REJECTION", "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
