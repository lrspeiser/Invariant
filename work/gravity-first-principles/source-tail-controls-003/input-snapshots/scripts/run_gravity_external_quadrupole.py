"""Append-only external-field screen with published and independent controls."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from invariant_gravity_extensions.external_quadrupole import (
    newtonian_external_ratio,
    q_to_Q2,
    quadrupole_integrals,
    reference_nu_delta,
    reference_nu_derivative,
    scalar_quadrupole,
)
from invariant_gravity_extensions.saturated_actions import (
    SaturatedActionSpec,
    generate_saturated_specs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "configs/gravity_external_quadrupole_v1.json"
    cards_path = ROOT / "configs/gravity_saturated_actions_v1.json"
    paths = [Path(__file__), config_path, cards_path,
             *sorted((ROOT / "src/invariant_gravity_extensions").glob("*.py"))]

    def hashes():
        return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}

    def write(name, value):
        with (args.output / name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    before = hashes()
    config = json.loads(config_path.read_bytes())
    provenance = {
        "input_hashes": before,
        "started_utc": datetime.now(UTC).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes_authoritative": True,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
    }
    write("started.json", {"config": config, **provenance})
    try:
        reference = []
        for control in config["reference_controls"]:
            alpha = control["alpha"]
            delta = lambda y, a=alpha: reference_nu_delta(y, a)
            derivative = lambda y, a=alpha: reference_nu_derivative(y, a)
            eta_n = newtonian_external_ratio(control["eta_physical"], delta)
            value = quadrupole_integrals(eta_n, delta, derivative, nodes=256)
            passed = (value["q_milgrom"] < 0 and
                      abs(-value["q_milgrom"]-control["q_magnitude"]) < control["rounding_tolerance"] and
                      value["absolute_agreement"] < 3e-6)
            reference.append({**control, **value, "passed": passed})
        rows = []
        limits = config["numerical_controls"]
        cassini = config["cassini_summary"]
        center = cassini["mean_Q2_s_minus2"]
        sigma = cassini["one_sigma_s_minus2"]
        width = cassini["screen_sigma_multiplier"]*sigma
        gm = config["gm_sun_m3_s2"]
        for shape in config["shapes"]:
            spec = SaturatedActionSpec("qumond", shape=shape, epsilon=config["epsilon"])
            for a0 in config["a0_m_s2"]:
                for external in config["physical_external_m_s2"]:
                    estimates = [{"nodes": n, **scalar_quadrupole(spec, external, a0, gm, nodes=n)}
                                 for n in config["quadrature_nodes"]]
                    high = estimates[-1]
                    epsilon_checks = [{"epsilon": eps, **scalar_quadrupole(
                        SaturatedActionSpec("qumond", shape=shape, epsilon=eps),
                        external, a0, gm, nodes=high["nodes"])} for eps in config["epsilon_sensitivity"]]
                    refinement = abs(high["q_milgrom"]-estimates[-2]["q_milgrom"])
                    sensitivity = max(abs(r["q_milgrom"]-high["q_milgrom"]) for r in epsilon_checks)
                    numerical_ok = (refinement < limits["max_last_refinement_q_change"] and
                                    high["absolute_agreement"] < limits["max_source_hessian_q_disagreement"] and
                                    sensitivity < limits["max_epsilon_sensitivity_q_change"])
                    empirical_q_spread = max(refinement, sensitivity, high["absolute_agreement"])
                    empirical_Q2_spread = abs(q_to_Q2(empirical_q_spread, a0, gm))
                    q2 = high["Q2_s_minus2"]
                    distance_to_edge = min(abs(q2-(center-width)), abs(q2-(center+width)))
                    if not numerical_ok:
                        status = "NUMERICALLY_UNRESOLVED"
                    elif distance_to_edge <= empirical_Q2_spread:
                        status = "NEAR_SCREEN_EDGE_REQUIRES_REFINEMENT"
                    elif abs(q2-center) <= width:
                        status = "WITHIN_DECLARED_SUMMARY_SCREEN"
                    else:
                        status = "OUTSIDE_DECLARED_SUMMARY_SCREEN"
                    rows.append({"shape": shape, "a0_m_s2": a0, "physical_external_m_s2": external,
                                 "Q2_s_minus2": q2, "summary_standardized_offset": (q2-center)/sigma,
                                 "status": status, "quadrature": estimates,
                                 "epsilon_sensitivity": epsilon_checks,
                                 "last_refinement_q_change": refinement,
                                 "max_epsilon_sensitivity_q_change": sensitivity,
                                 "empirical_Q2_spread_not_error_bound": empirical_Q2_spread,
                                 "numerical_controls_passed": numerical_ok})
        cards = generate_saturated_specs(json.loads(cards_path.read_bytes())["grammar"])
        applicability = [{"card_sha256": s.card()["content_sha256"], "family": s.family,
                          "status": ("SCALAR_SCENARIO_RESULTS_AVAILABLE" if s.family == "qumond"
                                     else "UNSUPPORTED_EXTERNAL_FIELD_SOLVER")}
                         for s in cards]
        if hashes() != before:
            raise RuntimeError("inputs changed during run")
        summary = {"scenarios": len(rows), "counts": dict(Counter(r["status"] for r in rows)),
                   "reference_controls_passed": all(r["passed"] for r in reference),
                   "numerical_controls_passed": all(r["numerical_controls_passed"] for r in rows),
                   "max_last_refinement_q_change": max(r["last_refinement_q_change"] for r in rows),
                   "max_source_hessian_q_disagreement": max(r["quadrature"][-1]["absolute_agreement"] for r in rows),
                   "max_epsilon_sensitivity_q_change": max(r["max_epsilon_sensitivity_q_change"] for r in rows)}
        write("result.json", {"config": config, **provenance, "summary": summary,
                              "reference_controls": reference, "rows": rows,
                              "card_applicability": applicability,
                              "discovery_claim": False, "full_solar_system_pass": False,
                              "all_family_parameters_falsified": False})
        if not summary["reference_controls_passed"] or not summary["numerical_controls_passed"]:
            raise RuntimeError("numerical controls failed; inspect preserved result, not a theory rejection")
        write("receipt.json", {"status": "COMPLETED_AT_DECLARED_SCOPE",
                               "result_sha256": sha256((args.output/"result.json").read_bytes()).hexdigest()})
        print(json.dumps(summary))
    except Exception as exc:
        write("failure.json", {"status": "EXECUTION_FAILURE_NOT_PHYSICAL_REJECTION", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
