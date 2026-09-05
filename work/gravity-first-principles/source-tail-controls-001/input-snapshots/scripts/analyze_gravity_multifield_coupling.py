"""Post-scan algebraic coupling bounds from fixed external-quadrupole solutions.

These are conditional compatibility intervals, not fitted constants, confidence
intervals, proof of all-parameter exclusion or a new confirmation experiment.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"work/gravity-first-principles/multifield-external-002/result.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = json.loads(args.input.read_bytes())
    groups = defaultdict(list)
    for row in result["rows"]:
        if row["mixing"] == result["config"]["grammar"]["mixing"][0]:
            groups[(row["shape"], row["a0_m_s2"], row["beta"], row["power"])].append(row)
    lower, upper = result["summary_interval_Q2_s_minus2"]
    rows = []
    for (shape, a0, beta, power), background_rows in sorted(groups.items()):
        scenario_bounds = []
        for row in background_rows:
            if not row["numerical_controls_pass"]:
                raise RuntimeError("Unresolved numerical input cannot define a coupling bound")
            coefficient = row["auxiliary_Q2_s_minus2"]/row["mixing"]**2
            if coefficient <= 0:
                raise RuntimeError("This diagnostic requires the observed positive coefficient; use a new derivation for another sign")
            base = row["scalar_Q2_s_minus2"]
            max_square = (upper-base)/coefficient
            min_square = max(0, (lower-base)/coefficient)
            scenario_bounds.append({"physical_external_m_s2": row["physical_external_m_s2"],
                                    "scalar_Q2_s_minus2": base, "coefficient_of_mixing_squared_s_minus2": coefficient,
                                    "squared_coupling_lower": min_square, "squared_coupling_upper": max_square,
                                    "empty": max_square < min_square})
        lo = max(r["squared_coupling_lower"] for r in scenario_bounds)
        hi = min(r["squared_coupling_upper"] for r in scenario_bounds)
        rows.append({"shape": shape, "a0_m_s2": a0, "beta": beta, "power": power,
                     "backgrounds": scenario_bounds, "intersection_empty": hi < lo,
                     "absolute_mixing_interval": None if hi < lo else [float(np.sqrt(lo)), float(np.sqrt(hi))],
                     "interpretation": "Intersection requires compatibility for both assumed backgrounds; they are scenarios, not independent observations. No certified numerical error or statistical confidence is assigned to the endpoints."})
    document = {"created_utc": datetime.now(UTC).isoformat(), "scope": "POST_SCAN_CONDITIONAL_ALGEBRAIC_DIAGNOSTIC",
                "source": args.input.relative_to(ROOT).as_posix(), "source_sha256": sha256(args.input.read_bytes()).hexdigest(),
                "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "identity": "Q2(lambda)=Q2(0)+lambda^2*Q2_aux(lambda=1); eta_N,beta,power,shape,a0 and external background fixed",
                "scan_parameters_refitted": False, "raw_observations_accessed": False, "rows": rows,
                "total_parameter_groups": len(rows), "empty_intersections": sum(r["intersection_empty"] for r in rows),
                "full_solar_system_pass": False, "discovery_claim": False}
    with (args.output/"result.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(document, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    (args.output/"script-snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"groups": len(rows), "empty": document["empty_intersections"],
                      "nonempty_absolute_mixing_upper_range": [min(r["absolute_mixing_interval"][1] for r in rows if not r["intersection_empty"]), max(r["absolute_mixing_interval"][1] for r in rows if not r["intersection_empty"])]}))


if __name__ == "__main__":
    main()
