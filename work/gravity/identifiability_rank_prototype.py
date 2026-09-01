from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from sigma_theory_compiler import gravity_cluster_comparator_suite as comparators
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59


def log_predictions(
    unit: np.ndarray,
    packets: list[dict[str, object]],
    family: dict[str, object],
    config: dict[str, object],
    config59: dict[str, object],
    row_ids: list[str],
) -> np.ndarray:
    _log_likelihood, predictions = uncertainty._evaluate_unit(
        unit, packets, family, config, config59
    )
    return np.log(np.asarray([predictions[row_id] for row_id in row_ids]))


def jacobian(
    unit: np.ndarray,
    packets: list[dict[str, object]],
    family: dict[str, object],
    config: dict[str, object],
    config59: dict[str, object],
    row_ids: list[str],
    step: float,
) -> np.ndarray:
    result = np.empty((len(row_ids), len(unit)))
    for dimension in range(len(unit)):
        low = unit.copy()
        high = unit.copy()
        low[dimension] -= step
        high[dimension] += step
        result[:, dimension] = (
            log_predictions(high, packets, family, config, config59, row_ids)
            - log_predictions(low, packets, family, config, config59, row_ids)
        ) / (2.0 * step)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config = uncertainty.load_config(root)
    config59 = item59.load_config(root)
    packets = comparators._development_packets(root, config59)
    family = config["candidate_and_control_families"][0]
    row_ids = [
        str(row["row_id"])
        for row in item59._rows(packets, "development_train")
    ]
    engine = qmc.Sobol(d=len(uncertainty.PARAMETERS), scramble=True, seed=596001)
    anchors = 0.2 + 0.6 * engine.random_base2(m=4)
    summaries = []
    aggregate = np.zeros((len(row_ids) * len(anchors), len(uncertainty.PARAMETERS)))
    for anchor_index, anchor in enumerate(anchors):
        matrix = jacobian(anchor, packets, family, config, config59, row_ids, 1e-5)
        aggregate[
            anchor_index * len(row_ids) : (anchor_index + 1) * len(row_ids)
        ] = matrix
        _left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        summaries.append(
            {
                "anchor": anchor_index,
                "singular_values": singular.tolist(),
                "rank_relative_1e_8": int(np.sum(singular > singular[0] * 1e-8)),
                "rank_relative_1e_6": int(np.sum(singular > singular[0] * 1e-6)),
            }
        )
        if anchor_index == 0:
            summaries[-1]["near_null_vectors"] = [
                {
                    parameter: float(value)
                    for parameter, value in zip(
                        uncertainty.PARAMETERS, vector, strict=True
                    )
                    if abs(float(value)) > 0.05
                }
                for vector in right[-7:]
            ]
    singular = np.linalg.svd(aggregate, compute_uv=False)
    print(
        json.dumps(
            {
                "rows_per_anchor": len(row_ids),
                "anchors": len(anchors),
                "evaluations": 2 * len(uncertainty.PARAMETERS) * len(anchors),
                "aggregate_singular_values": singular.tolist(),
                "aggregate_rank_relative_1e_8": int(
                    np.sum(singular > singular[0] * 1e-8)
                ),
                "aggregate_rank_relative_1e_6": int(
                    np.sum(singular > singular[0] * 1e-6)
                ),
                "per_anchor": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
