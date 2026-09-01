from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item22_polarization_superposition import _backend
from sigma_theory_compiler.gravity_item27_gravitational_memory import (
    _admissible_candidates,
    _base_design,
    _build_sample,
    _build_term_matrix,
    _oof_search,
    _predictor_rows,
    load_config,
)


root = Path.cwd()
config = load_config(root)
predictors, _ = _predictor_rows(root, config)
sample = _build_sample(predictors, config)
roles = {str(row["identity"]): row for row in sample["objects"]}
rows = []
for predictor in predictors:
    role = roles[str(predictor["normalized_identity"])]
    if role["role"] != "exploration":
        continue
    rows.append(
        {
            **predictor,
            "fold": int(role["outer_fold"]),
            "mass_stratum": int(role["mass_stratum"]),
        }
    )
arrays, _ = _admissible_candidates(config)
terms = _build_term_matrix(config, arrays, rows, "primary")
base = _base_design(rows, config, "primary")
coefficient = np.asarray([2.2, 1.0])
base_target = base @ coefficient
folds = np.asarray([int(row["fold"]) for row in rows])
xp, backend, device = _backend()
controls = []
for niche, injection in enumerate(
    config["candidate_generator"]["synthetic_injection_admissible_indices"]
):
    replay = _oof_search(
        xp,
        config,
        rows,
        base_target + terms[int(injection)],
        folds,
        terms,
        "primary",
    )
    selected_niches = [int(arrays["niche"][index]) for index in replay["selected"]]
    controls.append(
        {
            "injected_niche": niche,
            "injected_index": int(injection),
            "selected_indices": replay["selected"],
            "selected_niches": selected_niches,
            "recovered_folds": int(np.count_nonzero(np.asarray(selected_niches) == niche)),
        }
    )
result = {
    "backend": backend,
    "device": device,
    "objects": len(rows),
    "admissible_candidates": len(arrays["niche"]),
    "controls": controls,
}
Path("work/item27-memory-audit/target-blind-controls.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result, indent=2))
