"""run_all.py -- the Principle Synthesis Lane (Run BK), in order.

    1. register the run (reads registry.json: the ONE read outside the lane,
       made BEFORE the provenance guard is armed);
    2. arm the guard (universes/provenance.py, lane root = this directory);
    3. tensor_family   -> tensor_results.json
    4. path_family     -> path_results.json
    5. compile_families -> compile_results.json   (needs 3, 4)
    6. cards           -> cards.json               (needs 3, 4, 5)
    7. render_report   -> REPORT.md, card_tensor.md, card_path.md

`baseline_verdicts_prepatch.json` was written by hand-run code BEFORE the
compiler patch (see REPORT.md section 3) and is compared against in step 5.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "registry"))

if __name__ == "__main__":
    import registry as R
    R.register(
        "BK-synthesis", "synthesis",
        depends_on=["identifiability_gate", "holdout_seal", "confirmation_status"],
        outputs=["work/wellnet-2026-09/synthesis/cards.json",
                 "work/wellnet-2026-09/synthesis/compile_results.json",
                 "work/wellnet-2026-09/synthesis/tensor_results.json",
                 "work/wellnet-2026-09/synthesis/path_results.json",
                 "work/wellnet-2026-09/synthesis/REPORT.md"],
        note=("Principle Synthesis Lane (Run BJ.7). Theory construction only; "
              "opens NO observational data (asserted mechanically)."))
    import guard
    guard.arm()
    import tensor_family
    tensor_family.main()
    import path_family
    path_family.main()
    import compile_families
    compile_families.main()
    import cards
    cards.main()
    import render_report
    render_report.main()
