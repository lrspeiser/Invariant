from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item39_holographic_boundary import load_config
from sigma_theory_compiler.gravity_item39_holographic_boundary_lensing import (
    _nominal_error,
    _normalize_id,
    _plain_value,
    _sersic_fraction_and_growth,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item39_swells_transfer_is_unchanged_and_never_selects_on_lensing() -> None:
    transfer = load_config(ROOT)["lensing_transfer"]
    assert transfer["dynamics_result_commit"].startswith("738bcdd6")
    assert transfer["selected_candidate"]["candidate_id"] == 173808
    assert transfer["selection_use"] is False
    assert transfer["lensing_only_retuning"] is False
    assert transfer["post_selection_candidate_cells"] == 0
    assert transfer["paid_model_calls"] == 0
    assert transfer["expected_eligible_names"] == [
        "J0955+0101",
        "J1021+2028",
        "J1111+2234",
        "J1135+3720",
        "J1251-0208",
        "J1331+3638",
    ]


def test_item39_swells_published_table_parser_handles_mathml_duplication() -> None:
    assert _normalize_id("SDSSJ1251−-0208") == "J1251-0208"
    assert _normalize_id("SDSSJ1021++2028") == "J1021+2028"
    assert _nominal_error("10.63±0.1410.63\\pm 0.14") == (10.63, 0.14)
    assert _plain_value("1.621.62") == 1.62
    assert _plain_value("4.15\\phantom{1}4.15") == 4.15


def test_item39_swells_sersic_projection_has_physical_fraction_and_growth() -> None:
    bulge_fraction, bulge_growth = _sersic_fraction_and_growth(2.0, 2.0, 4.0, 7.66924944)
    disk_fraction, disk_growth = _sersic_fraction_and_growth(3.0, 3.0, 1.0, 1.67834699)
    assert np.isclose(bulge_fraction, 0.5, atol=1e-4)
    assert np.isclose(disk_fraction, 0.5, atol=1e-4)
    assert bulge_growth > 0.0
    assert disk_growth > 0.0
