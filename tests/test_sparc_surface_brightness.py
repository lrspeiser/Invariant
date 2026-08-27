"""Controls for the exploration-only SPARC surface-brightness supplement."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.sparc_surface_brightness import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_SHA256,
    SparcSurfaceBrightnessError,
    load_asset,
    validate_asset,
)

ROOT = Path(__file__).resolve().parents[1]


def test_source_archive_identity_is_frozen() -> None:
    assert EXPECTED_ARCHIVE_BYTES == 119_649
    assert EXPECTED_ARCHIVE_SHA256 == (
        "5e131528f5906e52f305f85d89ca7ea34b21d45fca0e7167d26ddc0c3e893227"
    )


def test_checked_supplement_is_exploration_only() -> None:
    asset = load_asset(ROOT)
    assert asset["counts"] == {
        "confirmation_galaxies": 0,
        "confirmation_rows": 0,
        "exploration_galaxies": 139,
        "exploration_rows": 2720,
    }
    assert len(asset["galaxies"]) == 139
    assert all(len(pair) == 2 for row in asset["galaxies"] for pair in row["rows"])


def test_supplement_tamper_fails_closed() -> None:
    asset = copy.deepcopy(load_asset(ROOT))
    asset["galaxies"][0]["rows"][0][0] = "999.99"
    asset.pop("content_sha256")
    asset["content_sha256"] = canonical_sha256(asset)
    with pytest.raises(SparcSurfaceBrightnessError, match="galaxy seal changed"):
        validate_asset(asset, root=ROOT)
