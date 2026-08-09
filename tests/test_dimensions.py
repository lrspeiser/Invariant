import pytest

from sigma_theory_compiler.dimensions import (
    Dimension,
    assert_dimensionless_invariants,
    normalized_invariant_dimensions,
)


def test_frozen_invariants_are_dimensionless() -> None:
    dimensions = normalized_invariant_dimensions()
    assert all(dimension.is_dimensionless for dimension in dimensions.values())


def test_dimensionful_atom_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimensionful"):
        assert_dimensionless_invariants({"bad": Dimension(length=1)})

