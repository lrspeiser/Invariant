"""A sparse stencil grid must neither lose active offsets nor shift inactive rows."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_sparse_stencil_lookup_preserves_interfaces_and_checks_missing_points():
    p = Path(__file__).resolve().parents[1]/'scripts/verify_gravity_source_tail.py'
    spec = importlib.util.spec_from_file_location('_tail_verifier_stencil_control', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    R, z, h = np.array([0., 1., 100.]), np.array([0.]), .001
    interfaces = np.array([False, True, False])
    rr = np.unique(abs(np.r_[(R[:, None]+h*np.arange(-2, 3)).ravel(),
        (R[interfaces, None]+h*np.array([-4, -3, 3, 4])).ravel()]))
    ri, zi, xr = module.stencil_indices(R, z, rr, z, 0, interfaces, 4, h)
    np.testing.assert_array_equal(rr[ri], [0., 1.004, 100.])
    np.testing.assert_array_equal(xr, [0., 1.004, 100.])
    np.testing.assert_array_equal(zi, [0])
    ri, _, xr = module.stencil_indices(R, z, rr, z, 0, ~interfaces, -2, h)
    assert xr[0] == -.002 and rr[ri[0]] == .002
    with pytest.raises(ValueError, match='required stencil coordinate'):
        module.stencil_indices(R, z, rr, z, 0, np.ones(3, bool), 4, h)
