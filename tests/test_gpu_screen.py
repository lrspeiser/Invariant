from sigma_theory_compiler.gpu_screen import _term_hessian, dense_grid, precompute_dense_hessians


def test_dense_grid_has_343_points():
    grid = dense_grid()
    assert len(grid["d"]) * len(grid["p"]) * len(grid["state"]) == 343


def test_q_term_has_expected_gradient_hessian():
    q = {"px": 0, "pq": 1, "pz": 0, "transform": "Identity"}
    hdd, hdp, hpp = _term_hessian(q, d=1.0, p=0.5, state=1.0)
    assert (hdd, hdp, hpp) == (0.0, 0.0, 2.0)


def test_dense_hessian_tensor_shape():
    basis = [
        {"px": 0, "pq": 1, "pz": 0, "transform": "Identity"},
        {"px": 1, "pq": 0, "pz": 1, "transform": "Saturate"},
    ]
    hessians = precompute_dense_hessians(basis, dense_grid())
    assert hessians.shape == (2, 343, 3)
    assert hessians.dtype.name == "float64"
