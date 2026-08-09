from __future__ import annotations

from sigma_theory_compiler.backgrounds import curved_background_principal_controls


def test_curved_background_metric_cone_controls_pass() -> None:
    result = curved_background_principal_controls()
    assert result["passed"], result
    assert result["flrw"]["orthonormal_physical_speed_squared"] == "1"
    assert result["static_spherical"]["orthonormal_physical_speed_squared"] == "1"
    assert result["schwarzschild_exterior"]["orthonormal_physical_speed_squared"] == "1"
    assert result["flrw"]["strong_hyperbolicity"]
    assert result["static_spherical"]["strong_hyperbolicity"]
