import numpy as np
import pytest

from invariant_gravity_extensions.galaxy_development import (
    Geometry,
    GridCachedSource,
    losses,
    paired_influence,
)
from invariant_gravity_extensions.isolated_axisymmetric import (
    MassComponent,
    MultipoleGrid,
    solve_isolated,
)
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


def test_exact_grid_cache_preserves_full_scalar_solution_and_other_coordinates():
    source = MassComponent('disk', 1, 1, .3)
    grid = MultipoleGrid(1e-3, 100, 129, 32, 12)
    cache = GridCachedSource(source, grid)
    R, z = np.array([.5, 1, 3]), np.array([.1, .4, .2])
    for key in source.fields(R, z):
        np.testing.assert_array_equal(source.fields(R, z)[key], cache.fields(R, z)[key])
    spec = SaturatedActionSpec('qumond', shape=1)
    direct = solve_isolated((source,), spec, .1, grid).evaluate(R, z)['acceleration']
    cached = solve_isolated((cache,), spec, .1, grid).evaluate(R, z)['acceleration']
    np.testing.assert_array_equal(direct, cached)


def test_geometry_and_shared_covariance_have_expected_limits():
    g = Geometry(10, 20, 60, 30)
    np.testing.assert_allclose(g.radii([1, 2]), [2, 4])
    assert g.velocity_factor() == pytest.approx(np.sqrt(3))
    assert g.distance_speed_factor(20) == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError):
        g.velocity_factor(-30)
    observed = np.array([100., 100., 100.])
    predicted = observed+10
    sigma = np.ones(3)
    zero = losses(predicted, observed, sigma, 45, 0)
    shifted = losses(predicted, observed, sigma, 45, 3)
    assert zero['random_error_loss'] == zero['inclination_covariance_loss'] == 100
    assert shifted['inclination_covariance_loss'] < 2
    assert shifted['five_kms_floor_loss'] == pytest.approx(100/26)
    with pytest.raises(ValueError):
        losses(predicted, observed, [1, 0, 1], 45, 3)


def test_influence_detects_a_single_dominating_comparative_residual():
    candidate = np.array([10, 0, 0, 0, 0.])
    comparator = np.full(5, 3.)
    result = paired_influence(candidate, comparator, .2)
    assert result['candidate_minus_RAR_loss'] == 11
    assert result['drop_one_radial_loss_difference'] == -9
    assert result['trimmed_radial_loss_difference'] == -9
    assert result['drop_one_sign_change'] and result['trim_sign_change']
    assert result['single_object_removal'] is None


def test_complete_development_score_recovers_a_manufactured_rotation_curve():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location('galaxy_runner', Path(__file__).resolve().parents[1]/'scripts/run_gravity_ngc3198_development.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    radii = np.array([2., 4., 8., 12., 16.])
    speed = np.array([100., 120., 140., 150., 150.])
    galaxy = {'rows': [[str(r), str(v), '2'] for r, v in zip(radii, speed, strict=True)]}
    fields = {'primary/fine': {'predictions': {'RAR_2016_ALGEBRAIC': speed**2/radii,
                                               'matched': speed**2/radii,
                                               'ten_percent_low': (.9*speed)**2/radii}}}
    config = {'source_variants': [{'id': 'primary'}],
              'geometry': {'distance_offsets_mpc': [-1., 0., 1.], 'inclination_offsets_deg': [-3., 0., 3.],
                           'published_inclination_error_deg': 3.},
              'radial_selection': {'outer_stratum_minimum_kpc': 12.},
              'scoring': {'symmetric_radial_influence_trim_fraction_each_tail': .05}}
    result = runner.score(config, Geometry(10, 10, 60, 60), galaxy, np.arange(5), radii, fields, np.arange(5))
    assert result['rotation_velocity_rows_scored'] == 5
    assert result['individual_velocity_values_converted'] == 10
    assert len(result['scenarios']) == 9
    assert result['summary']['matched']['nominal']['random_error_loss'] == 0
    assert result['summary']['matched']['raw_nominal_worse_than_RAR_galaxies'] == 0
    assert result['summary']['ten_percent_low']['raw_nominal_worse_than_RAR_galaxies'] == 1
    assert result['summary']['ten_percent_low']['nominal']['median_predicted_observed_ratio'] == pytest.approx(.9)
