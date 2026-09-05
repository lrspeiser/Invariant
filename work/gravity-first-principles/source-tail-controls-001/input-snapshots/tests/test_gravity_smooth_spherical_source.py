"""Conservation of mass, positive reconstruction and full radial variation."""
import numpy as np
import pytest
from scipy.integrate import quad

from invariant_gravity_extensions.length_screening import (
    LengthScreening,
    anomalous_flux,
    point_monopole_delta,
)
from invariant_gravity_extensions.smooth_spherical_source import (
    log_normal_interval,
    smooth_cumulative_mass,
    smooth_power_density,
    spherical_acceleration_derivatives,
    spherical_length_anomaly,
)


def test_far_normal_tails_remain_finite_and_symmetric():
    a = np.array([-1000.,-41,-2,1,39])
    b = a+.01
    values = log_normal_interval(a,b)
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values, log_normal_interval(-b,-a), rtol=1e-12)


def test_smooth_cumulative_mass_matches_independent_gaussian_convolution():
    r, M, width = np.array([1.,2.,3.,5.]), np.array([1.,1.7,1.4,4.]), .04
    model = smooth_cumulative_mass(r,M,width=width,nodes=4097)
    monotone = np.maximum.accumulate(M)

    def old(radius):
        return radius**3 if radius < 1 else np.interp(radius,r,monotone)

    for radius in [.3,1,1.9,2.5,3,4.9,7]:
        breaks = sorted(v for v in np.log(r/radius)/width if -12 < v < 12)
        expected = quad(lambda z, probe=radius: old(probe*np.exp(width*z))*np.exp(-z*z/2)/np.sqrt(2*np.pi),
                        -12,12,points=breaks,epsabs=1e-11,epsrel=1e-11)[0]
        assert model.evaluate(radius)['mass'] == pytest.approx(expected,rel=2e-7,abs=1e-10)
    assert model.cumulative_mass[-1] == pytest.approx(4,rel=2e-7)
    assert model.metadata['inherited_monotonic_corrections'] == 1


def test_density_and_derivative_describe_same_continuous_mass():
    model = smooth_cumulative_mass([1,2,3,5],[1,1.7,1.7,4],width=.03,nodes=4097)
    r = np.array([.7,1,1.3,2,2.6,3,4,5])
    delta = 1e-5
    mid = model.evaluate(r)
    lo, hi = model.evaluate(r*np.exp(-delta)), model.evaluate(r*np.exp(delta))
    D = 4*np.pi*r**3*mid['density']
    np.testing.assert_allclose((hi['mass']-lo['mass'])/(2*delta),D,rtol=2e-6,atol=1e-11)
    np.testing.assert_allclose((hi['density']-lo['density'])/(2*delta*r),mid['density_gradient'],rtol=2e-6,atol=1e-10)
    assert np.all(mid['density'] >= 0)


def test_gas_total_mass_and_homologous_scaling():
    r = np.array([1.,2.,4.,8.])
    rho = r**-2
    a = smooth_power_density(r,rho,width=.02,nodes=4097)
    b = smooth_power_density(3*r,2*rho,width=.02,nodes=4097)
    probes = np.geomspace(.1,8,97)
    first, second = a.evaluate(probes), b.evaluate(3*probes)
    np.testing.assert_allclose(second['mass'],54*first['mass'],rtol=1e-12)
    np.testing.assert_allclose(second['density'],2*first['density'],rtol=1e-12)
    np.testing.assert_allclose(second['density_gradient'],2/3*first['density_gradient'],rtol=1e-10,atol=1e-10)
    assert a.cumulative_mass[-1] == pytest.approx(a.expected_total_mass,rel=2e-7)


def test_radial_extended_force_matches_cartesian_action():
    r = np.geomspace(.05,20,129)
    gm,b,ell,a0 = 2,.7,.3,.2
    g = gm*r/(r*r+b*b)**1.5
    first = gm*(b*b-2*r*r)/(r*r+b*b)**2.5
    second = 3*gm*r*(2*r*r-3*b*b)/(r*r+b*b)**3.5
    p = np.array([g,np.zeros_like(g),np.zeros_like(g)])
    H = np.zeros((3,3,len(r)))
    H[0,0],H[1,1],H[2,2] = first,g/r,g/r
    dh = np.array([2*first*second+4*g/r*(first/r-g/r**2),np.zeros_like(g),np.zeros_like(g)])
    dlap = np.array([second+2*first/r-2*g/r**2,np.zeros_like(g),np.zeros_like(g)])
    spec = LengthScreening(2)
    ref = anomalous_flux(spec,p,H,dh,dlap,ell,a0)[0]
    actual = spherical_length_anomaly(spec,r,g,first,second,ell,a0)
    np.testing.assert_allclose(actual,ref,rtol=2e-12,atol=1e-14)
    np.testing.assert_allclose(spherical_length_anomaly(spec,r,1/r**2,-2/r**3,6/r**4,ell,1),
                               point_monopole_delta(spec,1/r**2,ell)/r**2,rtol=1e-10,atol=1e-14)


def test_acceleration_derivatives_match_plummer_density():
    r=np.geomspace(.1,10,80)
    mass=r**3/(r*r+1)**1.5
    rho=3/(4*np.pi)*(1+r*r)**-2.5
    drho=-5*r/(1+r*r)*rho
    g,d1,d2=spherical_acceleration_derivatives(r,mass,rho,drho,1)
    np.testing.assert_allclose(g,r/(1+r*r)**1.5)
    np.testing.assert_allclose(d1,(1-2*r*r)/(1+r*r)**2.5,atol=1e-15)
    np.testing.assert_allclose(d2,3*r*(2*r*r-3)/(1+r*r)**3.5,atol=1e-14)


def test_invalid_or_outside_source_fails_explicitly():
    with pytest.raises(ValueError):
        smooth_cumulative_mass([2,1],[1,2])
    model=smooth_cumulative_mass([1,2],[1,2])
    with pytest.raises(ValueError):
        model.evaluate(100)
