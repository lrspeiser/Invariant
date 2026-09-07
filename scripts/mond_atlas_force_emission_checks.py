"""Independent manufactured integration controls; no observation reads."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.signal import convolve2d
from mond_atlas_force_emission import Projection, emitters, render
from mond_atlas_pressure_support import gaussian_column
from mond_atlas_motion_controls import Instrument

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'work/gravity-first-principles/mond-atlas-force-emission-001'


def reference(r, flux, omega2, pressure_dispersion, density_scale, projection,
              tracer_dispersion, instrument, nphi):
    # Analytic harmonic/Gaussian closure and independent 3D matrix rotations.
    inc, pa = np.deg2rad([projection.inclination_deg, projection.position_angle_deg])
    rx = np.array([[1, 0, 0], [0, np.cos(inc), -np.sin(inc)],
                   [0, np.sin(inc), np.cos(inc)]])
    rz = np.array([[np.cos(pa), -np.sin(pa), 0],
                   [np.sin(pa), np.cos(pa), 0], [0, 0, 1]])
    h, n = instrument.beam_half_width, instrument.npix
    size = n+2*h
    grid = np.zeros((instrument.nchannel, size, size))
    positions, velocities = [], []
    for radius, weight in zip(r, flux):
        speed = radius*np.sqrt(omega2-(pressure_dispersion/density_scale)**2)
        for j in range(nphi):
            theta = (j+.5)*2*np.pi/nphi
            position = rz@rx@np.array([radius*np.cos(theta), radius*np.sin(theta), 0])
            velocity = rz@rx@np.array([-speed*np.sin(theta), speed*np.cos(theta), 0])
            positions.append(position)
            velocities.append(velocity[2]+projection.systemic_km_s)
            spectrum = np.diff(ndtr((instrument.edges-velocities[-1])/tracer_dispersion))*weight/nphi
            u, v = position[:2]/instrument.pixel_kpc+(n-1)/2+h
            ix, iy = int(np.floor(u)), int(np.floor(v))
            for yy in (iy, iy+1):
                for xx in (ix, ix+1):
                    if 0 <= yy < size and 0 <= xx < size:
                        grid[:, yy, xx] += spectrum*(1-abs(u-xx))*(1-abs(v-yy))
    a = np.arange(-h, h+1)*instrument.pixel_kpc
    if instrument.beam_sigma_kpc > 0:
        beam = np.exp(-(a[:, None]**2+a[None, :]**2)/(2*instrument.beam_sigma_kpc**2))
        beam /= beam.sum()
    else:
        beam = np.zeros((2*h+1, 2*h+1)); beam[h, h] = 1
    cube = np.array([convolve2d(channel, beam, mode='same') for channel in grid])
    return cube[:, h:h+n, h:h+n], np.array(positions), np.array(velocities)


def run():
    checks = []
    def check(name, value, tolerance, passed=None):
        checks.append(dict(name=name, value=float(value), tolerance=tolerance,
                           passed=bool(value <= tolerance if passed is None else passed)))
    r = np.linspace(.15, 3.75, 18)
    flux = np.exp(-r)*r*.2
    omega2, scale = 625., 1.2
    instrument = Instrument(npix=25, pixel_kpc=.4, beam_sigma_kpc=.45,
                            beam_half_width=4, nchannel=31)
    projection = Projection(61, 27, 11)
    for support in (0., 9.):
        column = gaussian_column(r, scale, support)
        cube, ledger, balance = render(column, omega2*r, flux, projection, 7., instrument)
        oracle, xyz, los = reference(r, flux, omega2, support, scale, projection, 7., instrument, 128)
        p = emitters(column, omega2*r, flux, projection)
        check(f'support_{support}_speed', np.max(abs(balance.speed()-r*np.sqrt(omega2-(support/scale)**2))), 1e-12)
        check(f'support_{support}_positions', np.max(abs(np.array([p['x'], p['y'], p['depth']]).T-xyz)), 1e-12)
        check(f'support_{support}_los', np.max(abs(p['los']-los)), 1e-12)
        check(f'support_{support}_independent_cube', np.linalg.norm(cube-oracle)/np.linalg.norm(oracle), 1e-11)
        check(f'support_{support}_flux_ledger', abs(cube.sum()+ledger['total_loss']-flux.sum()), 1e-12)
    column = gaussian_column(r, scale, 9.)
    cube, ledger, balance = render(column, omega2*r, flux, projection, 7., instrument)
    scaled = render(column, omega2*r, 3*flux, projection, 7., instrument)[0]
    check('emission_linearity', np.linalg.norm(scaled-3*cube)/np.linalg.norm(cube), 1e-12)
    wide, _, wide_balance = render(column, omega2*r, flux, projection, 14., instrument)
    check('linewidth_does_not_change_pressure_balance', np.max(abs(wide_balance.speed()-balance.speed())), 0)
    check('linewidth_changes_spectrum', np.linalg.norm(wide-cube), None, np.linalg.norm(wide-cube)>0)
    face = emitters(column, omega2*r, flux, Projection(0, 27, 11))
    check('face_on_systemic', np.max(abs(face['los']-11)), 1e-12)
    fine = render(column, omega2*r, flux, projection, 7., instrument, 256)[0]
    check('angular_refinement', np.linalg.norm(cube-fine)/np.linalg.norm(fine), .01)
    crop = Instrument(npix=7, pixel_kpc=.4, beam_sigma_kpc=.45, beam_half_width=4,
                      channel_min_km_s=-15, channel_max_km_s=15, nchannel=6)
    cropped, loss, _ = render(column, omega2*r, flux, projection, 7., crop)
    for key in ('spectral_loss', 'total_spatial_loss_after_band'):
        check(key, loss[key], None, loss[key]>0)
    check('cropped_flux_ledger', abs(cropped.sum()+loss['total_loss']-flux.sum()), 1e-12)
    for name, fn in [
        ('impossible_equilibrium', lambda: render(gaussian_column(r, scale, 40.), omega2*r, flux, projection, 7., instrument)),
        ('radial_flow', lambda: emitters(column, omega2*r, flux, projection, radial_flow=1.)),
        ('vertical_flow', lambda: emitters(column, omega2*r, flux, projection, vertical_flow=1.)),
        ('reversed_radii', lambda: emitters(gaussian_column(r[::-1], scale, 9.), omega2*r[::-1], flux, projection)),
    ]:
        try:
            fn(); rejected = False
        except ValueError:
            rejected = True
        check(name+'_rejected', int(rejected), None, rejected)
    bindings = []
    for relative in ['scripts/mond_atlas_force_emission.py', 'scripts/mond_atlas_force_emission_checks.py',
                     'scripts/mond_atlas_pressure_support.py', 'scripts/mond_atlas_motion_controls.py',
                     'scripts/mond_atlas_cube.py', 'work/gravity-first-principles/mond-atlas-force-emission-001/PREFLIGHT.md']:
        bindings.append(dict(path=relative, sha256=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()))
    result = dict(status='THEORY_BENCHMARK_ONLY', all_pass=all(c['passed'] for c in checks),
                  observed_response_reads=0, checks=checks, cropped_flux_ledger=loss, bindings=bindings)
    destination = OUT/'run001'; destination.mkdir(exist_ok=False)
    (destination/'checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
    if not result['all_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    run()
