"""Isolate CDF approximation in a separate process; change no on-disk primitive."""
import hashlib
import json
import numpy as np
from scipy.special import ndtr
import mond_atlas_cube
from mond_atlas_force_emission_checks import ROOT, OUT, reference
from mond_atlas_force_emission import Projection, render
from mond_atlas_pressure_support import gaussian_column
from mond_atlas_motion_controls import Instrument

r = np.linspace(.15, 3.75, 18)
flux = np.exp(-r)*r*.2
instrument = Instrument(npix=25, pixel_kpc=.4, beam_sigma_kpc=.45,
                        beam_half_width=4, nchannel=31)
projection = Projection(61, 27, 11)
original = mond_atlas_cube.gaussian_cdf
diagnostics = []
for support in (0., 9.):
    column = gaussian_column(r, 1.2, support)
    approximate = render(column, 625*r, flux, projection, 7., instrument)[0]
    # An explicit diagnostic substitution in this process only, restored below.
    mond_atlas_cube.gaussian_cdf = ndtr
    try:
        precise = render(column, 625*r, flux, projection, 7., instrument)[0]
    finally:
        mond_atlas_cube.gaussian_cdf = original
    oracle = reference(r, flux, 625, support, 1.2, projection, 7., instrument, 128)[0]
    diagnostics.append(dict(support_km_s=support,
        original_relative_error=float(np.linalg.norm(approximate-oracle)/np.linalg.norm(oracle)),
        ndtr_substitution_relative_error=float(np.linalg.norm(precise-oracle)/np.linalg.norm(oracle)),
        error_explained_by_cdf=float(np.linalg.norm((approximate-oracle)-(approximate-precise))/np.linalg.norm(oracle))))
assert max(d['ndtr_substitution_relative_error'] for d in diagnostics)<1e-11
result = dict(status='CDF_APPROXIMATION_ISOLATED_NOT_PRODUCTION_REPAIR',
              original_failed_checks_unchanged=True, on_disk_primitive_changes=0,
              diagnostics=diagnostics,
              script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
              primitive_sha256=hashlib.sha256((ROOT/'scripts/mond_atlas_cube.py').read_bytes()).hexdigest())
(OUT/'precision-diagnosis.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result))
