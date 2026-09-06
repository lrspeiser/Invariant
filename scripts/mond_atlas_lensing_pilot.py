"""CPU-only source ingest for a development SLACS pilot. Never evaluates gravity.

Raw downloads and FITS remain under the private root; output receipts contain
measurements, source hashes, checks, and gaps. Only explicitly selected catalog
columns and archive members are parsed. No mass from lensing is a source input.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import re
import tarfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    """Exclusive-create: a receipt is never overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def within(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if not path.is_relative_to(parent):
        raise ValueError(f'Path outside assigned root: {path}')
    return path


class DownloadCache:
    """Hash-verified immutable cache with a cumulative body-byte budget.

    Successful and partial transfers have receipts. Existing research downloads
    are counted conservatively from their sidecars, plus unknown research files.
    Replays never contact the network. HTTP framing/TLS overhead is not counted.
    """
    def __init__(self, private_root, limit=1_000_000_000, offline=False):
        self.root = Path(private_root).resolve()
        self.raw = self.root / 'raw'
        self.raw.mkdir(parents=True, exist_ok=True)
        self.limit = int(limit)
        if not 0 < self.limit <= 1_000_000_000:
            raise ValueError('Download budget must be positive and <= 1 GB')
        self.offline = offline
        self.used = 0
        known = set()
        for p in self.root.rglob('*.access.json'):
            d = json.loads(p.read_text(encoding='utf-8'))
            self.used += d.get('bytes', 0)
            known.add(p.with_name(p.name.removesuffix('.access.json')))
        # The discovery index used a differently named receipt; counting it twice
        # is conservative. Research scripts/text are also counted conservatively.
        research = self.root / 'research'
        if research.exists():
            self.used += sum(p.stat().st_size for p in research.rglob('*')
                             if p.is_file() and p not in known
                             and not p.name.endswith('.access.json'))
        self.used += sum(p.stat().st_size for p in self.raw.iterdir()
                         if p.is_file() and p not in known
                         and not p.name.endswith('.access.json'))
        if self.used > self.limit:
            raise ValueError('Existing accounted bytes already exceed cap')

    def fetch(self, key, url, max_bytes=200_000_000, expected_sha256=None):
        if Path(key).name != key or key in {'.', '..'}:
            raise ValueError('Cache key must be a basename')
        target = within(self.raw / key, self.raw)
        receipt = target.with_name(target.name + '.access.json')
        if target.exists() or receipt.exists():
            if not target.is_file() or not receipt.is_file():
                raise ValueError(f'Incomplete immutable download: {target}')
            d = json.loads(receipt.read_text(encoding='utf-8'))
            if (d['url'] != url or d.get('status') != 'OK'
                    or sha256(target) != d['sha256']
                    or target.stat().st_size != d['bytes']):
                raise ValueError(f'Cache provenance mismatch: {target}')
            if expected_sha256 and d['sha256'] != expected_sha256:
                raise ValueError('Pinned source hash mismatch')
            return target, d
        if self.offline:
            raise FileNotFoundError(f'Offline source unavailable: {key}')
        remaining = min(int(max_bytes), self.limit - self.used)
        if remaining <= 0:
            raise ValueError('Download cap reached')
        d = dict(url=url, retrieved_utc=utc_now(), bytes=0, status='INCOMPLETE')
        try:
            request = urllib.request.Request(url, headers={
                'User-Agent': 'Invariant-source-provenance-pilot/1.0',
                'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(request, timeout=60) as response:
                d.update(resolved_url=response.url, http_status=response.status,
                         headers=dict(response.headers))
                declared = response.headers.get('Content-Length')
                if declared and int(declared) > remaining:
                    raise ValueError('Content-Length exceeds download budget')
                with target.open('xb') as stream:
                    while d['bytes'] < remaining:
                        block = response.read(min(1024 * 1024, remaining - d['bytes']))
                        if not block:
                            break
                        stream.write(block)
                        d['bytes'] += len(block)
                    else:
                        if declared is None or d['bytes'] != int(declared):
                            raise ValueError('Bounded transfer reached limit without certified EOF')
                if declared and d['bytes'] != int(declared):
                    raise ValueError('Truncated HTTP body')
                d['sha256'] = sha256(target)
                if expected_sha256 and d['sha256'] != expected_sha256:
                    raise ValueError('Pinned source hash mismatch')
                d['status'] = 'OK'
        except Exception as exc:
            d.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            self.used += d['bytes']
            write_json(receipt, d)
        return target, d


def validate_config(config, root=ROOT):
    if config['disposition_before_implementation'] != 'SOURCE_BLOCKED':
        raise ValueError('Ingest cannot admit a scientific source builder')
    if config['observational_scoring_allowed']:
        raise ValueError('No scoring permitted in this pilot')
    p = root / config['historical_sample_manifest']
    if sha256(p) != config['historical_sample_sha256']:
        raise ValueError('Historical sample binding changed')
    historical = {x['name']: x for x in json.loads(p.read_text())['objects']}
    names = [t['name'] for t in config['targets']]
    if not 1 <= len(names) <= 3 or len(set(names)) != len(names):
        raise ValueError('Select 1-3 unique systems')
    for target in config['targets']:
        old = historical[target['name']]
        if old['role'] != 'exploration' or target['historical_role'] != 'exploration':
            raise ValueError('Reserved confirmations cannot be selected')
        if old['sdss'] != target['sdss']:
            raise ValueError('Historical identity mismatch')
    for spec in config['table_sources'].values():
        if set(spec['columns']) & set(config['forbidden_source_columns']):
            raise ValueError('Lens-inferred columns forbidden in source query')
        if spec['identity'] not in spec['columns']:
            raise ValueError('Explicit identity column required')
    return historical


def catalog_url(spec, name):
    # Quoted exact identity criteria are URL-encoded, including literal + signs.
    query = {'-source': spec['catalog'], '-out': ','.join(spec['columns']),
             '-out.max': '10', spec['identity']: '=' + name}
    return 'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?' + urllib.parse.urlencode(query)


def parse_catalog(path, spec, name):
    """Parse VizieR TSV without dropping unit or identity evidence."""
    text = Path(path).read_text(encoding='utf-8-sig')
    if '#ERROR' in text or '#INFO\tError=' in text or '<html' in text.lower():
        raise ValueError('Catalog service returned an error instead of TSV')
    lines = [s for s in text.splitlines() if s.strip() and not s.startswith('#')]
    if len(lines) < 4:
        raise ValueError('Missing VizieR table/row')
    fields = [x.strip() for x in lines[0].split('\t')]
    units = [x.strip() for x in lines[1].split('\t')]
    if fields != spec['columns'] or len(units) != len(fields):
        raise ValueError(f'Unexpected catalog columns: {fields}')
    if not all(re.fullmatch('-+', x.strip()) for x in lines[2].split('\t')):
        raise ValueError('Missing TSV field separator')
    rows = []
    for line in lines[3:]:
        values = [x.strip() for x in line.split('\t')]
        if len(values) != len(fields):
            raise ValueError('Catalog row width mismatch')
        rows.append(dict(zip(fields, values)))
    if len(rows) != 1 or rows[0][spec['identity']] != name:
        raise ValueError('Query must return exactly the selected identity')
    return {'row': rows[0], 'units': dict(zip(fields, units))}


def number(value):
    if value.strip() in {'', '-', '--', '---', '...'}:
        return None
    answer = float(value)
    if not math.isfinite(answer):
        raise ValueError('Nonfinite catalog measurement')
    return answer


def safe_selected_members(archive_path, prefixes):
    """Inventory names only; never extract unknown members or follow links."""
    selected = []
    inventory = []
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            inventory.append({'name': member.name, 'size': member.size,
                              'is_file': member.isfile()})
            if any(p in PurePosixPath(member.name).parts for p in prefixes):
                path = PurePosixPath(member.name)
                if (not member.isfile() or path.is_absolute() or '..' in path.parts
                        or '\\' in member.name or member.size > 100_000_000):
                    raise ValueError('Unsafe selected archive member')
                selected.append(member.name)
    return inventory, selected


def extract_selected(archive_path, prefixes, private_root):
    inventory, selected = safe_selected_members(archive_path, prefixes)
    destination = Path(private_root) / 'selected-legacy'
    paths = []
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            if member.name not in selected:
                continue
            target = within(destination / member.name, destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            content = archive.extractfile(member).read()
            if len(content) != member.size:
                raise ValueError('Truncated archive member')
            expected = hashlib.sha256(content).hexdigest()
            if target.exists():
                if sha256(target) != expected:
                    raise ValueError('Extracted archive member changed')
            else:
                with target.open('xb') as stream:
                    stream.write(content)
            paths.append((member.name, target))
    return inventory, paths


def array_stats(array):
    import numpy as np
    a = np.asarray(array)
    finite = np.isfinite(a)
    data = a[finite]
    return dict(shape=list(a.shape), dtype=str(a.dtype), pixels=int(a.size),
                finite_pixels=int(finite.sum()),
                minimum=float(data.min()) if data.size else None,
                maximum=float(data.max()) if data.size else None,
                median=float(np.median(data)) if data.size else None,
                negative_pixels=int((a < 0).sum()), zero_pixels=int((a == 0).sum()))


def audit_legacy(member_name, path):
    import numpy as np
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdus:
        hdus.verify('exception')
        if len(hdus) != 1 or hdus[0].data is None or hdus[0].data.ndim != 2:
            raise ValueError('Expected a single legacy 2D image')
        h = hdus[0].header
        stats = array_stats(hdus[0].data)
        role = re.search(r'_(subdrz|drzerr|drzcrm|drzftm)(435|814)\.fits$', member_name)
        if not role:
            raise ValueError('Unknown legacy product role')
        return dict(archive_member=member_name, path=str(path.relative_to(ROOT)),
                    sha256=sha256(path), bytes=path.stat().st_size,
                    product_suffix=role.group(1), band_from_filename=role.group(2),
                    header=dict(h), stats=stats,
                    fractional_values=int(((hdus[0].data % 1) != 0).sum()),
                    unit=h.get('BUNIT'), standard_wcs_available=False,
                    calibration_admitted=False,
                    role_status='Filename suggests role; error/mask scaling and polarity not certified.',
                    caveat='RAZERO/DECZERO alone are not a complete celestial WCS. No BUNIT, exposure, filter or photometric calibration keywords. Do not assume subdrz is a calibrated total-light image or drizzle error/mask files define a likelihood.')


def sdss_coordinate(sdss):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    m = re.fullmatch(r'(\d{2})(\d{2})(\d{2}\.\d{2})([+-])(\d{2})(\d{2})(\d{2}\.\d)', sdss)
    if not m:
        raise ValueError('Unexpected SDSS coordinate identity')
    a, b, c, sign, d, e, f = m.groups()
    return SkyCoord(f'{a}h{b}m{c}s', f'{sign}{d}d{e}m{f}s', frame='icrs')


def audit_native(path, target, private_output):
    """Bind an unresampled observed SCI/ERR/DQ rectangle to its full source."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import proj_plane_pixel_scales
    import astropy.units as u
    with fits.open(path, memmap=False) as hdus:
        hdus.verify('exception')
        primary = hdus[0].header
        if primary['TARGNAME'] != target['image_member_prefix']:
            raise ValueError('Native HST identity mismatch')
        if primary['INSTRUME'] != 'ACS' or primary['FILTER2'] != 'F814W':
            raise ValueError('Unexpected native instrument/filter')
        sky = sdss_coordinate(target['sdss'])
        pointing = SkyCoord(primary['RA_TARG'] * u.deg, primary['DEC_TARG'] * u.deg)
        separation = float(sky.separation(pointing).arcsec)
        if separation > 1.0:
            raise ValueError('Native pointing not matched within 1 arcsecond')
        candidates = []
        for ext in hdus:
            if ext.name != 'SCI':
                continue
            ver = ext.header['EXTVER']
            w = WCS(ext.header, hdus)
            pixel = w.all_world2pix([[sky.ra.deg, sky.dec.deg]], 0)[0]
            height, width = ext.data.shape
            if 100 <= pixel[0] < width - 100 and 100 <= pixel[1] < height - 100:
                candidates.append((ver, w, pixel))
        if len(candidates) != 1:
            raise ValueError('Need exactly one image chip containing a complete target ROI')
        ver, w, pixel = candidates[0]
        sci, err, dq = (hdus[key, ver] for key in ['SCI', 'ERR', 'DQ'])
        if sci.data.shape != err.data.shape or sci.data.shape != dq.data.shape:
            raise ValueError('Native SCI/ERR/DQ shape mismatch')
        if sci.header.get('BUNIT') != 'ELECTRONS' or err.header.get('BUNIT') != 'ELECTRONS':
            raise ValueError('Native SCI/ERR unit mismatch')
        x, y = np.rint(pixel).astype(int)
        bounds = dict(x_start=int(x-100), x_stop=int(x+101),
                      y_start=int(y-100), y_stop=int(y+101))
        region = np.s_[y-100:y+101, x-100:x+101]
        arrays = {key: np.array(h.data[region]) for key, h in
                  [('SCI', sci), ('ERR', err), ('DQ', dq)]}
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Nonfinite target pixels')
        if np.any(arrays['ERR'] <= 0) or arrays['DQ'].dtype.kind not in 'iu':
            raise ValueError('Invalid native ERR/DQ arrays')
        points = np.array([[x, y], [x-100, y-100], [x+100, y+100]], dtype=float)
        returned = w.all_world2pix(w.all_pix2world(points, 0), 0, tolerance=1e-8)
        roundtrip = float(np.max(np.abs(returned-points)))
        if roundtrip > 1e-4:
            raise ValueError('Celestial WCS roundtrip failed')
        # No resampling, foreground subtraction, mask threshold, or flux conversion.
        npz = Path(private_output) / (target['name'] + '-native-roi.npz')
        with npz.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(npz) as saved:
            exact = all(np.array_equal(saved[k], arrays[k]) for k in arrays)
        if not exact:
            raise ValueError('ROI byte-value replay failed')
        keys = ['ROOTNAME', 'TARGNAME', 'RA_TARG', 'DEC_TARG', 'DATE-OBS',
                'EXPTIME', 'INSTRUME', 'DETECTOR', 'FILTER1', 'FILTER2',
                'PROPOSID', 'CAL_VER', 'PCTECORR', 'PHOTCORR', 'FLATCORR',
                'CRCORR', 'DARKCORR', 'BIASCORR', 'PFLTFILE', 'DARKFILE',
                'IDCTAB', 'IMPHTTAB']
        skeys = ['BUNIT', 'CCDCHIP', 'PHOTFLAM', 'PHOTPLAM', 'PHOTZPT', 'WCSNAME']
        return dict(name=target['name'], path=str(path.relative_to(ROOT)),
            sha256=sha256(path), bytes=path.stat().st_size,
            primary_header={k: primary.get(k) for k in keys},
            science_header={k: sci.header.get(k) for k in skeys},
            sci_extver=ver, full_chip_shape=list(sci.data.shape),
            catalog_pointing_separation_arcsec=separation,
            target_pixel_zero_based=pixel.tolist(),
            approximate_projection_pixel_scale_arcsec=(proj_plane_pixel_scales(w)*3600).tolist(),
            wcs_roundtrip_max_error_pixel=roundtrip, roi_bounds_zero_based_half_open=bounds,
            roi_path=str(npz.relative_to(ROOT)), roi_sha256=sha256(npz),
            roi_exact_native_values_verified=exact,
            roi_arrays={k: array_stats(a) for k, a in arrays.items()},
            dq_nonzero_pixels=int(np.count_nonzero(arrays['DQ'])),
            dq_unique_values=[int(v) for v in np.unique(arrays['DQ'])],
            fits_embedded_checksums_present=any('CHECKSUM' in h.header for h in hdus),
            pixel_uncertainty='Native ERR in electrons; pipeline error model, not independently validated full covariance.',
            mask='Native integer DQ bits retained without applying a new acceptance mask; no lens/arc/foreground analysis mask.',
            psf=None, independent_arc_positions=None,
            gaps=['PSF kernel and focus/chromatic uncertainty not acquired.',
                  'Single FLT exposure is not cosmic-ray-cleaned or distortion-corrected; forward model must handle native geometry.',
                  'FLT has not received pixel-based CTE correction (PCTECORR is recorded, not assumed COMPLETE).',
                  'No native-to-legacy-pixel or later photometry processing equivalence established.',
                  'No foreground/source separation, light-profile fit, photometric flux measurement or lensing likelihood executed.'])


def normalized_record(target, tables, native, legacy):
    b = tables['bolton_observed']['row']
    if b['SDSS'] != target['sdss']:
        raise ValueError('Spectroscopic identity mismatch')
    plate_fiber = f"{int(b['Plate']):04d}-{b['MJD']}-{int(b['Fiber']):03d}"
    if plate_fiber != target['image_member_prefix']:
        raise ValueError('Spectroscopy-to-image identity mismatch')
    z_lens, z_source = number(b['zFG']), number(b['zBG'])
    sigma, sigma_error = number(b['sigma']), number(b['e_sigma'])
    if not (0 < z_lens < z_source and sigma > 0 and sigma_error > 0):
        raise ValueError('Invalid redshift or dispersion measurement')
    for key in ['sigma', 'e_sigma']:
        if tables['bolton_observed']['units'][key] != 'km/s':
            raise ValueError('Spectroscopy unit changed')
    g = tables['grillo_sps']
    populations = {}
    for key in ['MSalBC', 'MSalM', 'MChaBC', 'MKroM']:
        cols = [key, 'E_'+key, 'e_'+key]
        if any(g['units'][k] != '10+10solMass' for k in cols):
            raise ValueError('Photometric mass unit changed')
        values = [number(g['row'][k]) for k in cols]
        if any(v is None or v <= 0 for v in values):
            raise ValueError('Missing SPS estimate or asymmetric uncertainty')
        populations[key] = dict(value=values[0]*1e10, error_plus=values[1]*1e10,
            error_minus=values[2]*1e10, unit='Msun',
            uncertainty='Published asymmetric interval; conditional on SPS choices; integer catalog rounding retained.')
    photometry = {}
    p = tables['grillo_photometry']
    for band in 'ugriz':
        key = band+'mag'
        if p['units'][key] != 'mag' or p['units']['e_'+key] != 'mag':
            raise ValueError('Photometry unit changed')
        error = number(p['row']['e_'+key])
        if error is None or error <= 0:
            raise ValueError('Missing positive photometric error')
        photometry[band] = dict(value=number(p['row'][key]), rms_error=error, unit='mag_AB')
    return dict(name=target['name'], sdss=target['sdss'],
        disposition='SOURCE_BLOCKED', development_exposed=True,
        field_roles=dict(redshifts='shared_distance_context',
            velocity_dispersion='motion_response_not_baryonic_input',
            grillo_photometry='non_gravitational_stellar_source_constraint',
            grillo_sps='model_dependent_photometric_stellar_source_constraint',
            auger_sps='ancillary_inference_conditioned_on_motion_response',
            native_sci='foreground_plus_lensed_background_pixels; future joint light-source and image-response model must separate these roles',
            native_err_dq='instrument_support_not_validated_likelihood',
            lens_inferred_mass_or_convergence='not_ingested'),
        measurements=dict(z_lens=dict(value=z_lens, uncertainty=None, unit='dimensionless'),
            z_source=dict(value=z_source, uncertainty=None, unit='dimensionless'),
            redshift_uncertainty_gap='No object-level redshift errors supplied in this table; decimal precision is not an error.',
            stellar_velocity_dispersion=dict(value=sigma, rms_error=sigma_error, unit='km/s',
                aperture_diameter_arcsec=3.0, aperture_corrected=False,
                caveat='Bolton 2008 remeasurement with a minimum 5% error floor before catalog rounding; earlier release values differ. Single aperture, not resolved kinematics.'),
            sdss_photometry=photometry,
            sdss_photometry_method='Grillo Table 1: Galactic-extinction-corrected SDSS AB modelMag, r-band profile held fixed across bands; source-light contamination and photometric covariance still need validation.'),
        independent_population_constraints=dict(estimates=populations,
            input='Grillo 2009 total ugriz SED photometry and redshift only.',
            assumptions='Solar metallicity; dust-free BC03/Maraston templates, exponential star-formation history; Salpeter/Chabrier/Kroupa IMF alternatives. Integrated inferred stellar masses, not resolved baryonic truth.',
            dependency_boundary='The paper separately analyzes lensing masses. Only total photometric SPS columns are ingested here; no Einstein-aperture factors, total lens mass, or lens-selected preferred IMF is used.'),
        ancillary_nonindependent_population_constraints=dict(source='Auger 2009 Table 4',
            values=tables['auger_sps']['row'], units=tables['auger_sps']['units'],
            warning='Metallicity prior is conditioned on the lens velocity dispersion. Not an independent source constraint for a dynamics test.'),
        hst_photometry=dict(values=tables['auger_photometry']['row'],
            units=tables['auger_photometry']['units'],
            uncertainty_gap='Object-level magnitude and radius errors not supplied in selected table columns; no substituted error floor.',
            caveat='Profile-derived photometry from later images/reductions. Not claimed to be measured from the acquired 2004 single exposures.'),
        direct_image_constraints=dict(native=native, legacy=legacy,
            role='Observed foreground-plus-background native SCI pixels, with native ERR and DQ. No mass/convergence map or model-derived Einstein radius.',
            multiple_image_positions=None, background_source_reconstruction=None,
            independent_positions_gap='No astrometric multiple-image position/uncertainty table acquired; pixels provide the future direct-image response.'),
        source_table_bindings={k: {'source_sha256': v['source_sha256'], 'url': v['url'],
            'catalog': v['catalog']} for k,v in tables.items()},
        no_matched_hi_seed_established=True,
        remaining_gates=['PSF and validated pixel noise/mask likelihood; lens/source separation.',
            'Independent stellar M/L, gas/dust/missing baryons, 3D deprojection and line-of-sight environment.',
            'Resolved kinematics or a justified aperture dynamical model and orbital/seeing uncertainties.',
            'Explicit relativistic light-propagation closure relating matter dynamics to both metric potentials.',
            'Independent benchmarks and convergence tests for future source and field operators; prospective held-out evaluation.'])


AUDIT_PATHS = [
    'AGENTS.md', 'docs/MOND_OBSERVATION_ATLAS_GOAL.md',
    'docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md',
    'work/gravity-first-principles/mond-atlas-execution-011/README.md',
    'work/gravity-first-principles/mond-atlas-execution-011/execution-status.json',
    'work/gravity-first-principles/mond-atlas-execution-011/verification.json',
    'work/gravity-first-principles/mond-atlas-execution-011/publication-manifest.json',
    'work/gravity-first-principles/mond-atlas-ml-readiness-001/readiness.json',
    'docs/GRAVITY_ITEM17_SLACS_RUNNING_STRENGTH_RESULT.md',
    'configs/gravity_item17_slacs_running_strength_v1.json',
    'configs/lensing_direct_observable_evaluator_readiness.json',
    'work/gravity-cluster-audit-2026-09/runE/runE_slacs.py',
]


def run(config_path, output, offline=False):
    import sys
    import numpy as np
    import astropy
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    output = within(output, ROOT / config['output_root'])
    private = within(ROOT / config['private_root'], ROOT / 'work/private/mond-atlas-lensing-pilot-001')
    if output.exists():
        raise FileExistsError('Use a new immutable output directory')
    private_output = private / 'derived' / output.name
    if private_output.exists():
        raise FileExistsError('Use a unique private derived run name')
    output.mkdir(parents=True)
    private_output.mkdir(parents=True)
    bindings = {p: sha256(ROOT / p) for p in AUDIT_PATHS if (ROOT / p).is_file()}
    bindings.update({str(config_path.relative_to(ROOT)): sha256(config_path),
                     'scripts/mond_atlas_lensing_pilot.py': sha256(Path(__file__)),
                     'scripts/run_mond_atlas_lensing_pilot.py': sha256(ROOT/'scripts/run_mond_atlas_lensing_pilot.py'),
                     'tests/test_mond_atlas_lensing_pilot.py': sha256(ROOT/'tests/test_mond_atlas_lensing_pilot.py')})
    write_json(output/'input-bindings.json', dict(bindings=bindings, created_utc=utc_now(),
        statement='This is a post-discovery ingest binding, not prospective scientific preregistration. No scoring is authorized.'))
    write_json(output/'configuration-snapshot.json', config)
    cache = DownloadCache(private, config['max_download_bytes'], offline=offline)
    sources = []
    all_tables = {}
    for target in config['targets']:
        tables = {}
        for key, spec in config['table_sources'].items():
            url = catalog_url(spec, target['name'])
            name = target['name']+'-'+key+spec.get('cache_tag', '')+'.tsv'
            path, receipt = cache.fetch(name, url, max_bytes=1_000_000)
            parsed = parse_catalog(path, spec, target['name'])
            parsed.update(source_sha256=receipt['sha256'], url=url, catalog=spec['catalog'])
            tables[key] = parsed
            sources.append(dict(receipt, path=str(path.relative_to(ROOT)),
                                release=spec['release'], paper=spec['paper']))
        all_tables[target['name']] = tables
    archive, receipt = cache.fetch('driz_20050714.tar.gz', config['image_archive']['url'])
    sources.append(dict(receipt, path=str(archive.relative_to(ROOT)), **{
        k: config['image_archive'][k] for k in ['release','paper']}))
    inventory, paths = extract_selected(archive,
        [t['image_member_prefix'] for t in config['targets']], private)
    legacy = [audit_legacy(name, path) for name,path in paths]
    natives = {}
    for spec in config['native_images']:
        path, receipt = cache.fetch(spec['filename'], spec['url'], max_bytes=180_000_000)
        target = next(t for t in config['targets'] if t['name'] == spec['name'])
        natives[spec['name']] = audit_native(path, target, private_output)
        sources.append(dict(receipt, path=str(path.relative_to(ROOT)), release=spec['release']))
    for spec in config['instrument_documentation']:
        path, receipt = cache.fetch(spec['key'], spec['url'], max_bytes=3_000_000)
        sources.append(dict(receipt, path=str(path.relative_to(ROOT)), role='primary_instrument_documentation'))
    # Discovery copies are reused without changing them. On a fresh cache,
    # download the same primary documents and enforce their pinned hashes.
    for spec in config['primary_documents']:
        path = private / spec['discovery_path']
        if path.exists():
            receipt = json.loads(path.with_name(path.name+'.access.json').read_text())
        else:
            path, receipt = cache.fetch(spec['key'], spec['url'], max_bytes=15_000_000,
                                       expected_sha256=spec['sha256'])
        if sha256(path) != spec['sha256'] or receipt['sha256'] != spec['sha256']:
            raise ValueError('Primary documentation hash mismatch')
        sources.append(dict(receipt, path=str(path.relative_to(ROOT)), role='primary_paper_or_schema'))
    records = []
    for target in config['targets']:
        records.append(normalized_record(target, all_tables[target['name']], natives[target['name']],
            [x for x in legacy if x['archive_member'].startswith(target['image_member_prefix']+'/')]))
    write_json(output/'systems.json', records)
    write_json(output/'selected-source-tables.json', all_tables)
    write_json(output/'source-manifest.json', sources)
    write_json(output/'legacy-archive-inventory.json', dict(
        archive_sha256=sha256(archive), total_members=len(inventory),
        selected_members=[x['archive_member'] for x in legacy],
        other_members='Not parsed as FITS, visualized, or scored. Their compressed bytes are retained in the original public archive.'))
    write_json(output/'exposure-and-local-audit.json', dict(
        prior_work=bindings, exposure_disclosure=config['exposure_disclosure'],
        old_atlas='Catalog-004 had 137 assets and no lensing source records; separate historical SLACS experiments exist.',
        historical_comparisons='Item 17 and runE use SIE Einstein radii and simplified models; not direct pixel likelihoods or baryonic mass truth.',
        no_current_hi_match='These early-type SLACS lenses are not automatically matched to the twelve nearby HI seeds; no coordinate crossmatch to those seeds is performed.',
        source_failures=[
            'Web fetch of CDS ReadMe failed while direct urllib downloads succeeded.',
            'A&A publisher PDF HTTP 403; Grillo primary preprint acquired from arXiv.',
            'Auger physical table3 is not a VizieR query table; preserved error response, then used documented lenses join from zero-row schema query.',
            'Legacy cutout header lacks calibrated units and complete WCS; native MAST FLT acquired for the same three systems.']))
    summary = dict(status='ACTUAL_SOURCE_INGEST_COMPLETE_OBSERVATIONAL_SCORING_BLOCKED',
        source_admission_disposition='SOURCE_BLOCKED', systems=len(records),
        exact_selected_catalog_rows=len(records)*len(config['table_sources']),
        native_hst_exposures=len(natives), native_target_rois=len(natives),
        legacy_fits_files=len(legacy), primary_source_assets=len(sources),
        source_manifest_bytes=sum(x['bytes'] for x in sources),
        cumulative_download_body_bytes_conservative=cache.used,
        maximum_download_body_bytes=config['max_download_bytes'],
        measured_motion_responses=len(records), gravity_fits=0, lensing_scores=0,
        admitted_joint_likelihoods=0, gpu_used=False, offline=offline,
        runtime=dict(executable=sys.executable, python=platform.python_version(),
                     numpy=np.__version__, astropy=astropy.__version__),
        verification=dict(exact_target_queries=True, forbidden_mass_columns_absent=True,
            source_hashes_reverified=True, fits_structure_verified=True,
            native_identity_and_unit_checks_passed=True,
            native_roi_roundtrip_exact=True,
            maximum_wcs_roundtrip_error_pixel=max(n['wcs_roundtrip_max_error_pixel'] for n in natives.values()),
            legacy_calibration_admitted=False, physical_solver_benchmarks_executed=False,
            summary_is_not_a_gravity_result=True))
    write_json(output/'summary.json', summary)
    lines = ['name,z_lens,z_source,sigma_km_s,sigma_rms_km_s,stellar_MChaBC_Msun,stellar_error_plus_Msun,stellar_error_minus_Msun']
    for record in records:
        m=record['measurements']; s=record['independent_population_constraints']['estimates']['MChaBC']
        lines.append(','.join(str(v) for v in [record['name'],m['z_lens']['value'],m['z_source']['value'],
            m['stellar_velocity_dispersion']['value'],m['stellar_velocity_dispersion']['rms_error'],s['value'],s['error_plus'],s['error_minus']]))
    with (output/'measurements.csv').open('x',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    write_json(output/'output-manifest.json', {
        p.name: {'sha256':sha256(p),'bytes':p.stat().st_size}
        for p in sorted(output.iterdir()) if p.is_file()})
    return summary
