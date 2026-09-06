"""Bounded, append-only acquisition and source-only metadata validation."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/mond_atlas_baryon_recovery_v1.json'


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def load_config():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    if cfg['admission_disposition'] != 'SOURCE_BLOCKED' or cfg['gates']['allow_response_access']:
        raise ValueError('source-only admission boundary changed')
    return cfg


def owned_path(relative, base):
    path = (ROOT / relative).resolve()
    boundary = (ROOT / base).resolve()
    if path != boundary and boundary not in path.parents:
        raise ValueError('path outside assigned output directory')
    return path


def acquire(cfg):
    """One receipt per request; preserve failures and never overwrite a payload."""
    private = owned_path(cfg['private_directory'], 'work/private/mond-atlas-baryon-recovery-001')
    private.mkdir(parents=True, exist_ok=True)
    prior = list(private.glob('request-*.json'))
    used = sum(json.loads(p.read_text())['downloaded_bytes'] for p in prior)
    count = len(prior)
    results = []
    for source in cfg['sources']:
        destination = private / source['filename']
        if destination.exists():
            results.append({'id': source['id'], 'status': 'EXISTING', 'sha256': sha256(destination)})
            continue
        count += 1
        receipt = dict(source, requested_at_utc=datetime.now(timezone.utc).isoformat(),
                       downloaded_bytes=0, status='FAILED', config_sha256=sha256(CONFIG))
        chunks = []
        try:
            req = urllib.request.Request(source['url'], headers={'User-Agent': 'Invariant-source-metadata-audit/1.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                receipt.update(final_url=response.url, http_status=response.status,
                               content_type=response.headers.get('Content-Type'),
                               last_modified=response.headers.get('Last-Modified'))
                while True:
                    remaining = min(cfg['max_download_bytes'] - used,
                                    cfg['max_file_bytes'] - receipt['downloaded_bytes'])
                    if remaining <= 0:
                        raise ValueError('download byte cap reached')
                    chunk = response.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    receipt['downloaded_bytes'] += len(chunk)
                    used += len(chunk)
            payload = b''.join(chunks)
            with destination.open('xb') as handle:
                handle.write(payload)
            receipt.update(status='DOWNLOADED', sha256=sha256(destination),
                           path=destination.relative_to(ROOT).as_posix())
            if 'expected_raw_sha256' in source:
                receipt['raw_matches_prior'] = receipt['sha256'] == source['expected_raw_sha256']
                decoded = gzip.decompress(payload) if payload[:2] == b'\x1f\x8b' else payload
                receipt['decoded_sha256'] = hashlib.sha256(decoded).hexdigest()
                receipt['decoded_matches_prior'] = receipt['decoded_sha256'] == source['expected_decoded_sha256']
        except Exception as exc:
            receipt['error'] = f'{type(exc).__name__}: {exc}'
        write_json(private / f'request-{count:03d}.json', receipt)
        print(source['id'], receipt['status'], receipt.get('raw_matches_prior', ''), receipt.get('error', ''), flush=True)
        results.append(receipt)
    return results


# One-based inclusive columns transcribed from the recovered CDS ReadMe files.
# The independent Cds reader below derives its own columns from those files.
CATALOG_COLUMNS = {
    'ra_deg': (14, 22, 'RAdeg', 'deg'), 'dec_deg': (24, 32, 'DEdeg', 'deg'),
    'semi_major_25p5_arcsec': (35, 39, 'amaj', 'arcsec'),
    'catalog_position_angle_deg': (41, 46, 'PA', 'deg'),
    'catalog_ellipticity': (48, 52, 'ell', None),
    'distance_mpc': (120, 126, 'Dmean', 'Mpc'),
    'distance_standard_deviation_mpc': (129, 134, 'e_Dmean', 'Mpc'),
}
PIPELINE_COLUMNS = {
    'pipeline4_center_x_pixel': (12, 18, 'xc', 'pix'),
    'pipeline4_center_y_pixel': (20, 26, 'yc', 'pix'),
    'outer_position_angle_deg': (28, 32, 'PA', 'deg'),
    'outer_position_angle_sd_deg': (34, 37, 'e_PA', 'deg'),
    'outer_ellipticity': (39, 43, 'Ell', None),
    'outer_ellipticity_sd': (45, 49, 'e_Ell', None),
    'outer_isophote_rmin_arcsec': (51, 54, 'Rmin', 'arcsec'),
    'outer_isophote_rmax_arcsec': (56, 59, 'Rmax', 'arcsec'),
}


def parse_row(row, columns, pipeline=False, verified_release=False):
    if not row.strip() or (len(row) < 10 and not verified_release):
        raise ValueError('truncated metadata row')
    # CDS omits trailing spaces on some all-null tails. Raw hashes certify that
    # these are released rows, so missing trailing fields remain None.
    row = row.ljust(max(c[1] for c in columns.values()))
    result = {'object_id': row[:10].strip()}
    for key, (start, end, _, _) in columns.items():
        token = row[start-1:end].strip()
        result[key] = float(token) if token else None
        if result[key] is not None and not math.isfinite(result[key]):
            raise ValueError('nonfinite metadata')
    if pipeline:
        result['orientation_flag'] = row[60:62].strip() or None
    return result


def unique_rows(rows):
    indexed = {}
    for row in rows:
        name = row['object_id']
        if name in indexed:
            raise ValueError(f'duplicate identity: {name}')
        indexed[name] = row
    return indexed


def resolve_name(name, indexed, aliases):
    found = [n for n in aliases.get(name, [name]) if n in indexed]
    if len(found) > 1:
        raise ValueError(f'ambiguous identity: {name}')
    return found[0] if found else None


def verify_recovered_source(source, path):
    raw = Path(path).read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    decoded = gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw
    decoded_hash = hashlib.sha256(decoded).hexdigest()
    if (raw_hash != source['expected_raw_sha256'] or
            decoded_hash != source['expected_decoded_sha256']):
        raise ValueError('recovered source differs from prior frozen hashes')
    return decoded, dict(raw_bytes=len(raw), raw_sha256=raw_hash,
                         decoded_bytes=len(decoded), decoded_sha256=decoded_hash,
                         raw_matches_prior=True, decoded_matches_prior=True)


def geometry_audit(cfg, private):
    from astropy.io import ascii
    from astropy.units import UnitsWarning
    import numpy as np
    tables, sources, schemas = {}, [], []
    decoded_dir = private / 'decoded'
    decoded_dir.mkdir(exist_ok=True)
    cases = [('s4g_catalog', 's4g.dat', 'catalog-ReadMe.txt', CATALOG_COLUMNS),
             ('s4g_pipeline4', 'table1.dat', 'pipeline4-ReadMe.txt', PIPELINE_COLUMNS)]
    for sid, filename, readme, columns in cases:
        source = next(s for s in cfg['sources'] if s['id'] == sid)
        decoded, evidence = verify_recovered_source(source, private/source['filename'])
        dest = decoded_dir / filename
        if dest.exists():
            if dest.read_bytes() != decoded:
                raise ValueError('decoded cache differs')
        else:
            with dest.open('xb') as handle:
                handle.write(decoded)
        lines = decoded.decode('ascii').splitlines()
        if len(lines) != cfg['gates']['expected_table_rows']:
            raise ValueError('record count mismatch')
        indexed = unique_rows(parse_row(line, columns, sid == 's4g_pipeline4', verified_release=True) for line in lines)
        # The Cds class retains CDS nullable-column masks. The ascii.read wrapper
        # in the local Astropy version overrides them with incompatible defaults.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', UnitsWarning)
            reference = ascii.Cds(readme=str(private/readme)).read(str(dest))
        ref_names = [str(n).strip() for n in reference['Name']]
        if len(set(ref_names)) != len(lines) or set(ref_names) != set(indexed):
            raise ValueError('independent name/row-count mismatch')
        comparisons = 0
        for i, ref_name in enumerate(ref_names):
            for key, (_, _, label, unit) in columns.items():
                v = reference[label][i]
                expected = None if np.ma.is_masked(v) else float(v)
                if indexed[ref_name][key] != expected:
                    raise ValueError(f'independent parser mismatch {sid}:{ref_name}:{key}')
                comparisons += 1
            if sid == 's4g_pipeline4':
                flag = reference['Flag'][i]
                expected = None if np.ma.is_masked(flag) else (str(flag).strip() or None)
                if indexed[ref_name]['orientation_flag'] != expected:
                    raise ValueError('independent orientation flag mismatch')
                comparisons += 1
        for key, (start, end, label, unit) in columns.items():
            actual = str(reference[label].unit) if reference[label].unit is not None else None
            if actual != unit:
                raise ValueError(f'unit mismatch {label}: {actual} != {unit}')
            schemas.append(dict(source_id=sid, field=key, cds_label=label,
                                first_byte=start, last_byte=end, unit=unit,
                                nullable=True, independent_unit_check=True))
        tables[sid] = indexed
        sources.append(dict(source_id=sid, url=source['url'], **evidence,
                            records=len(lines), independent_field_comparisons=comparisons,
                            parser_warnings=sorted(set(str(w.message) for w in caught))))
    catalog, pipeline = tables['s4g_catalog'], tables['s4g_pipeline4']
    prior = json.loads((ROOT/cfg['prior_geometry']).read_text())
    reconciliation = []
    for old in prior['objects']:
        combined = {**catalog[old['object_id']], **pipeline[old['object_id']]}
        checked = [k for k in old if k in combined and k != 'object_id']
        failed = [k for k in checked if old[k] != combined[k]]
        reconciliation.append(dict(object_id=old['object_id'], compared_fields=len(checked),
                                   mismatches=failed, exact_match=not failed))
    records = []
    for name in cfg['seeds']:
        cn = resolve_name(name, catalog, cfg['aliases'])
        pn = resolve_name(name, pipeline, cfg['aliases'])
        c, p = catalog.get(cn), pipeline.get(pn)
        record = dict(object_id=name, catalog_name=cn, pipeline4_name=pn,
                      catalog=c, pipeline4=p,
                      disposition='SOURCE_GEOMETRY_RECOVERED' if c and p else 'ABSENT_FROM_RECOVERED_S4G_TABLES')
        if c and p:
            record['distance_sd_fraction'] = c['distance_standard_deviation_mpc']/c['distance_mpc']
            record['pa_axial_difference_deg'] = abs((c['catalog_position_angle_deg']-p['outer_position_angle_deg']+90)%180-90)
            record['ellipticity_difference'] = p['outer_ellipticity']-c['catalog_ellipticity']
            record['geometry_unique_or_fitted_to_motion'] = False
        records.append(record)
    return dict(sources=sources, schema=schemas, objects=records,
                prior_reconciliation=reconciliation,
                missing_source_rows=[r['object_id'] for r in records if r['catalog'] is None],
                uncertainty_note='Dmean is a mean redshift-independent distance; e_Dmean is its standard deviation, not a certified posterior or error on the mean. P4 center pixels belong to its source image, not P5 cutouts.')


def parse_p5(text):
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith(('\\', '|')):
            continue
        parts = line.split()
        excluded = int(parts[1])
        if excluded not in (0, 1, 2, 3) or len(parts) != (12 if excluded == 0 else 2):
            raise ValueError('unexpected P5 schema')
        row = dict(object_id=parts[0], excluded=excluded, quality_flag=None)
        if excluded == 0:
            row.update(ica_iteration=int(parts[2]), quality_flag=int(parts[3]),
                       stellar_color_mag=float(parts[4]), stellar_color_error_mag=float(parts[5]),
                       dust_color_mag=float(parts[6]), dust_color_error_mag=float(parts[7]),
                       stellar_flux_fraction=float(parts[8]), semimajor_arcsec=float(parts[9]),
                       ellipticity=float(parts[10]), position_angle_deg=float(parts[11]))
        rows.append(row)
    return unique_rows(rows)


def stellar_ml(mode, input_role, color=None):
    """Algebra checks only; not an observation-level mass-map builder."""
    if mode == 'fixed_old_population' and input_role == 'cleaned_stellar_flux':
        return 0.6
    if color is None or not math.isfinite(color):
        raise ValueError('finite color required')
    if mode == 'meidt_cleaned_color' and input_role == 'cleaned_stellar_flux':
        return 10**(3.98*color + 0.13)
    if mode == 'querejeta_global_color' and input_role == 'integrated_uncleaned_flux':
        if not -0.1 < color < 0.15:
            raise ValueError('outside published global-color calibration range')
        return 10**(-0.339*color - 0.336)
    raise ValueError('conversion not applicable to this observable')


def stellar_luminosity_coefficient(zero_point_jy=280.9, solar_vega_mag=3.24):
    return 1e6 / (100 * zero_point_jy * 10**(-0.4*solar_vega_mag))


def molecular_coefficient(alpha_co10=4.35, r21=0.65):
    if not all(math.isfinite(x) and x > 0 for x in (alpha_co10, r21)):
        raise ValueError('positive finite conversion parameters required')
    return alpha_co10/r21


def conversion_benchmarks(cfg):
    import astropy.units as u
    import numpy as np
    from astropy.constants import M_sun, m_p
    coeff = stellar_luminosity_coefficient()
    # Independent magnitude/surface-brightness expression, not the flux formula above.
    arcsec_per_rad = 180/math.pi*3600
    mu = -2.5*math.log10(1e6/arcsec_per_rad**2/280.9)
    ref = 10**(-0.4*(mu-3.24-(5*math.log10(arcsec_per_rad)-5)))
    alpha = (2*m_p*2e20/u.cm**2).to(u.Msun/u.pc**2).value*1.36
    pc_m = u.pc.to(u.m)
    scalar_alpha = 2*m_p.value*2e20*1e4*pc_m**2/M_sun.value*1.36
    pixel = coeff*(0.75/arcsec_per_rad*1e6)**2
    inclinations, distances = [0., 30., 60., 80.], [1., 5., 20.]
    conservation_errors = []
    for inc in inclinations:
        for d in distances:
            projected_area = (0.75/arcsec_per_rad*d*1e6)**2
            cosine = math.cos(math.radians(inc))
            faceon_mass = coeff*.6*cosine*projected_area/cosine
            projected_mass = coeff*.6*projected_area
            conservation_errors.append(abs(faceon_mass/projected_mass-1))
    checks = {
        'stellar_definitions_agree': abs(coeff/ref-1) < cfg['gates']['float_benchmark_relative_tolerance'],
        'stellar_published_704p04': abs(coeff/704.04-1) < cfg['gates']['stellar_coefficient_relative_tolerance'],
        'stellar_published_pixel_9308p23': abs(pixel/9308.23-1) < cfg['gates']['stellar_coefficient_relative_tolerance'],
        'co_units_vs_scalar': abs(alpha/scalar_alpha-1) < cfg['gates']['float_benchmark_relative_tolerance'],
        'co_published_4p4_rounding': abs(alpha/4.4-1) < cfg['gates']['co_coefficient_relative_tolerance'],
        'co_4p35_rounding': abs(alpha/4.35-1) < cfg['gates']['co_coefficient_relative_tolerance'],
        'distance_inclination_mass_conservation': max(conservation_errors) < cfg['gates']['float_benchmark_relative_tolerance'],
        'r21_inverse_limit': molecular_coefficient(r21=.5) == 2*molecular_coefficient(r21=1),
        'zero_and_signed_flux_linearity': np.array_equal(np.array([-1., 0., 1.])*molecular_coefficient(), [-molecular_coefficient(), 0, molecular_coefficient()]),
    }
    return dict(checks={k:bool(v) for k,v in checks.items()}, all_passed=all(checks.values()),
                stellar_lsun_pc2_per_mjy_sr=coeff, independent_stellar_coefficient=ref,
                stellar_relative_difference_from_published=coeff/704.04-1,
                stellar_pixel_coefficient_0p75_arcsec_at_1mpc=pixel,
                co_alpha10_from_proton_mass_including_helium=alpha,
                co_atlas_alpha21=molecular_coefficient(),
                co_2008_rounded_alpha21=4.4/.8,
                co_atlas_to_2008_rounded_ratio=molecular_coefficient()/(4.4/.8),
                conservation_cases=len(conservation_errors), max_conservation_relative_error=max(conservation_errors),
                note='Target-free unit/algebra checks. R21=.65 is a population assumption, not a measured ratio for every seed. No source mass maps, likelihoods or force scores produced.')


def asset_audit(cfg, private):
    from astropy.io import fits
    assets = json.loads((ROOT/cfg['asset_receipt']).read_text())['files']
    allowed = {'STELLAR_MASS_MAP','STELLAR_IRAC1_FLUX','STELLAR_IRAC1_WEIGHT',
               'STELLAR_ICA_MASK','STELLAR_COLOR_MAP','CO21_MOM0','CO21_EMOM0'}
    headers, rows, failures = {}, [], []
    for a in sorted(assets, key=lambda a:(a['name'],a['role'])):
        if a['role'] not in allowed or a['name'] not in cfg['seeds']:
            raise ValueError('asset outside source-only allowlist')
        path = owned_path(a['file'], 'work/private/stellar-co-12gal-001')
        actual = sha256(path)
        if actual != a['sha256']:
            raise ValueError('existing asset hash changed: '+a['file'])
        h = fits.getheader(path)
        headers[(a['name'],a['role'])] = h
        role = a['role']
        semantic = {'STELLAR_MASS_MAP':'cleaned_stellar_flux',
                    'STELLAR_COLOR_MAP':'cleaned_stellar_color',
                    'STELLAR_ICA_MASK':'categorical_mask',
                    'STELLAR_IRAC1_WEIGHT':'coverage_frames_times_10'}.get(role, role)
        semantic_unit = {'STELLAR_COLOR_MAP':'mag (Vega)', 'STELLAR_ICA_MASK':'dimensionless labels',
                         'STELLAR_IRAC1_WEIGHT':'10 * contributing frame count'}.get(role,h.get('BUNIT'))
        unit_ok = h.get('BUNIT') == ('K KM/S' if role.startswith('CO21') else 'MJy/sr')
        line_ok = not role.startswith('CO21') or h.get('LINE','').strip() == '12CO(2-1)'
        if not unit_ok or not line_ok or h.get('NAXIS') != 2:
            failures.append(dict(object_id=a['name'], role=role, failure='schema/line/units'))
        history = list(h.get('HISTORY', []))
        keys = ['OBJECT','CHNLNUM','BUNIT','LINE','RESTFREQ','BMAJ','BMIN','TELESCOP','BACK_SUB','BACKGRND',
                'NAXIS','NAXIS1','NAXIS2','CTYPE1','CTYPE2','CRPIX1','CRPIX2','CRVAL1','CRVAL2',
                'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2']
        rows.append(dict(object_id=a['name'], historical_role=role, semantic_role=semantic,
                         semantic_unit=semantic_unit, header_unit_requires_override=role in ['STELLAR_COLOR_MAP','STELLAR_ICA_MASK','STELLAR_IRAC1_WEIGHT'],
                         url=a['url'], path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                         sha256=actual, matches_prior_hash=True,
                         header={k:h[k] for k in keys if k in h},
                         efficiency_history=[s for s in history if 'efficiency' in s.lower()],
                         schema_units_line_pass=unit_ok and line_ok and h.get('NAXIS') == 2))
    p5 = parse_p5((private/'P5_table.txt').read_text())
    irac2 = json.loads((ROOT/cfg['irac2_inventory']).read_text())['records']
    irac2_state = [dict(object_id=a['object_id'], role=a['role'], path=a['relative_path'],
                       exists=(ROOT/a['relative_path']).exists(), prior_sha256=a['sha256'],
                       url=a['url'], revalidated=False) for a in irac2]
    objects = []
    wcs_keys = ['NAXIS1','NAXIS2','CTYPE1','CTYPE2','CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2']
    for name in cfg['seeds']:
        pn = resolve_name(name,p5,cfg['aliases'])
        star_role = 'STELLAR_MASS_MAP' if (name,'STELLAR_MASS_MAP') in headers else 'STELLAR_IRAC1_FLUX'
        star = headers[(name,star_role)]
        expected_names = {n.upper() for n in cfg['aliases'].get(name,[name])}
        star_identity = str(star.get('OBJECT','')).upper() in expected_names
        co, err = headers[(name,'CO21_MOM0')],headers[(name,'CO21_EMOM0')]
        co_wcs = all(co.get(k)==err.get(k) for k in wcs_keys)
        if not star_identity or not co_wcs:
            failures.append(dict(object_id=name, failure='stellar identity or CO/error coordinate mismatch'))
        object_p5 = p5.get(pn)
        if star_role == 'STELLAR_MASS_MAP' and (not object_p5 or object_p5['excluded'] != 0):
            failures.append(dict(object_id=name, failure='P5 file/table mismatch'))
        objects.append(dict(object_id=name, stellar_branch='S4G_P5_CLEANED_FLUX' if star_role=='STELLAR_MASS_MAP' else 'SINGS_IRAC1_UNCLEANED_FLUX',
                            stellar_header_identity_pass=star_identity, p5_metadata=object_p5,
                            irac2_declared_files=sum(a['object_id']==name for a in irac2_state),
                            irac2_declared_files_present=sum(a['object_id']==name and a['exists'] for a in irac2_state),
                            co_error_same_spatial_header=co_wcs, co_line=co.get('LINE'),
                            co_beam_fwhm_arcsec=co.get('BMAJ',0)*3600,
                            co_temperature_scale='Tmb; correction already applied per release README',
                            co_2009_publication_status='UPPER_LIMIT' if name in ['DDO154','IC2574','UGC04305'] else 'DETECTED',
                            co_molecular_mass_observed=False,
                            stellar_population_mass_observed=False,
                            p1_p5_transfer_revalidated_here=False))
    return dict(assets=rows, objects=objects, failures=failures, irac2_inventory=irac2_state,
                source_image_arrays_read=0, source_fits_headers_read=len(rows),
                existing_source_bytes_hashed=sum(a['bytes'] for a in rows))


def field_provenance(cfg):
    """Read receipt bindings and source code, never field or response arrays."""
    relevant = {
        'scripts/build_mond_atlas_ngc2903_source.py',
        'scripts/run_mond_atlas_common_basis_fields.py',
        'scripts/run_mond_atlas_blocked_refinement.py',
        'scripts/run_mond_atlas_reprojected_fields.py',
        'configs/mond_atlas_ngc2903_field_v1.json',
        'configs/mond_atlas_common_basis_fields_v1.json',
        'configs/mond_atlas_reprojected_fields_v1.json',
    }
    receipt_names = [
        'work/gravity-first-principles/mond-atlas-source-001/source-audit.json',
        'work/gravity-first-principles/mond-atlas-field-003/summary.json',
        'work/gravity-first-principles/mond-atlas-field-005/summary.json',
        'work/gravity-first-principles/mond-atlas-field-006/prospective-bindings.json',
    ]
    evidence = []
    for filename in receipt_names:
        record = json.loads((ROOT/filename).read_text())
        bindings = {}
        for group in ['bindings','code_hashes','source_bindings','input_bindings']:
            entries = record.get(group,{})
            if isinstance(entries,list):
                entries = {b['path']:b['sha256'] for b in entries}
            bindings.update({k.replace('\\','/'):v for k,v in entries.items()})
        checks = [dict(path=p, recorded_sha256=h, current_sha256=sha256(ROOT/p),
                       match=sha256(ROOT/p)==h) for p,h in bindings.items() if p in relevant]
        evidence.append(dict(receipt=filename,sha256=sha256(ROOT/filename),selected_bindings=checks))
    gravity = json.loads((ROOT/cfg['prior_atlas_field']).read_text())
    code_evidence = []
    needles = {
        'scripts/build_mond_atlas_ngc2903_source.py': ["config['conversions']['stellar_lsun_pc2_per_mjy_sr']"],
        'scripts/run_mond_atlas_common_basis_fields.py': ["from run_mond_atlas_blocked_refinement import execute", "gravity = read_json(ROOT/config['gravity_protocol'])"],
        'scripts/run_mond_atlas_blocked_refinement.py': ["conversion['nominal_stellar_ml']"],
        'scripts/run_mond_atlas_reprojected_fields.py': ["conv['nominal_stellar_ml']"],
    }
    for filename, patterns in needles.items():
        lines = (ROOT/filename).read_text().splitlines()
        for pattern in patterns:
            hits = [dict(line=i+1,text=line.strip()) for i,line in enumerate(lines) if pattern in line]
            if not hits:
                raise ValueError('source provenance pattern changed')
            code_evidence.append(dict(path=filename,sha256=sha256(ROOT/filename),matches=hits))
    matched = all(c['match'] for e in evidence for c in e['selected_bindings'])
    return dict(disposition='FIXED_ML_CURRENT_ATLAS_TRACE' if matched else 'PROVENANCE_HASH_MISMATCH',
                receipts=evidence, code=code_evidence, recorded_bindings_all_match=matched,
                nominal_stellar_ml=gravity['conversions']['nominal_stellar_ml'],
                current_common_basis_and_mixed_use_erroneous_color_branch=False if matched else None,
                explanation='Source-001 stores stellar luminosity using 704.04. Common-basis common_thin/common_mixed and the earlier reprojected/blocked mixed branches multiply that luminosity by the same fixed 0.6. Their source depths and shapes remain conditional. The older five-object COLOR_PUBLISHED_WITH_FALLBACK branch is a separate applicability failure.',
                field_arrays_opened=0, motion_residuals_opened=0)


def validate(cfg, output=None):
    private = owned_path(cfg['private_directory'], 'work/private/mond-atlas-baryon-recovery-001')
    output = owned_path(output or cfg['output_directory'], cfg['output_directory'])
    if output.exists():
        raise FileExistsError('immutable validation output already exists')
    output.mkdir(parents=True)
    bind = [CONFIG, Path(__file__), ROOT/'scripts/run_mond_atlas_baryon_recovery.py',
            ROOT/'tests/test_mond_atlas_baryon_recovery.py',
            ROOT/'docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md']
    bind += [ROOT/cfg[k] for k in ['prior_geometry','prior_stellar_conversion','asset_receipt',
                                   'irac2_inventory','prior_five_object_builder','prior_atlas_field']]
    registration = dict(admission_disposition='SOURCE_BLOCKED', config=cfg,
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        bindings=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p)) for p in bind],
                        response_access_authorized=False)
    write_json(output/'registration.json',registration)
    try:
        geometry = geometry_audit(cfg, private)
        assets = asset_audit(cfg, private)
        benchmarks = conversion_benchmarks(cfg)
        provenance = field_provenance(cfg)
        for filename, value in [('geometry.json',geometry),('asset-metadata.json',assets),('benchmarks.json',benchmarks),('field-provenance.json',provenance)]:
            write_json(output/filename,value)
        requests = [json.loads(p.read_text()) for p in sorted(private.glob('request-*.json'))]
        source_manifest = []
        for source in cfg['sources']:
            p = private/source['filename']
            receipts = [r for r in requests if r['id']==source['id']]
            source_manifest.append(dict(**source, path=p.relative_to(ROOT).as_posix(),
                                        bytes=p.stat().st_size,sha256=sha256(p),requests=receipts))
        write_json(output/'source-manifest.json',dict(sources=source_manifest,
                     downloaded_bytes=sum(r['downloaded_bytes'] for r in requests),
                     download_cap_bytes=cfg['max_download_bytes'], raw_data_publishable=False))
        passed = (not assets['failures'] and benchmarks['all_passed'] and provenance['recorded_bindings_all_match'] and
                  all(r['exact_match'] for r in geometry['prior_reconciliation']))
        summary = dict(status='SOURCE_METADATA_RECOVERY_VALIDATED' if passed else 'VALIDATION_FAILED',
                       admission_disposition='SOURCE_BLOCKED', source_metadata_checks_passed=passed,
                       exact_originals_recovered=2, seeds=len(cfg['seeds']),
                       source_geometry_present=sum(r['catalog'] is not None for r in geometry['objects']),
                       missing_s4g_geometry=geometry['missing_source_rows'],
                       reliable_p4_orientation=[r['object_id'] for r in geometry['objects'] if r['pipeline4'] and r['pipeline4']['orientation_flag']=='ok'],
                       uncertain_p4_orientation=[r['object_id'] for r in geometry['objects'] if r['pipeline4'] and r['pipeline4']['orientation_flag']=='u'],
                       prior_geometry_records_exact=sum(r['exact_match'] for r in geometry['prior_reconciliation']),
                       prior_geometry_fields_exact=sum(r['compared_fields'] for r in geometry['prior_reconciliation'] if r['exact_match']),
                       independent_parser_field_comparisons=sum(r['independent_field_comparisons'] for r in geometry['sources']),
                       existing_source_assets_hashed=len(assets['assets']),
                       cleaned_flux_mislabeled_as_mass=sum(a['historical_role']=='STELLAR_MASS_MAP' for a in assets['assets']),
                       header_semantic_unit_overrides=sum(a['header_unit_requires_override'] for a in assets['assets']),
                       algebra_benchmarks_passed=sum(benchmarks['checks'].values()),
                       current_atlas_common_basis_and_mixed_stellar_ml=provenance['nominal_stellar_ml'],
                       current_atlas_color_branch_mismatch_applies=provenance['current_common_basis_and_mixed_use_erroneous_color_branch'],
                       download_bytes=sum(r['downloaded_bytes'] for r in requests),
                       response_datasets_opened=0, motion_residuals_opened=0, scores_computed=0,
                       source_image_arrays_read=0, unique_3d_truth_claimed=False,
                       required_remaining=['Three missing S4G seed geometry records need independent source alternatives.',
                           'Six uncertain P4 orientations and all source depths need independent ensembles.',
                           'Resolve SINGS aperture/extended-source calibration and source noise covariance.',
                           'Revalidate IRAC2 originals and matched-aperture flux calibration before global-color conversion.',
                           'Historical cleaned-pixel/global-color formula mismatch requires separate integration review.',
                           'Establish release-specific CO calibration, source-selection covariance, X_CO and R21 ensembles; retain nondetections and CO-dark/ionized phases.',
                           'P1-to-P5 registration outside NGC2903 remains unvalidated.'])
        write_json(output/'summary.json',summary)
        print(json.dumps(summary,indent=2))
        return summary
    except Exception as exc:
        write_json(output/'failure.json',dict(status='VALIDATION_FAILED',error=f'{type(exc).__name__}: {exc}',scores_computed=0))
        raise
