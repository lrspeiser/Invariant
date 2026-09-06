"""Target-free ingest failure cases and read-only integration checks on the pilot."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import mond_atlas_lensing_pilot as pilot


class IngestUnitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            dir=ROOT/'work/private/mond-atlas-lensing-pilot-001')
        self.base = Path(self.temp.name)
        self.spec = {'identity':'Name','catalog':'test/catalog',
                     'columns':['Name','sigma','e_sigma']}

    def tearDown(self):
        self.temp.cleanup()

    def table(self, rows='J0001+0001\t200\t12\n', units='\tkm/s\tkm/s'):
        p=self.base/'table.tsv'
        p.write_text('# VizieR\nName\tsigma\te_sigma\n'+units+
                     '\n----------\t---\t---\n'+rows)
        return p

    def test_units_identity_and_null_preserved(self):
        result=pilot.parse_catalog(self.table(),self.spec,'J0001+0001')
        self.assertEqual(result['units']['sigma'],'km/s')
        self.assertEqual(result['row']['sigma'],'200')
        self.assertIsNone(pilot.number('---'))
        with self.assertRaises(ValueError): pilot.number('NaN')

    def test_extra_response_rows_are_rejected(self):
        p=self.table('J0001+0001\t200\t12\nJ0002+0002\t300\t15\n')
        with self.assertRaises(ValueError): pilot.parse_catalog(p,self.spec,'J0001+0001')

    def test_duplicate_rows_are_rejected(self):
        p=self.table('J0001+0001\t200\t12\n'*2)
        with self.assertRaises(ValueError): pilot.parse_catalog(p,self.spec,'J0001+0001')

    def test_wrong_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            pilot.parse_catalog(self.table(),self.spec,'J0002+0002')

    def test_extra_mass_column_is_rejected(self):
        p=self.table(); p.write_text(p.read_text().replace('e_sigma','Mtotlen'))
        with self.assertRaises(ValueError): pilot.parse_catalog(p,self.spec,'J0001+0001')

    def test_service_error_is_not_an_empty_measurement(self):
        p=self.base/'error.tsv'; p.write_text('#INFO\tError=No such table\n')
        with self.assertRaises(ValueError): pilot.parse_catalog(p,self.spec,'J0001+0001')

    def test_plus_sign_and_exact_match_are_url_encoded(self):
        u=pilot.catalog_url(self.spec,'J0001+0001')
        self.assertIn('Name=%3DJ0001%2B0001',u)
        self.assertNotIn('-out=all',u)

    def test_receipts_are_immutable(self):
        p=self.base/'receipt.json'; pilot.write_json(p,{'one':1})
        with self.assertRaises(FileExistsError): pilot.write_json(p,{'one':2})
        self.assertEqual(json.loads(p.read_text()),{'one':1})

    def test_path_escape_is_rejected(self):
        with self.assertRaises(ValueError): pilot.within(self.base/'../escape',self.base)

    def test_archive_rejects_traversal(self):
        p=self.base/'evil.tar.gz'
        with tarfile.open(p,'w:gz') as archive:
            member=tarfile.TarInfo('wanted/../../escape.fits'); member.size=1
            archive.addfile(member,io.BytesIO(b'x'))
        with self.assertRaises(ValueError): pilot.safe_selected_members(p,['wanted'])

    def test_archive_rejects_symbolic_link(self):
        p=self.base/'link.tar.gz'
        with tarfile.open(p,'w:gz') as archive:
            member=tarfile.TarInfo('wanted/image.fits'); member.type=tarfile.SYMTYPE
            member.linkname='elsewhere'; archive.addfile(member)
        with self.assertRaises(ValueError): pilot.safe_selected_members(p,['wanted'])

    def test_archive_selection_is_exact_directory_component(self):
        p=self.base/'good.tar.gz'
        with tarfile.open(p,'w:gz') as archive:
            for name in ['wanted/image.fits','unwanted/image.fits']:
                member=tarfile.TarInfo(name); member.size=1; archive.addfile(member,io.BytesIO(b'x'))
        _,selected=pilot.safe_selected_members(p,['wanted'])
        self.assertEqual(selected,['wanted/image.fits'])

    def test_cache_tampering_is_detected_before_network(self):
        cache=pilot.DownloadCache(self.base)
        p=cache.raw/'source'; p.write_bytes(b'original')
        pilot.write_json(p.with_name('source.access.json'),dict(url='https://example.com',
            status='OK',bytes=8,sha256=hashlib.sha256(b'original').hexdigest()))
        p.write_bytes(b'tampered')
        with patch('urllib.request.urlopen',side_effect=AssertionError('Network forbidden')):
            with self.assertRaises(ValueError): cache.fetch('source','https://example.com')

    def test_offline_missing_source_fails_without_network(self):
        cache=pilot.DownloadCache(self.base,offline=True)
        with patch('urllib.request.urlopen',side_effect=AssertionError('Network forbidden')):
            with self.assertRaises(FileNotFoundError): cache.fetch('missing','https://example.com')

    def test_cap_counts_orphan_partial_files(self):
        (self.base/'raw').mkdir(); (self.base/'raw/orphan').write_bytes(b'x'*11)
        with self.assertRaises(ValueError): pilot.DownloadCache(self.base,limit=10)

    def test_content_length_over_cap_rejected_without_body_read(self):
        class Response:
            url='https://example.com'; status=200; headers={'Content-Length':'11'}
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,*args): raise AssertionError('Body must not be read')
        cache=pilot.DownloadCache(self.base,limit=10)
        with patch('urllib.request.urlopen',return_value=Response()):
            with self.assertRaises(ValueError): cache.fetch('test','https://example.com')
        self.assertEqual(cache.used,0)

    def test_configuration_rejects_reserved_and_mass_columns(self):
        c=json.loads((ROOT/'configs/mond_atlas_lensing_pilot_v1.json').read_text())
        pilot.validate_config(c)
        c['table_sources']['bolton_observed']['columns'].append('bSIE')
        with self.assertRaises(ValueError): pilot.validate_config(c)
        c=json.loads((ROOT/'configs/mond_atlas_lensing_pilot_v1.json').read_text())
        c['targets'][0].update(name='J0037-0942',sdss='003753.21-094220.1')
        with self.assertRaises(ValueError): pilot.validate_config(c)


class ActualSourceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=ROOT/'work/gravity-first-principles/mond-atlas-lensing-pilot-001/ingest-001'
        if not cls.folder.exists():
            raise unittest.SkipTest('Run actual ingest before read-only source integration checks')
        cls.records=json.loads((cls.folder/'systems.json').read_text())

    def test_published_measurements_agree_with_prior_exploration_source(self):
        """Independent older source serialization catches ingestion/join regressions."""
        old=ROOT/'runs/gravity/roadmap/item-17-slacs-running-strength-v1-source'
        manifest=json.loads((old/'sample-manifest.json').read_text())
        refs={o['name']:o for o in manifest['objects']}
        with (old/'exploration-responses.tsv').open() as f:
            responses={r['Name']:r for r in csv.DictReader(f,delimiter='\t')}
        for r in self.records:
            ref=refs[r['name']]
            self.assertEqual(ref['role'],'exploration')
            self.assertEqual(r['measurements']['z_lens']['value'],float(ref['z_lens']))
            self.assertEqual(r['measurements']['z_source']['value'],float(ref['z_source']))
            sigma=r['measurements']['stellar_velocity_dispersion']
            self.assertEqual(sigma['value'],float(responses[r['name']]['sigma']))
            self.assertEqual(sigma['rms_error'],float(responses[r['name']]['e_sigma']))
            for model,estimate in r['independent_population_constraints']['estimates'].items():
                self.assertEqual(estimate['value'],float(ref['stellar_masses_msun'][model]))

    def test_roi_direct_native_equality_and_units(self):
        import numpy as np
        from astropy.io import fits
        for record in self.records:
            n=record['direct_image_constraints']['native']; b=n['roi_bounds_zero_based_half_open']
            with fits.open(ROOT/n['path'],memmap=False) as hdus, np.load(ROOT/n['roi_path']) as roi:
                for key in ['SCI','ERR','DQ']:
                    data=hdus[key,n['sci_extver']].data[b['y_start']:b['y_stop'],b['x_start']:b['x_stop']]
                    self.assertTrue(np.array_equal(roi[key],data))
                self.assertEqual(hdus['SCI',n['sci_extver']].header['BUNIT'],'ELECTRONS')
            self.assertLess(n['wcs_roundtrip_max_error_pixel'],1e-4)

    def test_source_manifest_and_outputs_rehash(self):
        for item in json.loads((self.folder/'source-manifest.json').read_text()):
            self.assertEqual(pilot.sha256(ROOT/item['path']),item['sha256'])
        for name,expected in json.loads((self.folder/'output-manifest.json').read_text()).items():
            self.assertEqual(pilot.sha256(self.folder/name),expected['sha256'])

    def test_no_missing_psf_or_independence_is_silently_admitted(self):
        for r in self.records:
            self.assertEqual(r['disposition'],'SOURCE_BLOCKED')
            self.assertIsNone(r['direct_image_constraints']['native']['psf'])
            self.assertIn('velocity dispersion',r['ancillary_nonindependent_population_constraints']['warning'])
            self.assertTrue(r['no_matched_hi_seed_established'])
            for product in r['direct_image_constraints']['legacy']:
                self.assertFalse(product['calibration_admitted'])


if __name__ == '__main__':
    unittest.main()
