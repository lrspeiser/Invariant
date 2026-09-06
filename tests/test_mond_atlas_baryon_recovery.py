"""Source conversion falsifiers, including historical applicability mistakes."""
import gzip
import hashlib
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mond_atlas_baryon_recovery as recovery


class BaryonRecoveryTests(unittest.TestCase):
    def test_cleaned_and_uncleaned_color_relations_are_not_interchangeable(self):
        with self.assertRaises(ValueError):
            recovery.stellar_ml('querejeta_global_color','cleaned_stellar_flux',-.06)
        with self.assertRaises(ValueError):
            recovery.stellar_ml('meidt_cleaned_color','integrated_uncleaned_flux',-.06)
        self.assertAlmostEqual(recovery.stellar_ml('meidt_cleaned_color','cleaned_stellar_flux',-.1),10**(-.268))

    def test_global_calibration_range_is_not_extended(self):
        for c in [-.15,-.1,.15,float('nan')]:
            with self.assertRaises(ValueError):
                recovery.stellar_ml('querejeta_global_color','integrated_uncleaned_flux',c)
        self.assertAlmostEqual(recovery.stellar_ml('querejeta_global_color','integrated_uncleaned_flux',0),10**(-.336))

    def test_fixed_old_population_requires_cleaned_context(self):
        self.assertEqual(recovery.stellar_ml('fixed_old_population','cleaned_stellar_flux'),.6)
        with self.assertRaises(ValueError):
            recovery.stellar_ml('fixed_old_population','integrated_uncleaned_flux')

    def test_r21_must_divide_and_helium_must_not_be_repeated(self):
        self.assertAlmostEqual(recovery.molecular_coefficient(4.4,.8),5.5)
        self.assertAlmostEqual(recovery.molecular_coefficient(4.35,.65),6.692307692307692)
        self.assertEqual(recovery.molecular_coefficient(4.35,.5),2*recovery.molecular_coefficient(4.35,1))
        self.assertNotAlmostEqual(recovery.molecular_coefficient(4.35,.65),1.36*4.35/.65)

    def test_invalid_conversion_parameters_fail(self):
        for value in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                recovery.molecular_coefficient(r21=value)

    def test_brightness_conversion_has_no_distance_factor(self):
        tolerance=recovery.load_config()['gates']['stellar_coefficient_relative_tolerance']
        self.assertLess(abs(recovery.stellar_luminosity_coefficient()/704.04-1),tolerance)

    def test_missing_numeric_metadata_is_none(self):
        r=recovery.parse_row('FAKE'.ljust(156),recovery.CATALOG_COLUMNS)
        self.assertIsNone(r['distance_mpc'])
        self.assertIsNone(r['catalog_ellipticity'])

    def test_truncation_duplicate_and_alias_collision_fail(self):
        with self.assertRaises(ValueError):
            recovery.parse_row('NGC2903',recovery.CATALOG_COLUMNS)
        with self.assertRaises(ValueError):
            recovery.unique_rows([{'object_id':'A'},{'object_id':'A'}])
        with self.assertRaises(ValueError):
            recovery.resolve_name('A',{'A':{},'B':{}},{'A':['A','B']})

    def test_absence_not_fabricated_match(self):
        self.assertIsNone(recovery.resolve_name('DDO154',{}, {'DDO154':['DDO154','UGC08024']}))

    def test_p5_exclusion_does_not_mean_zero_stellar_mass(self):
        r=recovery.parse_p5('IC2574 2\nNGC2841 1\n')
        self.assertEqual(r['IC2574']['excluded'],2)
        self.assertIsNone(r['IC2574']['quality_flag'])
        self.assertNotIn('stellar_mass',r['IC2574'])

    def test_output_cannot_escape_owned_tree(self):
        with self.assertRaises(ValueError):
            recovery.owned_path('docs/unsafe.json','work/gravity-first-principles/mond-atlas-baryon-recovery-001')

    def test_real_sources_match_frozen_hashes(self):
        cfg=recovery.load_config()
        for s in cfg['sources'][:2]:
            data,e=recovery.verify_recovered_source(s,recovery.ROOT/cfg['private_directory']/s['filename'])
            self.assertEqual(len(data.decode('ascii').splitlines()),2352)
            self.assertTrue(e['raw_matches_prior'])

    def test_real_geometry_and_independent_parser(self):
        cfg=recovery.load_config()
        r=recovery.geometry_audit(cfg,recovery.ROOT/cfg['private_directory'])
        self.assertEqual(r['missing_source_rows'],['DDO154','NGC6946','NGC7331'])
        self.assertTrue(all(x['exact_match'] for x in r['prior_reconciliation']))
        self.assertEqual(sum(x['compared_fields'] for x in r['prior_reconciliation']),65)

    def test_target_free_dimensional_controls(self):
        self.assertTrue(recovery.conversion_benchmarks(recovery.load_config())['all_passed'])


if __name__=='__main__':
    unittest.main()
