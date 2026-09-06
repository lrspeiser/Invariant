import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_mond_atlas_stellar_transfer_checked import validate_output_locations


class TransferOutputGuardTests(unittest.TestCase):
    def test_prior_samples_cannot_be_replaced_with_a_new_report_name(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);private=root/'work/private/run';private.mkdir(parents=True)
            packet=private/'G-samples.npz';packet.write_bytes(b'prior immutable bytes')
            config=dict(private_directory='work/private/run',objects=['G'])
            with self.assertRaises(FileExistsError):validate_output_locations(config,root/'work/gravity-first-principles/new',root)
            self.assertEqual(packet.read_bytes(),b'prior immutable bytes')

    def test_fresh_output_and_escape_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=dict(private_directory='work/private/new',objects=['G'])
            validate_output_locations(config,root/'work/gravity-first-principles/new',root)
            config['private_directory']='../outside'
            with self.assertRaises(ValueError):validate_output_locations(config,root/'work/gravity-first-principles/new',root)


if __name__=='__main__':unittest.main()
