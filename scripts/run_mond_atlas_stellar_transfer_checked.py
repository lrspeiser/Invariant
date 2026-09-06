"""Guard immutable private samples before invoking the frozen transfer implementation.

Use a copied configuration with a new private_directory for each new run.
The original implementation is preserved because two executed runs bind its hash.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from mond_atlas_common import ROOT,read_json
from run_mond_atlas_stellar_transfer import main


def validate_output_locations(config,output,root=ROOT):
    output=Path(output).resolve();root=Path(root).resolve()
    private=(root/config['private_directory']).resolve()
    if not private.is_relative_to(root/'work/private'):
        raise ValueError('private samples must remain below the repository work/private directory')
    if output.exists():raise FileExistsError('immutable public output already exists')
    if not output.is_relative_to(root/'work/gravity-first-principles'):
        raise ValueError('public output must remain below the research report directory')
    if any((private/(name+'-samples.npz')).exists() for name in config['objects']):
        raise FileExistsError('private samples already exist; use a copied configuration with a new private_directory')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.config=args.config.resolve()
    validate_output_locations(read_json(args.config),args.output)
    main(args)
