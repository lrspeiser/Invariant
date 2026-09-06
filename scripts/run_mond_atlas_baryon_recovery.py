"""Execute the source metadata recovery without touching motion responses."""
import argparse
from mond_atlas_baryon_recovery import acquire, load_config, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['acquire', 'validate'])
    parser.add_argument('--output', help='New immutable directory inside the assigned output root')
    args = parser.parse_args()
    cfg = load_config()
    if args.command == 'acquire':
        acquire(cfg)
    else:
        result = validate(cfg, args.output)
        if not result['source_metadata_checks_passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
