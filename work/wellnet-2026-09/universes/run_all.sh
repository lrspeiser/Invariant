#!/bin/sh
# The full Stage 5 chain, in order.  Each stage reads the previous stage's JSON.
set -e
cd "$(dirname "$0")/.."
python -m universes.run_stage5
python -m universes.run_finescan
python -m universes.run_equiv_amplitude
python -m universes.fingerprint
python -m universes.render_report
