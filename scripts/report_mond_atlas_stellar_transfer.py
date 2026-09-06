"""Summarize both fixed image-registration partitions, retaining local failures."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest


def report(first,second,output):
    if output.exists():raise FileExistsError('immutable output')
    a=read_json(first/'summary.json');b=read_json(second/'summary.json')
    for folder in [first,second]:
        for path,h in read_json(folder/'prospective-bindings.json')['bindings'].items():
            if digest(ROOT/path)!=h:raise ValueError('changed input '+path)
    other={r['galaxy']:r for r in b['results']};rows=[]
    for r in a['results']:
        reverse=other[r['galaxy']]
        rows.append(dict(galaxy=r['galaxy'],dx_p1_pixel=r['fit']['shift'][0],dy_p1_pixel=r['fit']['shift'][1],
            zero_shift_validation_rms_percent=100*r['validation_before']['relative_rms'],
            corrected_validation_rms_percent=100*r['validation_after']['relative_rms'],
            reversed_validation_rms_percent=100*reverse['validation_after']['relative_rms'],
            split_shift_difference_p1_pixels=float(np.linalg.norm(np.array(r['fit']['shift'])-reverse['fit']['shift'])),
            first_relative_pass=r['relative_transfer_pass'],reverse_relative_pass=reverse['relative_transfer_pass'],
            prior_absolute_footprint_pass=r['prior_absolute_footprint_pass'],
            first_maximum_quadrant_relative_rms_percent=100*max(q['relative_rms'] for q in r['validation_quadrants'] if q['relative_rms'] is not None)))
    output.mkdir(parents=True);write_csv(output/'comparisons.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    x=np.arange(len(rows));names=[r['galaxy'] for r in rows]
    axes[0].bar(x-.18,[r['zero_shift_validation_rms_percent'] for r in rows],.36,label='Nominal image coordinates',color='#a4abb1')
    axes[0].bar(x+.18,[r['corrected_validation_rms_percent'] for r in rows],.36,label='Fitted source-image offset',color='#177a8c')
    axes[0].set_ylabel('Held-block reconstruction RMS (%)');axes[0].legend(fontsize=8)
    axes[0].set_title('Offsets explain much of the image mismatch')
    axes[1].plot(x,[r['corrected_validation_rms_percent'] for r in rows],'o-',label='First calibration blocks',color='#177a8c')
    axes[1].plot(x,[r['reversed_validation_rms_percent'] for r in rows],'s--',label='Reversed calibration blocks',color='#9156a0')
    axes[1].axhline(5,color='#777777',ls=':',label='Declared overall RMS gate')
    axes[1].set_ylabel('Held-block reconstruction RMS (%)');axes[1].set_title('All five pass overall in both partitions');axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x,names,rotation=25);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Stellar source registration, not a gravity measurement',fontsize=14)
    fig.text(.5,-.035,'P5 stars + dust compared with P1 flux; scale/background fit on calibration blocks only.\nNGC4214 retains deficient absolute astrometry and a 9.1% local quadrant mismatch.',ha='center',fontsize=10,color='#555555')
    fig.tight_layout();fig.savefig(output/'source-transfer.png',dpi=170,bbox_inches='tight');plt.close(fig)
    lines=['# Relative stellar-map transfer expanded to all five cleaned seeds','',
        'All five P5 reconstructions pass the declared overall relative-image gate in both checkerboard partitions. Four also pass the existing finite-footprint Gaia test on P1. **NGC4214 remains unsupported for absolute registration and has a local mismatch.** This is source-registration evidence, not a new gravity result or mass calibration.','',
        '![Relative image transfer](source-transfer.png)','',
        '| Galaxy | Shift dx, dy (P1 pixels) | Before RMS | After RMS | Reversed RMS | Split shift difference | Prior absolute gate |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['galaxy']} | {r['dx_p1_pixel']:.3f}, {r['dy_p1_pixel']:.3f} | {r['zero_shift_validation_rms_percent']:.2f}% | {r['corrected_validation_rms_percent']:.2f}% | {r['reversed_validation_rms_percent']:.2f}% | {r['split_shift_difference_p1_pixels']:.3f} px | {'pass' if r['prior_absolute_footprint_pass'] else 'insufficient'} |")
    lines+=['',
        'The [publisher P5 specification](https://irsa.ipac.caltech.edu/data/SPITZER/S4G/docs/P5_README.html) identifies stellar and nonstellar components of IRAC1 and describes cutouts and excluded ICA regions. We downloaded the five nonstellar counterparts, verified component units/coordinates and reconstructed their sum. Every nonzero ICA label is excluded; source maps and headers remain unchanged.','',
        'The P5-to-P1 mapping explicitly uses the previously selected plain TAN projection. Inherited SIP terms are recorded and ignored for this declared comparison. A separate Astropy core-WCS transform agrees within the frozen 1e-6 pixel tolerance. This does not validate arbitrary distortions or replace the prior Gaia test.','',
        'Translations are in P1 pixel coordinates: sample P1 at its nominal mapped coordinates plus dx,dy. Search covers all integer shifts within +/-8 pixels, followed by continuous refinement. Flux scale and background are fitted only to calibration pixels. Brightness selection uses the 80th percentile of calibration P5 reconstructed flux, then the same threshold on validation blocks. A full shift-search finite-footprint margin and ten-pixel block-edge guards keep all shift candidates on the same supported samples.','',
        'The first partition uses alternating 80-pixel blocks; the second reverses their roles under a new frozen configuration after the first result. Gates remain RMS below 5%, correlation above 0.99, positive fitted scale and a shift away from the search boundary. Both runs retain per-quadrant diagnostics, optimizer state and source hashes. The two partitions are sensitivity checks on the same exposed data, not independent observations or a posterior uncertainty distribution.','',
        'NGC4214 has 9.11% RMS and correlation 0.982 in one first-run validation quadrant despite passing the overall gate. Its P1 finite-footprint Gaia validation remains insufficient. Do not promote it to an absolute source-position pass. NGC2976, NGC3198 and NGC3521 now have explicit relative-transfer evidence in addition to NGC2903. Full source-noise, absolute-flux and 3D-depth admission remains incomplete.','',
        'NGC2903 was an already exposed control with historical fixed shift (-3,-1). Its best integer shift reproduces that choice; the new continuous fit and altered source-block selection produce a slightly different estimate. Earlier fields and their fixed shift are preserved. No old fit or hash was silently replaced.','',
        'Four numerical tests pass: independent bilinear interpolation; fractional-shift recovery against manufactured images on separate patches; zero-shift/axis convention; and explicit failure for featureless calibration. All prospective bindings for both runs match. Downloaded nonstellar images total 15,644,160 bytes per cached copy; raw files and sample arrays stay under work/private.','',
        'Next, use these measured offsets with retained split sensitivity and source masks in additional conditional source ensembles. An alignment pass does not measure stellar mass, remove missing matter, or admit a galaxy motion likelihood.','',
        '```text','python scripts/run_mond_atlas_stellar_transfer.py --output <new-first-output>',
        'python scripts/run_mond_atlas_stellar_transfer.py --config configs/mond_atlas_stellar_transfer_v2.json --output <new-reversed-output>',
        '```','',
        'For a full replay, use a new private_directory in a copied configuration as well as a new output directory; the recorded private samples belong to these frozen runs. The runner should not be used to overwrite their sample packets.']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    write_json(output/'summary.json',dict(status='RELATIVE_SOURCE_TRANSFER_VALIDATED_SCOPE_LIMITED',relative_pass_both_partitions=names,
        relative_and_prior_absolute_pass=[r['galaxy'] for r in rows if r['prior_absolute_footprint_pass']],
        ngc4214_local_and_absolute_limitations_retained=True,comparison_rows=rows,new_motion_scores=0,
        bindings={p.relative_to(ROOT).as_posix():digest(p) for p in [first/'summary.json',second/'summary.json',Path(__file__)]}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['first','second','output']:p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();report(a.first.resolve(),a.second.resolve(),a.output.resolve())
