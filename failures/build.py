"""build.py -- generate the failure catalogue from one structured definition.

Every file under artefacts/, archive-traps/ and eliminated/ is written from the
CATALOGUE below, so index.json and the human-readable pages cannot drift apart.
Edit here, re-run, commit.

    python build.py
"""
from __future__ import annotations

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ARTEFACTS = [
    dict(
        id="01-label-control",
        title="The label control killed seven variables at once",
        run="the first environmental variable search",
        claimed="Seven environmental variables predict the gravitational residual.",
        number="1.1% gain on the real data",
        null="6.1% gain on data containing only survey structure and no physics",
        mechanism=(
            "The search space was rich enough that the selection procedure could fit "
            "survey geometry. Running it on structure-only data did not merely "
            "reproduce the result -- it scored HIGHER, because the structure-only data "
            "had no physics competing with the geometry."),
        control="Run the identical search on data that cannot contain the answer.",
        lesson=(
            "A null that beats the data is not a weak result, it is a diagnosis. The "
            "method was measuring the survey."),
        cost="One extra run of a search that had already been written."),
    dict(
        id="02-retracted-correlation",
        title="A correlation whose sign belonged to the estimator",
        run="Run K.2 (retraction)",
        claimed="An environmental correlation of rho_p = -0.304.",
        number="-0.304",
        null="the sign did not survive re-derivation",
        mechanism=(
            "The estimator itself carried a sign preference that was mistaken for a "
            "property of the sky."),
        control="Re-derive the estimator independently before quoting its sign.",
        lesson=(
            "Recorded because retraction is the rarest outcome in this catalogue. The "
            "programme record carries the retraction next to the original claim rather "
            "than replacing it."),
        cost="One re-derivation."),
    dict(
        id="03-voidfinder-bias",
        title="38 sigma from data containing no effect",
        run="Run Q",
        claimed="A redshift path-length effect in void catalogues.",
        number="30-38 sigma",
        null="the same significance on data constructed with no path effect at all",
        mechanism=(
            "The void-finding algorithm imprinted its own bias on the path-length "
            "statistic, and the bias differed between algorithms -- VoidFinder and "
            "REVOLVER gave different amounts of it."),
        control=(
            "Simulate the null separately for each algorithm and subtract it. A "
            "transverse decomposition orthogonal to distance cut the bias 60x for "
            "VoidFinder and 24x for REVOLVER, and still did not reach zero."),
        lesson=(
            "The null must be simulated per algorithm and subtracted, never assumed "
            "to be zero."),
        cost="One synthetic catalogue per void-finder."),
    dict(
        id="04-shared-shape-factor",
        title="Both axes contained the same term",
        run="Run Z.4",
        claimed="A correlation between a shape factor and a gravitational quantity.",
        number="a confident correlation of plausible size",
        null="the correlation is a restatement of the shared term",
        mechanism=(
            "The quantity being correlated appeared inside the construction expression "
            "of BOTH axes. The correlation measured the sharing, not the physics."),
        control=(
            "Write down the construction expression for each axis and look for common "
            "factors, before quoting any correlation."),
        lesson=(
            "This check needs no compute and no data. It takes minutes and it has now "
            "caught two of the nine."),
        cost="Minutes, with a pen."),
    dict(
        id="05-fit-noise-with-a-sign",
        title="Published fit noise with a preferred sign",
        run="the potential-depth lane",
        claimed="A significant negative potential-depth effect.",
        number="-6.6 sigma",
        null="-0.0666 +- 0.0101 with the true effect set to zero",
        mechanism=(
            "Noise in a PUBLISHED X-ray density fit, propagated through the estimator, "
            "did not average out. It had a preferred direction, and the estimator "
            "converted it into a significant negative value."),
        control=(
            "Set the true effect to zero, redraw the catalogue inputs at their "
            "published uncertainties, and push them through the actual estimator."),
        lesson=(
            "The bias belonged to the catalogue that was combined in, not to the "
            "measurement being reported. Inherited uncertainty is still your problem."),
        cost="A Monte Carlo over published error bars."),
    dict(
        id="06-fit-noise-on-an-amplitude",
        title="The same noise biased an amplitude low by a factor of 2.2",
        run="the slip lane",
        claimed="A measured amplitude Sigma_s below unity.",
        number="factor 2.2 low, at 17 sigma",
        null=("-0.026 dex at error scale 0.25, -0.125 at 0.50, -0.336 at 1.00 -- "
              "monotonic in the input error"),
        mechanism=(
            "Identical mechanism to case 05, but applied to a pure amplitude with no "
            "free parameter able to absorb it. The largest artefact in the set."),
        control=(
            "Scale the input errors up and down and watch the estimator move. A real "
            "amplitude does not track the error scale of its inputs."),
        lesson=(
            "Every such amplitude must be read against this null rather than against "
            "1. Its factor-two width became the lane's dominant uncertainty."),
        cost="Three Monte Carlo runs at different error scales."),
    dict(
        id="07-axis-from-shot-noise",
        title="A cosmic axis made entirely of shot noise",
        run="Run AP",
        claimed="A preferred global axis in the well network.",
        number="growth anisotropy 12.4 at 100 wells",
        null="1.10 at 100,000 wells -- the amplitude is set by particle count",
        mechanism=(
            "The tensor estimators manufactured a global axis out of catalogue shot "
            "noise. In the continuum limit the effect is exactly zero; the ensemble "
            "quadrupole falls as roughly N^-0.557."),
        control="Vary the sample size and watch the amplitude.",
        lesson=(
            "A real anisotropy is indifferent to how many objects you used. A "
            "shot-noise anisotropy is a function of it."),
        cost="Re-running the same estimator at three sample sizes."),
    dict(
        id="08-r500-is-r",
        title="A trend against r/R500 was a trend against r",
        run="Run AT",
        claimed='A cluster trend "organised by r/R500", implying a scale-free result.',
        number="the trend itself survived",
        null="the normalisation carries no information the raw radius does not",
        mechanism=(
            "R500 was derived from the same profile being binned, so dividing by it "
            "is an exact identity: r/R500 IS r, reparameterised. The ratio was not a "
            "scale-free coordinate."),
        control=(
            "Check whether the normalising quantity is independent of the quantity "
            "being normalised."),
        lesson=(
            "The mildest failure here and the most common in the literature. The "
            "result lived; every sentence claiming it was scale-free had to be "
            "rewritten."),
        cost="One line of algebra."),
    dict(
        id="09-undersized-null",
        title="A permutation null too small to show its own width",
        run="Run BS, 2026-09-07",
        claimed=(
            "The asymmetry of a cluster's gas predicts the asymmetry of its "
            "gravitational lensing."),
        number="+2.67 sigma, p = 0.000, at 40 permutations",
        null="+1.64 sigma, p = 0.055, at 200 permutations on identical data",
        mechanism=(
            "The null's standard deviation was 0.0095 at 40 draws and 0.0144 at 200. "
            "Forty draws could not resolve the null's own width, so the measured gain "
            "of +0.0146 was compared against a distribution that looked far tighter "
            "than it is. Nothing about the data changed between the two runs."),
        control=(
            "Increase the permutation count until the null's standard deviation stops "
            "moving. Plot sd against draw count."),
        lesson=(
            "An undersized permutation null is not a conservative control. It inflates "
            "significance, and it does so silently. This one was found while writing "
            "up the other eight."),
        cost="160 more permutations -- about four minutes."),
]

TRAPS = [
    dict(
        id="vizier-fuzzy-fallback",
        service="VizieR asu-tsv",
        asked="a specific catalogue identifier",
        returned=(
            "HTTP 200 and a DIFFERENT, REAL catalogue -- a near-infrared young "
            "stellar object survey (J/MNRAS/430/1125) served in place of a cluster "
            "redshift catalogue. Twelve identifiers were affected in one lane. In "
            "another run the payload was the wrong PAPER entirely, and in a third the "
            "response carried CatalogsExamined=10213 after a fuzzy fallback."),
        detector=(
            "Three detectors, ALL required, none sufficient alone. D1: the #Name: line "
            "must echo the EXACT identifier requested. D2: refuse any payload "
            "containing CatalogsExamined. D3: #Title: must match the expected author "
            "and year."),
        note=(
            "A zero-row response is not an absence until D1-D3 have passed. This "
            "programme's own client asserted the D1 check in its docstring and never "
            "performed it -- the echo was computed into a variable and discarded."),
        code="work/wellnet-2026-09/goldcluster/archives.py"),
    dict(
        id="chandra-ocat-ignores-position",
        service="Chandra ocatList.do",
        asked="observations near a sky position (ra, dec, radius parameters)",
        returned=(
            "HTTP 200 with the position parameters silently ignored. A search "
            "centred on Abell 370 at declination -1.6 deg returned engineering "
            "pointings at declination -80 deg. The table was large and correctly "
            "formed."),
        detector=(
            "Resolve targets by NAME, then reject any returned observation whose own "
            "RA/Dec lies more than a set tolerance from the target. The position is "
            "the detector, never the query parameter."),
        note=(
            "A later cross-match by eRASS1 designation returned zero matches for every "
            "cluster. Tested against Abell 2744's own J-name -- which has 98 ACIS "
            "observations -- it also returned zero. The name resolver does not handle "
            "those designations, so the zero was a broken search, not an absence. "
            "Fixed by downloading all 15,734 ACIS observations and matching locally, "
            "with two positive controls that must fire before any null is believed."),
        code="work/wellnet-2026-09/clusterxray/search.py"),
    dict(
        id="tap-error-as-http-200",
        service="IVOA TAP endpoints (NOIRLab Astro Data Lab)",
        asked="an ADQL query",
        returned=(
            "HTTP 200 wrapping a VOTable whose QUERY_STATUS is ERROR. Parsed as CSV "
            "it reads as zero rows -- a confident false negative. Triggered here by "
            "CAST(FLOOR(x) AS INTEGER) and by GROUP BY on an expression, both of "
            "which the ADQL parser rejects; and separately by a whole-table COUNT, "
            "which returns HTTP 504 after several minutes."),
        detector=(
            "Test for a leading <?xml envelope before parsing anything as data."),
        note=(
            "To sample a very large table without a full scan, use MOD(id, N) = 0 "
            "rather than TOP N, which returns a contiguous block, not a sample."),
        code="work/wellnet-2026-09/goldcluster/archives.py"),
    dict(
        id="acis-vignetting-mimics-a-cool-core",
        service="Chandra ACIS event lists",
        asked="a gas temperature map",
        returned=(
            "An apparent cool core in every one of six clusters, relaxed and merging "
            "alike. Hardness rose +0.124 from centre to 8 arcmin with only 0.030 "
            "scatter BETWEEN clusters -- the same profile in a relaxed cool-core "
            "cluster at z=0.077 as in a quadruple merger at z=0.546."),
        detector=(
            "Two clusters of different dynamical state cannot share a thermal "
            "profile. When they appear to, the profile belongs to the instrument. "
            "The cause is energy-dependent vignetting: soft photons lose more "
            "effective area off-axis, so the band ratio hardens outward whatever the "
            "gas is doing."),
        note=(
            "Corrected properly with exposure maps from CIAO and CALDB. Corrected "
            "here empirically, by removing each cluster's own smoothed radial median "
            "-- which is valid only because vignetting is azimuthally symmetric by "
            "construction, and which removes any genuinely spherical thermal "
            "structure along with it."),
        code="work/wellnet-2026-09/clusterxray/maps.py"),
    dict(
        id="binning-artefact-resembles-the-signal",
        service="the correction itself",
        asked="remove the radial gradient above",
        returned=(
            "Concentric rings across every map, one at each annular bin edge. Rings "
            "in a cluster thermal map resemble shock fronts, which are exactly what "
            "such a map is searched for."),
        detector=(
            "Build the radial profile in fine bins, smooth it, then interpolate, "
            "rather than subtracting bin by bin."),
        note=(
            "The most dangerous class in this list, because the artefact resembles "
            "the target. A correction must not manufacture the feature it is "
            "correcting for."),
        code="work/wellnet-2026-09/clusterxray/maps.py"),
]

ELIMINATED = [
    dict(id="new-gravity-law", title="A new law of gravity",
         status="NOT FOUND",
         detail=(
             "Nine independent lanes plus a cross-domain synthetic steering system "
             "and a 400-card replay. No robust positive evidence for a modification "
             "to gravity anywhere in galaxies, weak lensing, cluster outskirts or "
             "wide binaries. Six of the nine lanes never opened a protected "
             "response, so most are BLOCKED BEFORE SCORING rather than empirically "
             "refuted -- a block is not a theory rejection.")),
    dict(id="every-candidate-family-round-2", title="Every candidate family, round 2",
         status="ELIMINATED",
         detail=(
             "All proposed families eliminated. What survived is a cluster-only "
             "excess organised by radius, and both blind-checked holdouts land on "
             "the standard empirical relation.")),
    dict(id="seven-modified-gravity-universes-are-one-class",
         title="Seven modified-gravity universes are observationally identical",
         status="INDISTINGUISHABLE",
         detail=(
             "Run BF forward-modelled ten universes through identical instruments. "
             "At amplitudes the corpus can actually detect, seven of the ten form ONE "
             "observational class -- every modified-gravity variant proposed by this "
             "programme is indistinguishable from every other. Only dark matter and "
             "pure systematics stand apart. Nine pairs are separated by none of three "
             "simulated improvements: 16x source density, 4x better systematics, 1.5x "
             "larger survey.")),
    dict(id="bounded-response", title="Any bounded response",
         status="NO-GO",
         detail=(
             "A bounded response can only renormalise G. Proved, data-free.")),
    dict(id="speed-only-modification", title="Changing propagation speed alone",
         status="NO-GO",
         detail=(
             "Cannot enhance a stationary Poisson-limit field. The programme's "
             "strongest theorem, and data-free.")),
    dict(id="redshift-families-e4-e5", title="Redshift families E4 and E5",
         status="KILLED",
         detail="E6 survives but is empty."),
    dict(id="density-contrast-law", title="The density-contrast law",
         status="DEAD",
         detail="The correct repair of B2/B5/B6, and still dead."),
    dict(id="nonlocal-kernel", title="The nonlocal kernel",
         status="DEAD",
         detail="Numerics sound, physics dead. Died structurally, not on data."),
    dict(id="geometric-redshift-class", title="The geometric redshift class",
         status="EXCLUDED",
         detail=(
             "Void path length crossed with Planck: measured and excluded at the "
             "candidate's own amplitude. Time dilation kills the mechanism in its "
             "natural form.")),
    dict(id="cluster-asymmetry-signal",
         title="Baryon asymmetry predicting lensing asymmetry",
         status="NULL",
         detail=(
             "Run BS, 2026-09-07. 52 clusters, 1,201 cells, nothing azimuthally "
             "averaged. Gain +0.0146 against an azimuthal-permutation null of "
             "-0.0091 +- 0.0144: z = +1.64, p = 0.055. The limiting factor is "
             "measured, not guessed -- median signal-to-noise per cell is 0.78.")),
    dict(id="million-law-search", title="A 679,120-model law search on cluster lensing",
         status="NO NEW LAW",
         detail=(
             "Every subset of size 1-4 over 64 atoms, fitted exactly. A SINGLE term "
             "reaches 78% of what four free parameters reach, and that term is close "
             "to what an isothermal sphere gives for free. The other three parameters "
             "buy 0.021 more gain while the label-scramble null climbs from 0.026 to "
             "0.046. More search in the same space produces overfitting faster, not "
             "discovery.")),
]


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    io.open(path, "w", newline="\n", encoding="utf-8").write(text)


def main():
    for i, a in enumerate(ARTEFACTS, 1):
        w(os.path.join(HERE, "artefacts", a["id"] + ".md"), """# {n:02d}. {title}

**Where:** {run}

## What was claimed

{claimed}

|  |  |
|---|---|
| the number that looked real | **{number}** |
| the same number under control | **{null}** |

## Why it happened

{mechanism}

## What caught it

{control}

## The lesson

{lesson}

**Cost of the control:** {cost}
""".format(n=i, **a))

    for t in TRAPS:
        w(os.path.join(HERE, "archive-traps", t["id"] + ".md"), """# {service}

**Asked for:** {asked}

## What came back

{returned}

## The detector

{detector}

## Note

{note}

**Implemented in:** `{code}`
""".format(**t))

    w(os.path.join(HERE, "eliminated", "README.md"),
      "# Eliminated, blocked and null\n\n"
      "What the programme looked for and did not find. A **block** is not a "
      "refutation: it means the candidate never reached an interpretable score "
      "against a protected response, usually because the data to score it does "
      "not exist publicly.\n\n"
      + "\n".join(
          "## %s\n\n**%s**\n\n%s\n" % (e["title"], e["status"], e["detail"])
          for e in ELIMINATED))

    index = dict(
        generated_by="failures/build.py",
        counts=dict(artefacts=len(ARTEFACTS), archive_traps=len(TRAPS),
                    eliminated=len(ELIMINATED)),
        artefacts=[dict(n=i, id=a["id"], title=a["title"], run=a["run"],
                        number=a["number"], null=a["null"], control=a["control"])
                   for i, a in enumerate(ARTEFACTS, 1)],
        archive_traps=[dict(id=t["id"], service=t["service"], asked=t["asked"],
                            detector=t["detector"], code=t["code"]) for t in TRAPS],
        eliminated=[dict(id=e["id"], title=e["title"], status=e["status"])
                    for e in ELIMINATED])
    w(os.path.join(HERE, "index.json"), json.dumps(index, indent=1) + "\n")

    print("wrote %d artefacts, %d archive traps, %d eliminated entries, index.json"
          % (len(ARTEFACTS), len(TRAPS), len(ELIMINATED)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
