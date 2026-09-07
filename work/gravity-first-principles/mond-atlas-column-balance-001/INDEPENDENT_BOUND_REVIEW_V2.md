# PCHIP emitter002 envelope replay

The parent retained emitter001 after identifying its linear force interpolation as inconsistent with the force table's validated PCHIP interpolation. This second independent review reads emitter002 and leaves the original review and receipt intact.

`independent_bound_review_v2.py` independently reconstructs the same A/H operators and verifies all252 reported envelope values across the84 PCHIP cases. The maximum remains5.524681958109126e-6mJy/beam at emission multiplier1, or1.1049363916218251e-5mJy/beam at multiplier2. Its inequality, channel-width treatment, units and limits are documented in `INDEPENDENT_BOUND_REVIEW.md`; they do not depend on the force interpolation scheme.

The receipt remains conditional on the supplied invalid-emitter masks and their beam-weighted positive flux. This review does not independently recompute force or pressure interpolation, nor does it make any of the84 failed global steady solutions valid. No source spectra or gravity-response targets were accessed. Both versions and their exact input hashes remain available.
