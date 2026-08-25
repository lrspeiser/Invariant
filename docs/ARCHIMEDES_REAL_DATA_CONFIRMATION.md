# Archimedes real-data confirmation

This control replays the bounded relation search against force-sensor readings published by
the University of Tennessee Physics 221 laboratory. The readings come from photographs of
four physical objects measured in air, submerged in water, and through displaced water.
External photographs are not copied into the repository; the frozen config binds every
transcription to its source URL and SHA-256 digest.

The search receives four force columns but no expected coefficient vector. It exhaustively
enumerates 112 normalized primitive homogeneous integer relations with coefficients from
`-2` through `2` and L1 norm at most `4`. On the first three objects it selects

```text
empty_container + object_air - object_submerged
    - container_with_displaced_water = 0
```

Equivalently, the loss of apparent object weight in water is approximately equal to the
weight of displaced water. This is the standard experimental form of Archimedes' principle,
not a new theory. The fourth object was held out, and the same relation also wins all four
leave-one-object-out searches.

The result is intentionally classified as limited confirmation. Only two of four residuals
include zero using display quantization alone, because the source provides no complete
uncertainty budget. In addition, all 24 permutations of displaced-water readings have the
same aggregate absolute residual. The small, low-variation dataset therefore cannot identify
object-level pairing or establish causality. It demonstrates real-data relation selection,
not proof or novelty.

Replay with:

```shell
python -m sigma_theory_compiler.archimedes_real_data_confirmation validate --root .
python -m pytest -q tests/test_archimedes_real_data_confirmation.py
```
