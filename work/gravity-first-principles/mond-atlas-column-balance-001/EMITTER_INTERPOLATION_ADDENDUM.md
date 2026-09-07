# Match the validated force-table interpolation

Before emitter002, preserve emitter001 and its linear-interpolation calculation.
The force-table radial gates used PCHIP, whereas the initial emitter join used
linear interpolation. Version2 uses the validated PCHIP convention for force
components and the raw HI denominator, without changing source, pressure or
gravity parameters. Recompute all84 cases and retain both results. This is a
numerical join correction; no observed spectrum has been read. Independent
spectral-tail bounds remain conditional on the retained invalid-emitter masks.
