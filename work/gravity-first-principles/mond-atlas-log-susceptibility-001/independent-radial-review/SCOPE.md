# Independent radial replay of the logarithmic kernel

Use relative radial coordinates, conserved specific angular momentum and Radau rather than the production Cartesian DOP853 solver. Reproduce all eight fine cases and 401 samples each from run002. Fixed tolerances1e-10/1e-12, maxstep0.05; compare radius, oscillator coordinate, total energy and internal signed energy to1e-7 absolute. Require radial energy drift below1e-7. No production functions are imported, and no observed source or velocity is used. This numerical verification does not validate the proposed physical mechanism.
