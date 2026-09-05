# Independent stellar alignment and projected-source milestone

Twelve stellar fields were checked against independent Gaia DR3 foreground-star positions. Coordinate interpretations were selected on calibration stars, then evaluated on separate stars. All twelve selected the linear TAN interpretation rather than applying the ambiguous SIP coefficients. Only NGC2903 passed the strict overall gate: 67 stars, held-out median offset about 0.17 arcsec. A failed gate does not prove bad astrometry: faint Gaia stars without detected infrared counterparts also fail this matching procedure.

The NGC2903 P5 cleaned maps had an additional relative offset. A source-only comparison to the Gaia-checked P1 image found a translation of (-3,-1) P1 pixels, about 2.37 arcsec in magnitude. After applying that shift in memory, stellar plus nonstellar flux agrees with P1 at 1.15% RMS on validation blocks after calibration-only scale/background adjustment. Raw FITS files are unchanged. The first failed transfer is retained in ngc2903-matter-001.

A common 48-arcsec beam model with the official ICA masks, HI coverage and CO error coverage admits 27 of 242 geometric positions. In this selected region, the median atomic-gas fraction is 6.5% of the nominal projected stellar+atomic+molecular model. Twelve positions have CO below the conservative three-error threshold. Conversion ranges are sensitivity assumptions, not probability intervals; volume density and a complete uncertainty model are not established.

Sources: https://www.cosmos.esa.int/web/gaia-users/archive/programmatic-access ; https://irsa.ipac.caltech.edu/data/SPITZER/S4G/docs/P5_README.html ; https://www.iram.fr/ILPA/LP001/README . Exact queries and asset hashes are in each object receipt.
