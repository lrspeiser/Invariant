# Lane 7 V4: ESO 325-G004 extended-source preflight

- Exact source bytes: **PASS**.
- FITS structure without array decode: **PASS_EXACT_BYTES_AND_FITS_STRUCTURE_ONLY**.
- Target-free one-state solver: **PASS_TARGET_FREE_SHARED_STATE**.
- Real ESO score: **BLOCK_BEFORE_SCIENTIFIC_ARRAY_DECODE**.
- SLACS confirmation: **SEALED; zero response values opened**.

The public paper supplement corrects the earlier Voronoi assumption: its MUSE analysis used 0.6-arcsec square pixels. The archive contains the raw/reduced HST and MUSE products, but not the registered paper mask, PSFs, kinematic table/covariance, posterior, or joint likelihood. V4 therefore freezes an independent reduction and refuses to invent covariance. The synthetic shared-state gate can pass without granting permission to decode or score ESO arrays.
