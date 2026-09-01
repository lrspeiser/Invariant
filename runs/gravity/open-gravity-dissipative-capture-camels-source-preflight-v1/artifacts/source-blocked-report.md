# CAMELS same-IC dissipative-capture source preflight

Decision: **SOURCE BLOCKED**, not a model failure.

The exact CV_0 IllustrisTNG/IllustrisTNG_DM pair has 15 directly accessible paired snapshots from z=0.95 to z=0, both SubLink trees, all hydro group catalogs, the required hydro gas fields, and one z=0 group-matching file. The 160,096-byte matching file was downloaded and SHA-256 hashed without opening its HDF5 structure or rows. However, all 15 DMO group catalogs return HTTP 403, so their exact bytes, ETags, and required /IDs membership data cannot be receipted through the direct public URL. The inspected official documentation also does not establish a per-subhalo cross-tree mapping across the history. The cadence can identify merger intervals but makes first pericenter and coalescence interval-censored. No payload exposes an official cryptographic checksum.

TNG100 remains the preferred source. CAMELS maps or global summaries must not replace merger histories.
