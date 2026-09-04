"""VizieR metadata / keyword search helper. Prints catalogue IDs + descriptions.

Usage: python _search_vizier.py "keyword phrase" [more phrases...]
"""
import sys
from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = -1
Vizier.TIMEOUT = 180

for q in sys.argv[1:]:
    print("=" * 78)
    print("QUERY:", q)
    print("=" * 78)
    try:
        cats = Vizier.find_catalogs(q, max_catalogs=60)
    except Exception as e:
        print("  ERROR:", type(e).__name__, e)
        continue
    if not cats:
        print("  (no catalogues)")
        continue
    for k, v in cats.items():
        print(f"  {k}   ::   {v.description}")
    print()
