"""Parse VizieR multi-table asu-tsv responses into per-table cleaned TSVs +
manifests. VizieR concatenates every table of a catalogue into one response, so a
naive row count is inflated by the interleaved headers of later tables."""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import write_manifest

D = os.path.dirname(os.path.abspath(__file__))


def parse_vizier(path):
    """Return (catalog_id, title, [tables]). Each table:
    {name, columns:[{name,unit,desc}], rows:[[...]], nrows}"""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    cat = title = None
    # column descriptions declared in the header block: #Column <name> (fmt) desc
    coldesc = {}
    for l in lines:
        if l.startswith("#Name:") and cat is None:
            cat = l.split(":", 1)[1].strip()
        elif l.startswith("#Title:") and title is None:
            title = l.split(":", 1)[1].strip()
        elif l.startswith("#Column"):
            p = l.split("\t")
            if len(p) >= 4:
                nm = p[1].strip()
                desc = p[3].strip()
                coldesc.setdefault(nm, desc)

    tables = []
    i = 0
    n = len(lines)
    while i < n:
        l = lines[i]
        if l.startswith("#Table"):
            tname = l.split("\t", 1)[1].strip().rstrip(":") if "\t" in l else l
            # skip forward to the header line: first non-# non-blank line
            j = i + 1
            while j < n and (lines[j].startswith("#") or not lines[j].strip()):
                j += 1
            if j >= n:
                break
            hdr = lines[j].split("\t")
            units = lines[j + 1].split("\t") if j + 1 < n else [""] * len(hdr)
            # line j+2 is the ---- separator
            k = j + 3
            rows = []
            while k < n:
                lk = lines[k]
                if lk.startswith("#") or not lk.strip():
                    # a blank/# line ends the data block
                    if lk.startswith("#Table") or lk.startswith("#RESOURCE"):
                        break
                    if not lk.strip():
                        # blank line: VizieR ends tables with a blank line
                        break
                    k += 1
                    continue
                rows.append(lk.split("\t"))
                k += 1
            cols = []
            for ci, cn in enumerate(hdr):
                u = units[ci].strip() if ci < len(units) else ""
                cols.append({"name": cn.strip(), "unit": u,
                             "description": coldesc.get(cn.strip(), "")})
            tables.append({"name": tname, "columns": cols,
                           "rows": rows, "nrows": len(rows)})
            i = k
        else:
            i += 1
    return cat, title, tables


if __name__ == "__main__":
    log = json.load(open(os.path.join(D, "crot_vizier_fetch_log.json"), encoding="utf-8"))
    summary = []
    for rec in log:
        if not rec["ok"]:
            continue
        raw = os.path.join(D, rec["file"])
        cat, title, tables = parse_vizier(raw)
        print("\n##### %s  [%s]" % (rec["short"], cat))
        print("      %s" % title)
        tinfo = []
        for t in tables:
            print("   table %-40s nrows=%-6d ncols=%-3d  %s"
                  % (t["name"], t["nrows"], len(t["columns"]),
                     [c["name"] for c in t["columns"]][:10]))
            tinfo.append({"table": t["name"], "nrows": t["nrows"],
                          "ncols": len(t["columns"]),
                          "columns": [{"name": c["name"], "unit": c["unit"],
                                       "description": c["description"]}
                                      for c in t["columns"]]})
        summary.append({"short": rec["short"], "catalog": cat, "title": title,
                        "vizier_catalog_requested": rec["catalog"],
                        "raw_file": rec["file"], "url": rec["url"],
                        "tables": tinfo,
                        "total_data_rows": sum(t["nrows"] for t in tables)})
    with open(os.path.join(D, "crot_vizier_table_index.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nWROTE crot_vizier_table_index.json")
