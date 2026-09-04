# -*- coding: utf-8 -*-
"""List a remote .tar's members using HTTP Range requests, then fetch selected members."""
import requests, sys

def _get(url, start, length):
    h = {"Range":"bytes=%d-%d" % (start, start+length-1)}
    r = requests.get(url, headers=h, timeout=300)
    if r.status_code not in (200,206): raise RuntimeError("range %d HTTP %d" % (start, r.status_code))
    return r.content

def index_tar(url, maxmembers=4000, verbose=True):
    r = requests.head(url, timeout=120, allow_redirects=True)
    total = int(r.headers.get("Content-Length","0"))
    if verbose: print("  total bytes:", total, " accept-ranges:", r.headers.get("Accept-Ranges"))
    off = 0; members = []; longname=None
    # read headers in chunks; we jump member-to-member
    while off < total and len(members) < maxmembers:
        hdr = _get(url, off, 512)
        if len(hdr) < 512: break
        if hdr[:512] == b"\0"*512:
            # end-of-archive marker (two of them); stop
            break
        name = hdr[0:100].rstrip(b"\0").decode("utf-8","replace")
        prefix = hdr[345:500].rstrip(b"\0").decode("utf-8","replace")
        szf = hdr[124:136].rstrip(b"\0 ").decode("ascii","replace").strip()
        try: size = int(szf, 8) if szf else 0
        except ValueError: size = 0
        typeflag = hdr[156:157].decode("ascii","replace")
        data_off = off + 512
        padded = ((size + 511)//512)*512
        if typeflag == "L":   # GNU long name
            longname = _get(url, data_off, size).rstrip(b"\0").decode("utf-8","replace")
            off = data_off + padded; continue
        if longname: name = longname; longname=None
        elif prefix: name = prefix + "/" + name
        if typeflag in ("0","","0\0"):
            members.append({"name":name,"size":size,"offset":data_off})
        off = data_off + padded
    return total, members

def fetch_member(url, m, outpath):
    data = _get(url, m["offset"], m["size"])
    with open(outpath,"wb") as f: f.write(data)
    return len(data)

if __name__=="__main__":
    u = "https://archive.stsci.edu/hlsps/hff-deepspace/abell2744/multi/hlsp_hff-deepspace_hst_acs-wfc3_abell2744-clu_multi_v1_catalogs.tar"
    total, mem = index_tar(u)
    print("members:", len(mem))
    for m in mem:
        if m["size"]>0 and not m["name"].endswith((".fits",".png")):
            print("  %10d  %s" % (m["size"], m["name"]))
