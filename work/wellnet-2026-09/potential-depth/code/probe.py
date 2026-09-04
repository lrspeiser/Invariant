"""Probe VizieR for candidate group-scale catalogues.

Existence test: VizieR emits an explicit `#INFO Error=Table or Catalog not found`
line for a bad -source=, WHILE STILL RETURNING HTTP 200. That line, not the
status code, is the test. (The CDS FTP 200/404 test recorded elsewhere in this
programme is broken -- anti-bot wall returns 200 for everything.)
"""
import sys, os, urllib.parse, urllib.request, socket

UA = {"User-Agent": "gravity-programme-acquire/1.0 (research)"}
VIZ = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"

CANDIDATES = [
    ("J/ApJ/693/1142",   "Sun+2009 Chandra groups (43)"),
    ("J/A+A/573/A118",   "Lovisari+2015 XMM groups (20)"),
    ("J/ApJ/778/14",     "Gonzalez+2013 stellar+ICM baryons"),
    ("J/MNRAS/350/1511", "Osmond&Ponman 2004 GEMS groups"),
    ("J/A+A/601/A95",    "O'Sullivan+2017 CLoGS"),
    ("J/A+A/592/A12",    "Eckert+2016 XXL XIII"),
    ("J/ApJ/669/158",    "Gastaldello+2007 16 groups resolved M(r)"),
    ("J/ApJ/646/899",    "Humphrey+2006 ETG/groups resolved M(r)"),
    ("J/ApJ/640/691",    "Vikhlinin+2006 13 relaxed clusters M(r)"),
    ("J/MNRAS/451/1460", "Kettula+2015 CFHTLenS-XMM group WL"),
    ("J/ApJ/709/97",     "Leauthaud+2010 COSMOS group WL"),
    ("J/A+A/685/A106",   "Bulbul+2024 eRASS1 clusters"),
    ("J/A+A/691/A188",   "Bahar+2024 eRASS1 groups?"),
    ("J/A+A/661/A7",     "Bahar+2022 eFEDS groups"),
    ("J/A+A/621/A39",    "Ettori+2019 X-COP"),
    ("J/ApJ/887/76",     "Umetsu+2020 XXL HSC WL"),
    ("J/MNRAS/484/60",   "Mulroy+2019 LoCuSS"),
    ("J/MNRAS/497/4684", "Herbonnet+2020 WL"),
    ("J/ApJ/755/116",    "Sun+2012?"),
    ("J/MNRAS/449/199",  "?"),
    ("J/A+A/650/A104",   "Lovisari+2021?"),
    ("J/ApJS/174/117",   "Gastaldello?"),
    ("J/A+A/636/A15",    "Eckert+2020 X-COP profiles?"),
    ("J/ApJ/778/74",     "?"),
    ("J/MNRAS/470/4583", "?"),
    ("J/A+A/620/A5",     "Adami+2018 XXL-365-GC"),
]


def probe(src, timeout=25):
    url = VIZ + "?" + urllib.parse.urlencode(
        [("-source", src), ("-out", "**"), ("-out.max", "3")], safe="*/+.")
    try:
        txt = urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=timeout
        ).read().decode("utf-8", "replace")
    except Exception as e:
        return "NET", f"{type(e).__name__}: {e}", ""
    if "Error=Table or Catalog not found" in txt:
        return "MISSING", "", ""
    err = [l for l in txt.splitlines() if "#INFO\tError=" in l or l.startswith("#INFO Error")]
    if err:
        return "ERR", err[0][:120], ""
    title = ""
    for l in txt.splitlines():
        if l.startswith("#Title:"):
            title = l.split(":", 1)[1].strip()
            break
    names = [l.split(":", 1)[1].strip() for l in txt.splitlines() if l.startswith("#Name:")]
    return "OK", title[:90], " | ".join(names)[:110]


if __name__ == "__main__":
    socket.setdefaulttimeout(30)
    for src, desc in CANDIDATES:
        st, a, b = probe(src)
        print(f"{st:8s} {src:20s} {desc[:34]:34s} {a}")
        if b:
            print(f"{'':29s} tables: {b}")
        sys.stdout.flush()
