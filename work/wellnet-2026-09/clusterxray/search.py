"""Find ACIS observations of the six clusters, validated by POSITION not name."""
import math, subprocess, sys, urllib.parse, json

UA = "gravity-clustershear/1.0 (research)"
BASE = "https://cda.harvard.edu/srservices/ocatList.do"

TARGETS = {
 "ABELL_0370":        ("Abell 370",       39.9700,  -1.5767),
 "ABELL_1063S":       ("RXC J2248.7-4431",342.1833, -44.5306),
 "ABELL_2029":        ("Abell 2029",     227.7337,   5.7449),
 "ABELL_2744":        ("Abell 2744",       3.5862, -30.3997),
 "MACS_J0717_5p3745": ("MACS J0717.5+3745",109.3817, 37.7550),
 "MACS_J1149_5p2223": ("MACS J1149.5+2223",177.3987, 22.3985),
}
ALIASES = {"ABELL_1063S": ["RXC J2248.7-4431", "Abell S1063", "AS1063", "RXJ2248"],
           "ABELL_0370":  ["Abell 370", "A370"],
           "ABELL_2029":  ["Abell 2029", "A2029"],
           "ABELL_2744":  ["Abell 2744", "A2744"],
           "MACS_J0717_5p3745": ["MACS J0717.5+3745", "MACSJ0717.5+3745", "MACS0717"],
           "MACS_J1149_5p2223": ["MACS J1149.5+2223", "MACSJ1149.5+2223", "MACS1149"]}

def curl(url):
    p = subprocess.run(["curl","-s","--max-time","120","-A",UA,url], capture_output=True)
    return p.stdout.decode("utf-8","replace")

def sex2deg(ra, dec):
    h,m,s = [float(x) for x in ra.split()]
    dd = [float(x) for x in dec.split()]
    sign = -1.0 if dec.strip().startswith("-") else 1.0
    return (h+m/60+s/3600)*15.0, sign*(abs(dd[0])+dd[1]/60+dd[2]/3600)

def sep(ra1,de1,ra2,de2):
    d2r=math.pi/180
    return math.degrees(math.acos(max(-1,min(1,
        math.sin(de1*d2r)*math.sin(de2*d2r)+
        math.cos(de1*d2r)*math.cos(de2*d2r)*math.cos((ra1-ra2)*d2r)))))

out={}
for key,(name,ra0,de0) in TARGETS.items():
    found={}
    for alias in ALIASES[key]:
        url = BASE+"?"+urllib.parse.urlencode({"target":alias,"format":"text","resolve":"no"})
        txt = curl(url)
        lines=[l for l in txt.splitlines() if l and not l.startswith("#")]
        if len(lines)<3: continue
        hdr=lines[0].split("\t")
        try:
            iO,iI,iE,iT,iR,iD = (hdr.index("Obs ID"),hdr.index("Instrument"),
                                 hdr.index("Exposure (ks)"),hdr.index("Target Name"),
                                 hdr.index("RA"),hdr.index("Dec"))
        except ValueError: continue
        for l in lines[2:]:
            c=l.split("\t")
            if len(c)<=max(iO,iI,iE,iT,iR,iD): continue
            if not c[iI].startswith("ACIS"): continue
            try: ra,de = sex2deg(c[iR],c[iD])
            except Exception: continue
            d = sep(ra,de,ra0,de0)
            if d > 0.25:                       # POSITION detector: must be on target
                continue
            try: exp=float(c[iE])
            except ValueError: exp=0.0
            if exp < 5.0: continue             # too short to map
            found[c[iO].strip()] = dict(obsid=c[iO].strip(), instrument=c[iI],
                                        exposure_ks=exp, target=c[iT].strip(),
                                        ra=round(ra,4), dec=round(de,4),
                                        offset_deg=round(d,4))
    obs=sorted(found.values(), key=lambda o:-o["exposure_ks"])
    out[key]=obs
    tot=sum(o["exposure_ks"] for o in obs)
    print("%-20s %2d ACIS obs on target, %6.1f ks total   max offset %.3f deg"
          % (key, len(obs), tot, max([o["offset_deg"] for o in obs], default=0)))
    for o in obs[:4]:
        print("      obsid %-7s %-10s %6.1f ks  off %.3f  '%s'"
              % (o["obsid"],o["instrument"],o["exposure_ks"],o["offset_deg"],o["target"]))
json.dump(out, open(sys.argv[1] if len(sys.argv)>1 else "chandra_obs.json","w"), indent=1)
