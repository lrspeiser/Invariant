"""Acquire and parse primary binary-star sources without deriving mass labels from light."""
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'work/private/stellar-mass-audit-001'
SOURCES = {
    'benedict_source.dat': ('https://arxiv.org/src/1608.04775', '9186d9e39f68c9b91b632234d2e934e5fdc4270fedf486d472e452e124d9e2bd'),
    'mann_source.dat': ('https://arxiv.org/src/1811.06938', 'e0bc9c7202dfae15a3cfc84aa9213857a9f259d44664276e8ad9552b056a1c38'),
    'mann_table1.dat': ('https://cdsarc.cds.unistra.fr/ftp/J/ApJ/871/63/table1.dat', '86ff24fe62bb52410270887e5eaa7abfb284ea3e105fe9492df5174138e93714'),
    'mann_table5.dat': ('https://cdsarc.cds.unistra.fr/ftp/J/ApJ/871/63/table5.dat', 'ee0d3a5ca9ffc3d711b3ede1d7321571bc86b85806b87b70fd1ad0742be60a0a'),
    'mann_readme.txt': ('https://cdsarc.cds.unistra.fr/ftp/J/ApJ/871/63/ReadMe', None),
    'benedict_readme.txt': ('https://cdsarc.cds.unistra.fr/ftp/J/AJ/152/141/ReadMe', None),
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def canonical(name):
    name = name.replace(r'\,', '').replace(' ', '').upper()
    if name.startswith('GL'):
        name = 'GJ' + name[2:]
    return {'G250-029': 'GJ3412', 'G193-027': 'GJ3421', 'YYGEM': 'GJ278'}.get(name, name)


def numbers(cell):
    return [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?', cell)]


def section(tex, marker):
    # Drop commented-out obsolete tables before locating the active table.
    tex = '\n'.join(line.split('%')[0] for line in tex.splitlines())
    body = tex.split(marker, 1)[1].split(r'\startdata', 1)[1].split(r'\enddata', 1)[0]
    return [line.split('&') for line in body.splitlines() if '&' in line]


def parse_sources():
    import numpy as np
    btex = (PRIVATE / 'benedict_source_MLRv16.tex').read_text(encoding='utf-8')
    mtex = (PRIVATE / 'mann_source_Masses_1.tex').read_text(encoding='utf-8')
    stars, missing = [], []
    for cells in section(btex, r'\label{tbl-MMVMK}'):
        assert len(cells) == 13
        full = canonical(cells[0].strip())
        system, components = full[:-2], full[-2:]
        for j, component in enumerate(components):
            m, em = numbers(cells[1+j])
            v = numbers(cells[6+j]); k = numbers(cells[11+j])
            if len(v) != 2 or len(k) != 2:
                missing.append(system + component); continue
            stars.append(dict(name=system+component, system=system, mass=m, e_mass=em,
                              MV=v[0], e_MV=v[1], MK=k[0], e_MK=k[1]))
    assert len(stars) == 28 and len(set(r['system'] for r in stars)) == 14
    assert missing == ['GJ54A', 'GJ54B']
    pairs = []
    for s in (PRIVATE / 'mann_table1.dat').read_text(encoding='utf-8').splitlines():
        if not s.strip(): continue
        f = lambda a, b: float(s[a-1:b])
        r = dict(system=canonical(s[2:12].strip()), components=s[13:15].strip(),
                 Ks=f(38,43), e_Ks=f(45,49), Ks_flag=s[50:51].strip(),
                 delta=f(53,57), e_delta=f(59,63), mass=f(65,71), e_mass=f(73,79),
                 feh=f(81,85), feh_flag=s[86:87].strip(), plx=f(89,95), e_plx=f(97,101))
        apparent = r['Ks'] + 2.5*np.log10(1+10**(-0.4*r['delta']))
        r['MK1'] = float(apparent + 5*np.log10(r['plx']) - 10)
        r['MK2'] = r['MK1'] + r['delta']
        pairs.append(r)
    assert len(pairs) == 62 and len(set(r['system'] for r in pairs)) == 62
    orbits = {}
    for s in (PRIVATE / 'mann_table5.dat').read_text(encoding='utf-8').splitlines():
        if not s.strip(): continue
        orbits[canonical(s[:10].strip())] = dict(q=float(s[173:182]), e_q=float(s[183:192]))
    for r in pairs:
        if r['system'] in orbits:
            r.update(orbits[r['system']])
            r['orbital_mass_replay'] = r['q'] / (r['plx']/1000)**3
    external = []
    for cells in section(mtex, r'\tablecaption{Targets with Individual Masses }'):
        assert len(cells) == 7
        full = canonical(cells[0].strip()); system = canonical(full[:-1])
        m, em = numbers(cells[1]); k, ek = numbers(cells[2]); p, ep = numbers(cells[3])
        external.append(dict(name=full, system=system, mass=m, e_mass=em, Ks=k, e_Ks=ek,
                             plx=p, e_plx=ep, MK=float(k+5*np.log10(p)-10), kind=cells[4].strip()))
    assert len(external) == 29 and sum(r['kind']=='EB' for r in external) == 22
    seen = {r['system'] for r in pairs}
    admitted = [r for r in external if r['kind']=='EB' and r['system'] not in seen]
    assert len(admitted) == 22 and len(set(r['system'] for r in admitted)) == 11
    return dict(benedict=stars, benedict_missing_photometry=missing, mann=pairs,
                external_all=external, external_EB=admitted)


def main():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (url, expected) in SOURCES.items():
        path = PRIVATE / name
        if not path.exists():
            r = requests.get(url, timeout=50); r.raise_for_status(); path.write_bytes(r.content)
        payload = path.read_bytes(); digest = hashlib.sha256(payload).hexdigest()
        if expected is not None: assert digest == expected, (name, 'source content changed')
        manifest.append(dict(path=str(path.relative_to(ROOT)), url=url, bytes=len(payload), sha256=digest))
        if name.endswith('_source.dat'):
            prefix = name.removesuffix('.dat')
            with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                for member in archive.getmembers():
                    if member.isfile() and member.name.endswith('.tex'):
                        # Never extract archive paths into the workspace.
                        (PRIVATE / (prefix+'_'+Path(member.name).name)).write_bytes(archive.extractfile(member).read())
    data = parse_sources()
    save(PRIVATE / 'parsed-binaries.json', data)
    receipt = dict(sources=manifest, counts={k:len(v) for k,v in data.items()},
                   exclusions='Benedict mixed-distance supplementary table not used; GJ54 lacks component K photometry; external astrometric/triple systems not counted as independent EB validation',
                   data_path=str((PRIVATE/'parsed-binaries.json').relative_to(ROOT)),
                   data_sha256=hashlib.sha256((PRIVATE/'parsed-binaries.json').read_bytes()).hexdigest())
    save(PRIVATE/'source-manifest.json', receipt)
    print(json.dumps(receipt['counts']))


if __name__ == '__main__': main()
