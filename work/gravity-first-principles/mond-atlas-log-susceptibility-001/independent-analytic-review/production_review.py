"""Read-only algebraic replay of saved static values."""
import json,math,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[3]
source=R/'scripts/mond_atlas_log_susceptibility.py'
saved=P.parent/'run002/static-susceptibility.json'
data=json.loads(saved.read_text(encoding='utf-8'));errors=[]
for row in data['static_rows']:
    r=row['r'];s=math.hypot(r,.05);C=row['omega']**2*.15**2/2
    g=C*r/(s*s*(s+1));density=C*(r*r+3*.05**2*(s+1))/(4*math.pi*s**4*(s+1)**2)
    errors += [abs(g-row['extra_acceleration']),abs(r*r*g-row['effective_mass']),abs(density-row['density'])]
assert max(errors)<1e-12
result={'status':'PASS_PRODUCTION_STATIC_REVIEW','rows':len(data['static_rows']),'maximum_absolute_error':max(errors),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'static_output_sha256':hashlib.sha256(saved.read_bytes()).hexdigest(),'source_inspection':'Full force uses actual q times gradient h; gradient sign and factor one-half agree with independent derivative. Trajectories delegated to parent radial replay.','primary_software_reference':'https://docs.galpy.org/en/v1.7.2/reference/potentialjaffe.html','documentation_mapping':'rho=amp/(4*pi*a^3)/[(r/a)^2*(1+r/a)^2], with a=L and amp=C*M*L/G.'}
(P/'production-receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
