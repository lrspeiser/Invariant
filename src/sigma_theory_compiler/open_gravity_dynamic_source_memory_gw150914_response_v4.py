"""Pre-response correction: exact nuisance-grid injections and Whittle scaling."""
from __future__ import annotations
import argparse, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import h5py
import numpy as np
from scipy import signal
from . import open_gravity_dynamic_source_memory_gw150914_response_v3 as v3

CONFIG=Path("configs/open_gravity_dynamic_source_memory_gw150914_response_v4.json")
MODULE=Path("src/sigma_theory_compiler/open_gravity_dynamic_source_memory_gw150914_response_v4.py")
TEST=Path("tests/test_open_gravity_dynamic_source_memory_gw150914_response_v4.py")

def _root(): return Path(__file__).resolve().parents[2]
def _hash(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def load_config(root=None): return json.loads(((root or _root())/CONFIG).read_text(encoding="utf-8"))
def validate(root:Path,c:Mapping[str,Any]):
    p=c["predecessor"]
    for key,pathkey in (("v3_config_sha256","v3_config_path"),("v3_module_sha256","v3_module_path"),("v3_source_receipt_sha256","v3_source_receipt_path"),("v3_schema_receipt_sha256","v3_schema_receipt_path")):
        if _hash(root/p[pathkey]) != p[key]: raise v3.GateError(f"changed binding: {key}")
    if c["response_access_at_freeze"]["strain_values_read"] != 0: raise v3.GateError("not response blind")

def seal_intent(root=None):
    root=(root or _root()).resolve(); c=load_config(root); validate(root,c)
    out={"schema_version":"invariant-response-blind-correction-intent-1.0","analysis_id":c["analysis_id"],"sealed_utc":datetime.now(timezone.utc).isoformat(),
         "config_sha256":_hash(root/CONFIG),"module_sha256":_hash(root/MODULE),"test_sha256":_hash(root/TEST),
         "v3_source_receipt_sha256":c["predecessor"]["v3_source_receipt_sha256"],"v3_schema_receipt_sha256":c["predecessor"]["v3_schema_receipt_sha256"],
         "strain_values_read_before_intent":0,"real_scores_before_intent":0,"post_response_tuning":0}
    v3._write_once(root/c["package"]["intent_receipt"],out); return out

def _correlation_scores(d,q,band_indices,n,offsets):
    full=np.zeros(n,dtype=complex); full[band_indices]=d*np.conj(q)
    corr=np.fft.ifft(full)*n; den=float(np.vdot(q,q).real)
    return np.abs(corr[np.mod(offsets,n)])**2/den

def _maximize_pair(h,l):
    best=(-np.inf,0,0)
    for i in range(len(h)):
        vals=h[i]+l[i:i+83]; j=int(np.argmax(vals))
        if vals[j]>best[0]: best=(float(vals[j]),i,j)
    return best

def validate_injections(root=None):
    root=(root or _root()).resolve(); c=load_config(root); validate(root,c); base=v3.load_config(root)
    intent=root/c["package"]["intent_receipt"]
    if not intent.is_file(): raise v3.GateError("v4 intent missing")
    templates=v3.build_templates(root,base); n=32768; sr=4096; freqs=np.fft.rfftfreq(n,1/sr); mask=(freqs>=20)&(freqs<=512); bins=np.flatnonzero(mask)
    q={k:templates[k][mask] for k in v3.KERNEL_IDS}; q={k:x/math.sqrt(float(np.vdot(x,x).real)) for k,x in q.items()}
    tg=np.rint(v3.declared_grid(base["analysis"]["time_grid_seconds"])*sr).astype(int)
    lg=np.arange(-246,247,dtype=int)
    fam={kid:name for name,ids in base["equivalence_families"].items() for kid in ids}; rows=[]
    for ki,kid in enumerate(v3.KERNEL_IDS):
        for snr in base["injections"]["network_snr"]:
            for j in range(128):
                rng=np.random.default_rng(base["injections"]["seed_start"]+100000*ki+1000*snr+j); amp=snr/math.sqrt(2)
                data=[]
                for _ in range(2): data.append(amp*q[kid]+(rng.normal(size=len(bins))+1j*rng.normal(size=len(bins)))/math.sqrt(2))
                scores={}
                for k in v3.KERNEL_IDS:
                    hs=_correlation_scores(data[0],q[k],bins,n,tg); ls=_correlation_scores(data[1],q[k],bins,n,lg)
                    scores[k]=_maximize_pair(hs,ls)[0]
                top=max(scores.values()); winners=[k for k,s in scores.items() if top-s<=1e-9]; good=any(fam[w]==fam[kid] for w in winners)
                rows.append({"injected":kid,"snr":snr,"realization":j,"winning_templates":winners,"correct_equivalence_family":good,"invalid":False})
    summary={}
    for kid in v3.KERNEL_IDS:
        for snr in base["injections"]["network_snr"]:
            z=[r for r in rows if r["injected"]==kid and r["snr"]==snr]; summary[f"{kid}@{snr}"]=sum(r["correct_equivalence_family"] for r in z)/len(z)
    passed=all(summary[f"{k}@20"]>=.9 for k in v3.KERNEL_IDS)
    out={"schema_version":"invariant-exact-grid-injection-recovery-1.0","analysis_id":c["analysis_id"],"intent_sha256":_hash(intent),"cells":len(rows),
         "candidate_fits":len(rows)*7,"common_time_grid_points":411,"detector_delay_grid_points":83,"rows":rows,"family_recovery_fraction":summary,
         "noncausal_early_response_detections":0,"pass_rule_met":passed,"strain_values_read":0}
    v3._write_once(root/c["package"]["injection_receipt"],out); return out

def _psd(x,sr):
    f,p=signal.welch(signal.detrend(x-np.mean(x)),fs=sr,window="hann",nperseg=4*sr,noverlap=2*sr,average="median")
    return f,p

def _one_score(d,q,psd,df,t_offsets,freqs):
    w=np.where(np.isfinite(psd)&(psd>0),1/psd,0.0); den=float(np.sum(w*np.abs(q)**2)); phases=np.exp(2j*np.pi*np.outer(t_offsets,freqs))
    corr=phases@(w*d*np.conj(q)); return 4*df*np.abs(corr)**2/den

def score(root=None):
    root=(root or _root()).resolve(); c=load_config(root); validate(root,c); base=v3.load_config(root); inj=root/c["package"]["injection_receipt"]
    if not inj.is_file() or not json.loads(inj.read_text())["pass_rule_met"]: raise v3.GateError("exact injection gate failed")
    sr=4096; n=32768; dt=1/sr; df=.125; freqs=np.fft.rfftfreq(n,dt); mask=(freqs>=20)&(freqs<=512); fb=freqs[mask]
    tg=v3.declared_grid(base["analysis"]["time_grid_seconds"]); dg=v3.declared_grid(base["analysis"]["delta_LH_grid_seconds"]); lg=np.arange(tg[0]+dg[0],tg[-1]+dg[-1]+.5*dt,dt)
    templates=v3.build_templates(root,base); data={}; psd={}
    for p in base["sources"]["products"]:
        with h5py.File(root/p["path"],"r") as h: strain=np.asarray(h[base["sources"]["strain_dataset"]][...],float); dq=np.asarray(h[base["sources"]["dq_dataset"]][...],int)
        if not np.all((dq[12:20]&1)==1): raise v3.GateError("DQ gate changed")
        a=signal.detrend(strain[12*sr:20*sr]-np.mean(strain[12*sr:20*sr])); data[p["detector"]]=np.fft.rfft(a*signal.windows.tukey(n,.125))*dt
        sides=[]
        for x in (strain[:12*sr],strain[20*sr:]):
            ff,pp=_psd(x,sr); sides.append(np.interp(freqs,ff,pp))
        psd[p["detector"]]={"pre":sides[0],"post":sides[1],"mean":.5*(sides[0]+sides[1])}
    rows=[]; cal=c["calibration_diagnostic"]
    variants={"nominal":np.ones_like(fb,dtype=complex)}
    for s in (-.05,.05): variants[f"cal_slope_{s:+.2f}"]=np.exp((s+1j*s)*np.log(fb/100))
    for pc in ("mean","pre","post"):
        for model,q0 in templates.items():
            for cv,mult in variants.items():
                q=q0[mask]*dt*mult; hs=_one_score(data["H1"][mask],q,psd["H1"][pc][mask],df,tg,fb); ls=_one_score(data["L1"][mask],q,psd["L1"][pc][mask],df,lg,fb)
                best=_maximize_pair(hs,ls); rows.append({"psd":pc,"calibration_variant":cv,"model":model,"delta_2logL_vs_null":best[0],
                    "t0_seconds":float(tg[best[1]]),"delta_LH_seconds":float(dg[best[2]]),"H1_at_network_best":float(hs[best[1]]),"L1_at_network_best":float(ls[best[1]+best[2]]),
                    "H1_independent_max":float(np.max(hs)),"L1_independent_max":float(np.max(ls)),"invalid":False})
    nominal=[r for r in rows if r["psd"]=="mean" and r["calibration_variant"]=="nominal"]; gr=next(r for r in nominal if r["model"]=="K00_INSTANTANEOUS")
    for r in rows: r["delta_2logL_vs_nominal_GR_control"]=r["delta_2logL_vs_null"]-gr["delta_2logL_vs_null"]
    best=max(nominal,key=lambda x:x["delta_2logL_vs_null"]); survives=best["model"] not in {"K00_INSTANTANEOUS","K01_RETARDED","K02_EXPONENTIAL","K03_BIEXPONENTIAL","K06_STOCHASTIC_OU","C04_SOURCE_RINGDOWN"}
    out={"schema_version":"invariant-gw150914-exact-grid-development-result-1.0","analysis_id":c["analysis_id"],"injection_receipt_sha256":_hash(inj),"rows":rows,
         "nominal_GR_fixed_NR_control":gr,"best_nominal_mean_psd":best,"new_kernel_survives_obvious_same_data_countermodels":survives,
         "C05_OU_noise":"NOT_IDENTIFIABLE_FROM_DETERMINISTIC_MEAN_K06_EQUALS_K02","strain_values_read":262144,"dq_values_read_during_score":64,"response_files_opened":2,
         "claim":"DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_MEMORY_EVIDENCE","calibration_gate":"EMPIRICAL_CLAIM_BLOCKED_EXACT_PUBLISHED_FREQUENCY_DEPENDENT_UNCERTAINTY_ARTIFACTS_ABSENT",
         "GR_control_limit":"fixed normalized SXS waveform with time and complex amplitude only; not masses/spins/waveform-systematics marginalization",
         "trials":"8 waveform models x 3 PSD choices x 3 calibration variants retained; no sigma conversion"}
    v3._write_once(root/c["package"]["result_receipt"],out); return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["seal-intent","validate-injections","score"]); a=p.parse_args(); fn={"seal-intent":seal_intent,"validate-injections":validate_injections,"score":score}[a.command]; print(json.dumps(fn(),indent=2,sort_keys=True))
if __name__=="__main__": main()
