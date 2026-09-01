"""Exact fractional-origin, bounded nuisance-grid successor; frozen before strain access."""
from __future__ import annotations
import argparse, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import h5py
import numpy as np
from scipy import signal
from . import open_gravity_dynamic_source_memory_gw150914_response_v3 as v3

CONFIG=Path("configs/open_gravity_dynamic_source_memory_gw150914_response_v5.json"); MODULE=Path("src/sigma_theory_compiler/open_gravity_dynamic_source_memory_gw150914_response_v5.py"); TEST=Path("tests/test_open_gravity_dynamic_source_memory_gw150914_response_v5.py")
def _root(): return Path(__file__).resolve().parents[2]
def _hash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_config(root=None): return json.loads(((root or _root())/CONFIG).read_text())
def validate(root,c):
    b=c["bindings"]
    for stem in ("v4_config","v4_module","v4_intent","v4_injection","v3_source_receipt","v3_schema_receipt"):
        if _hash(root/b[f"{stem}_path"])!=b[f"{stem}_sha256"]: raise v3.GateError(f"changed binding {stem}")
    if c["response_access_at_freeze"]["strain_values_read"]!=0: raise v3.GateError("response already read")
def grid(r):
    lo=float(r["min"]); hi=float(r["max"]); step=float(r["step"]); n=int(math.floor((hi-lo)/step+1e-12))+1
    x=lo+np.arange(n)*step
    if np.any(x>hi+1e-15): raise v3.GateError("grid above maximum")
    return x
def seal_intent(root=None):
    root=(root or _root()).resolve(); c=load_config(root); validate(root,c)
    out={"schema_version":"invariant-fractional-grid-response-blind-intent-1.0","analysis_id":c["analysis_id"],"sealed_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":_hash(root/CONFIG),"module_sha256":_hash(root/MODULE),"test_sha256":_hash(root/TEST),"source_receipt_sha256":c["bindings"]["v3_source_receipt_sha256"],"schema_receipt_sha256":c["bindings"]["v3_schema_receipt_sha256"],"strain_values_read_before_intent":0,"post_response_tuning":0}
    v3._write_once(root/c["package"]["intent_receipt"],out); return out
def corr(d,q,bins,n,freqs,start,count,weight=None,scale=1.0):
    w=np.ones_like(freqs) if weight is None else weight; den=float(np.sum(w*np.abs(q)**2)); full=np.zeros(n,dtype=complex)
    full[bins]=w*d*np.conj(q)*np.exp(2j*np.pi*freqs*start); z=np.fft.ifft(full)*n
    return scale*np.abs(z[:count])**2/den
def maxpair(h,l,nd):
    best=(-np.inf,0,0)
    for i in range(len(h)):
        z=h[i]+l[i:i+nd]; j=int(np.argmax(z))
        if z[j]>best[0]: best=(float(z[j]),i,j)
    return best
def validate_injections(root=None):
    root=(root or _root()).resolve(); c=load_config(root); validate(root,c); base=v3.load_config(root); intent=root/c["package"]["intent_receipt"]
    if not intent.is_file(): raise v3.GateError("intent missing")
    n=32768; sr=4096; f=np.fft.rfftfreq(n,1/sr); mask=(f>=20)&(f<=512); bins=np.flatnonzero(mask); fb=f[mask]
    templates=v3.build_templates(root,base); q={k:templates[k][mask] for k in v3.KERNEL_IDS}; q={k:x/math.sqrt(float(np.vdot(x,x).real)) for k,x in q.items()}
    tg=grid(base["analysis"]["time_grid_seconds"]); dg=grid(base["analysis"]["delta_LH_grid_seconds"]); nl=len(tg)+len(dg)-1
    raw=np.loadtxt(root/base["sources"]["nr_source_path"]); rel=np.arange(n)/sr-(base["sources"]["event_gps"]-base["analysis"]["analysis_interval_gps"][0]); source=np.interp(rel,raw[:,0],raw[:,1],left=0,right=0); first=int(np.flatnonzero(source)[0]); causal={k:float(np.max(np.abs(np.fft.irfft(templates[k],n)[:first]))) for k in v3.KERNEL_IDS}; early=sum(x>1e-12 for x in causal.values())
    fam={kid:name for name,ids in base["equivalence_families"].items() for kid in ids}; rows=[]
    for ki,kid in enumerate(v3.KERNEL_IDS):
        for snr in base["injections"]["network_snr"]:
            for j in range(128):
                rng=np.random.default_rng(base["injections"]["seed_start"]+100000*ki+1000*snr+j); amp=snr/math.sqrt(2); data=[amp*q[kid]+(rng.normal(size=len(fb))+1j*rng.normal(size=len(fb)))/math.sqrt(2) for _ in range(2)]; scores={}
                for k in v3.KERNEL_IDS:
                    hs=corr(data[0],q[k],bins,n,fb,tg[0],len(tg)); ls=corr(data[1],q[k],bins,n,fb,tg[0]+dg[0],nl); scores[k]=maxpair(hs,ls,len(dg))[0]
                top=max(scores.values()); winners=[k for k,s in scores.items() if top-s<=1e-9]; good=any(fam[w]==fam[kid] for w in winners); rows.append({"injected":kid,"snr":snr,"realization":j,"winning_templates":winners,"correct_equivalence_family":good,"invalid":False})
    summary={f"{k}@{s}":sum(r["correct_equivalence_family"] for r in rows if r["injected"]==k and r["snr"]==s)/128 for k in v3.KERNEL_IDS for s in base["injections"]["network_snr"]}; passed=early==0 and all(summary[f"{k}@20"]>=.9 for k in v3.KERNEL_IDS)
    out={"schema_version":"invariant-fractional-grid-injection-recovery-1.0","analysis_id":c["analysis_id"],"intent_sha256":_hash(intent),"cells":len(rows),"candidate_fits":len(rows)*7,"common_grid_points":len(tg),"delay_grid_points":len(dg),"grid_endpoints":{"t0":[float(tg[0]),float(tg[-1])],"delta_LH":[float(dg[0]),float(dg[-1])]},"causal_pre_source_max_abs":causal,"noncausal_early_response_detections":early,"rows":rows,"family_recovery_fraction":summary,"pass_rule_met":passed,"strain_values_read":0}
    v3._write_once(root/c["package"]["injection_receipt"],out); return out
def _psd(x,sr):
    return signal.welch(signal.detrend(x-np.mean(x)),fs=sr,window="hann",nperseg=4*sr,noverlap=2*sr,average="median")
def score(root=None):
    root=(root or _root()).resolve(); c=load_config(root); validate(root,c); base=v3.load_config(root); inj=root/c["package"]["injection_receipt"]
    if not inj.is_file() or not json.loads(inj.read_text())["pass_rule_met"]: raise v3.GateError("fractional exact injection gate failed")
    sr=4096; n=32768; dt=1/sr; df=.125; f=np.fft.rfftfreq(n,dt); mask=(f>=20)&(f<=512); bins=np.flatnonzero(mask); fb=f[mask]; tg=grid(base["analysis"]["time_grid_seconds"]); dg=grid(base["analysis"]["delta_LH_grid_seconds"]); nl=len(tg)+len(dg)-1
    templates=v3.build_templates(root,base); data={}; psd={}
    for p in base["sources"]["products"]:
        with h5py.File(root/p["path"],"r") as h: strain=np.asarray(h[base["sources"]["strain_dataset"]][...],float); dq=np.asarray(h[base["sources"]["dq_dataset"]][...],int)
        if not np.all((dq[12:20]&1)==1): raise v3.GateError("DQ changed")
        x=signal.detrend(strain[12*sr:20*sr]-np.mean(strain[12*sr:20*sr])); data[p["detector"]]=np.fft.rfft(x*signal.windows.tukey(n,.125))*dt; pp=[]
        for side in (strain[:12*sr],strain[20*sr:]): ff,power=_psd(side,sr); pp.append(np.interp(f,ff,power))
        psd[p["detector"]]={"pre":pp[0],"post":pp[1],"mean":.5*(pp[0]+pp[1])}
    variants={"nominal":np.ones_like(fb,dtype=complex)}
    for s in (-.05,.05): variants[f"cal_slope_{s:+.2f}"]=np.exp((s+1j*s)*np.log(fb/100))
    rows=[]
    for pc in ("mean","pre","post"):
        for model,q0 in templates.items():
            for cv,mult in variants.items():
                q=q0[mask]*dt*mult; ss={}
                for det,start,count in (("H1",tg[0],len(tg)),("L1",tg[0]+dg[0],nl)):
                    P=psd[det][pc][mask]; valid=np.isfinite(P)&(P>0); w=np.where(valid,1/P,0); ss[det]=corr(data[det][mask],q,bins,n,fb,start,count,w,4*df)
                best=maxpair(ss["H1"],ss["L1"],len(dg)); rows.append({"psd":pc,"calibration_variant":cv,"model":model,"delta_2logL_vs_null":best[0],"t0_seconds":float(tg[best[1]]),"delta_LH_seconds":float(dg[best[2]]),"H1_at_network_best":float(ss["H1"][best[1]]),"L1_at_network_best":float(ss["L1"][best[1]+best[2]]),"H1_independent_max":float(np.max(ss["H1"])),"L1_independent_max":float(np.max(ss["L1"])),"invalid":False})
    nominal=[r for r in rows if r["psd"]=="mean" and r["calibration_variant"]=="nominal"]; gr=next(r for r in nominal if r["model"]=="K00_INSTANTANEOUS");
    for r in rows: r["delta_2logL_vs_nominal_GR_control"]=r["delta_2logL_vs_null"]-gr["delta_2logL_vs_null"]
    best=max(nominal,key=lambda r:r["delta_2logL_vs_null"]); ordinary={"K00_INSTANTANEOUS","K01_RETARDED","K02_EXPONENTIAL","K03_BIEXPONENTIAL","K06_STOCHASTIC_OU","C04_SOURCE_RINGDOWN"}
    out={"schema_version":"invariant-gw150914-fractional-grid-development-result-1.0","analysis_id":c["analysis_id"],"injection_receipt_sha256":_hash(inj),"rows":rows,"nominal_GR_fixed_NR_control":gr,"best_nominal_mean_psd":best,"new_kernel_survives_obvious_same_data_countermodels":best["model"] not in ordinary,"C05_OU_noise":"UNIDENTIFIABLE_FROM_DETERMINISTIC_MEAN_K06_EQUALS_K02","strain_values_read":262144,"dq_values_read_during_score":64,"response_files_opened":2,"claim":"DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_MEMORY_EVIDENCE","calibration_gate":"BLOCKED_EXACT_PUBLISHED_FREQUENCY_DEPENDENT_UNCERTAINTY_ARTIFACTS_ABSENT","GR_control_limit":"fixed normalized SXS waveform, time and complex amplitude only; no mass/spin/waveform-systematics marginalization","trials":"8 waveform models x 3 PSD choices x 3 calibration variants; all rows retained; no sigma conversion"}
    v3._write_once(root/c["package"]["result_receipt"],out); return out
def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["seal-intent","validate-injections","score"]); a=p.parse_args(); print(json.dumps({"seal-intent":seal_intent,"validate-injections":validate_injections,"score":score}[a.command](),indent=2,sort_keys=True))
if __name__=="__main__": main()
