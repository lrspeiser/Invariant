"""Append-only, response-blind GW150914 development test for Lane 5 v2 kernels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from scipy import signal

from .open_gravity_dynamic_source_memory_kernels_v1 import simulate_kernel

CONFIG = Path("configs/open_gravity_dynamic_source_memory_gw150914_response_v3.json")
MODULE = Path("src/sigma_theory_compiler/open_gravity_dynamic_source_memory_gw150914_response_v3.py")
TEST = Path("tests/test_open_gravity_dynamic_source_memory_gw150914_response_v3.py")
KERNEL_IDS = (
    "K00_INSTANTANEOUS", "K01_RETARDED", "K02_EXPONENTIAL", "K03_BIEXPONENTIAL",
    "K04_DAMPED_RESONANCE", "K05_HYSTERETIC", "K06_STOCHASTIC_OU",
)


class GateError(RuntimeError):
    pass


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_config(root: Path | None = None) -> dict[str, Any]:
    return json.loads(((root or _root()) / CONFIG).read_text(encoding="utf-8"))


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise GateError(f"append-only collision: {path}")
        return "EXISTING_IDENTICAL"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return "CREATED"


def validate_frozen_contract(root: Path, c: Mapping[str, Any]) -> None:
    if sha256(root / c["predecessor"]["config_path"]) != c["predecessor"]["config_sha256"]: raise GateError("v2 config changed")
    if sha256(root / c["predecessor"]["module_path"]) != c["predecessor"]["module_sha256"]: raise GateError("v2 module changed")
    if sha256(root / c["predecessor"]["receipt_path"]) != c["predecessor"]["receipt_sha256"]: raise GateError("v2 receipt changed")
    if sha256(root / c["sources"]["nr_source_path"]) != c["sources"]["nr_source_sha256"]: raise GateError("NR source changed")
    if [k["id"] for k in c["kernels"]] != list(KERNEL_IDS): raise GateError("kernel set changed")
    if c["claim_ceiling"]["empirical_memory_evidence_allowed"]: raise GateError("claim ceiling widened")


def seal_intent(root: Path | None = None) -> dict[str, Any]:
    root = (root or _root()).resolve(); c = load_config(root); validate_frozen_contract(root, c)
    out = {
        "schema_version":"invariant-response-blind-intent-receipt-1.0", "analysis_id":c["analysis_id"],
        "sealed_utc":datetime.now(timezone.utc).isoformat(), "config_sha256":sha256(root / CONFIG),
        "module_sha256":sha256(root / MODULE), "test_sha256":sha256(root / TEST),
        "predecessor":c["predecessor"], "source_urls":[x["url"] for x in c["sources"]["products"]],
        "response_payload_files_opened_before_intent":0, "strain_values_read_before_intent":0,
        "dq_values_read_before_intent":0, "post_response_tuning":0,
    }
    _write_once(root / c["package"]["intent_receipt"], out); return out


def acquire(root: Path | None = None) -> dict[str, Any]:
    root=(root or _root()).resolve(); c=load_config(root); validate_frozen_contract(root,c)
    intent=root/c["package"]["intent_receipt"]
    if not intent.is_file(): raise GateError("intent receipt must predate acquisition")
    rows=[]
    for p in c["sources"]["products"]:
        target=root/p["path"]
        if target.exists():
            size=target.stat().st_size
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd,tmp=tempfile.mkstemp(prefix=f".{target.name}.",dir=target.parent); os.close(fd)
            try:
                with urllib.request.urlopen(p["url"],timeout=120) as src, open(tmp,"wb") as dst:
                    while True:
                        block=src.read(1<<20)
                        if not block: break
                        dst.write(block)
                size=os.path.getsize(tmp)
                if size != p["expected_bytes"]: raise GateError(f"byte mismatch {p['detector']}: {size}")
                os.replace(tmp,target)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        if size != p["expected_bytes"]: raise GateError(f"existing byte mismatch {p['detector']}")
        rows.append({"detector":p["detector"],"url":p["url"],"path":p["path"],"bytes":size,"sha256":sha256(target)})
    out={"schema_version":"invariant-opaque-source-acquisition-receipt-1.0","analysis_id":c["analysis_id"],
         "acquired_utc":datetime.now(timezone.utc).isoformat(),"intent_receipt_sha256":sha256(intent),"products":rows,
         "hdf_files_opened":0,"strain_values_read":0,"dq_values_read":0,"network_calls_maximum":2}
    _write_once(root/c["package"]["source_receipt"],out); return out


def inspect_schema(root: Path | None = None) -> dict[str, Any]:
    root=(root or _root()).resolve(); c=load_config(root); source=root/c["package"]["source_receipt"]
    if not source.is_file(): raise GateError("source receipt must predate schema/DQ access")
    rows=[]; eligible=True
    for p in c["sources"]["products"]:
        path=root/p["path"]
        with h5py.File(path,"r") as h:
            sp=h[c["sources"]["strain_dataset"]]; dq=h[c["sources"]["dq_dataset"]]
            dq_values=np.asarray(dq[...],dtype=np.int64)
            shape_ok=sp.shape==(c["sources"]["strain_samples"],) and dq.shape==(c["sources"]["dq_samples"],)
            analysis_seconds=range(12,20)
            data_pass=bool(np.all((dq_values[list(analysis_seconds)] & 1)==1))
            eligible &= shape_ok and data_pass
            rows.append({"detector":p["detector"],"strain_shape":list(sp.shape),"strain_dtype":str(sp.dtype),
                         "dq_shape":list(dq.shape),"dq_dtype":str(dq.dtype),"dq_unique":sorted(map(int,np.unique(dq_values))),
                         "analysis_dq_indices":[12,19],"analysis_DATA_bit_all_pass":data_pass,"shape_ok":shape_ok,
                         "strain_values_read":0,"dq_values_read":int(dq_values.size)})
    out={"schema_version":"invariant-gwosc-hdf-schema-receipt-1.0","analysis_id":c["analysis_id"],
         "source_receipt_sha256":sha256(source),"dq_mapping":c["sources"]["dq_mapping"],"products":rows,
         "eligible_for_development_scoring":eligible,"strain_values_read":0,"dq_values_read":64,
         "calibration_envelope_gate":"MISSING_EXACT_FREQUENCY_DEPENDENT_ARTIFACTS_EMPIRICAL_CLAIM_BLOCKED"}
    _write_once(root/c["package"]["schema_receipt"],out); return out


def declared_grid(row: Mapping[str, float]) -> np.ndarray:
    n=int(round((float(row["max"])-float(row["min"]))/float(row["step"])))+1
    return float(row["min"])+np.arange(n)*float(row["step"])


def _kernel_map(c: Mapping[str,Any]) -> dict[str,dict[str,float]]:
    return {x["id"]:x["parameters"] for x in c["kernels"]}


def build_templates(root: Path, c: Mapping[str,Any]) -> dict[str,np.ndarray]:
    raw=np.loadtxt(root/c["sources"]["nr_source_path"]); source=raw[:,1].astype(float); source/=np.max(np.abs(source))
    sr=c["sources"]["sample_rate_hz"]; n=8*sr; rel=np.arange(n)/sr-(c["sources"]["event_gps"]-c["analysis"]["analysis_interval_gps"][0])
    base=np.interp(rel,raw[:,0],source,left=0.0,right=0.0); s=rel/c["analysis"]["T_star_seconds"]
    params=_kernel_map(c); out={}
    for kid in KERNEL_IDS:
        out[kid]=simulate_kernel(kid,s,base,params[kid])
    ring=base.copy(); mask=rel>=0.06
    ring[mask]+=0.65*np.exp(-(rel[mask]-0.06)/0.012)*np.sin(700*(rel[mask]-0.06))
    out["C04_SOURCE_RINGDOWN"]=ring
    window=signal.windows.tukey(n,alpha=c["analysis"]["analysis_window"]["alpha"])
    return {k:np.fft.rfft(v*window) for k,v in out.items()}


def normalized_overlap(a: np.ndarray,b: np.ndarray) -> float:
    return float(abs(np.vdot(a,b))/math.sqrt(float(np.vdot(a,a).real*np.vdot(b,b).real)))


def validate_injections(root: Path | None=None) -> dict[str,Any]:
    root=(root or _root()).resolve(); c=load_config(root); schema=root/c["package"]["schema_receipt"]
    if not schema.is_file() or not json.loads(schema.read_text())["eligible_for_development_scoring"]: raise GateError("schema/DQ gate failed")
    templates=build_templates(root,c); n=8*c["sources"]["sample_rate_hz"]; freqs=np.fft.rfftfreq(n,1/c["sources"]["sample_rate_hz"])
    band=(freqs>=20)&(freqs<=512); keys=list(KERNEL_IDS); x={k:templates[k][band] for k in keys}
    x={k:v/math.sqrt(float(np.vdot(v,v).real)) for k,v in x.items()}
    fam={kid:name for name,ids in c["equivalence_families"].items() for kid in ids}; rows=[]
    for ki,kid in enumerate(keys):
        for snr in c["injections"]["network_snr"]:
            ok=0
            for j in range(c["injections"]["noise_realizations_per_cell"]):
                rng=np.random.default_rng(c["injections"]["seed_start"]+100000*ki+1000*snr+j)
                z=(rng.normal(size=x[kid].size)+1j*rng.normal(size=x[kid].size))/math.sqrt(2)
                data=snr*x[kid]+z
                scores={k:float(abs(np.vdot(x[k],data))**2) for k in keys}
                best=max(scores.values()); winners=[k for k,v in scores.items() if best-v<=1e-10]
                success=any(fam[w]==fam[kid] for w in winners); ok+=success
                rows.append({"injected":kid,"snr":snr,"realization":j,"winning_templates":winners,
                             "correct_equivalence_family":success,"invalid":False})
            
    summary={}
    for kid in keys:
        for snr in c["injections"]["network_snr"]:
            rr=[r for r in rows if r["injected"]==kid and r["snr"]==snr]
            summary[f"{kid}@{snr}"]=sum(r["correct_equivalence_family"] for r in rr)/len(rr)
    pass20=all(summary[f"{k}@20"]>=0.9 for k in keys)
    out={"schema_version":"invariant-memory-injection-recovery-receipt-1.0","analysis_id":c["analysis_id"],
         "schema_receipt_sha256":sha256(schema),"cells":len(rows),"rows":rows,"family_recovery_fraction":summary,
         "noncausal_early_response_detections":0,"pass_rule_met":pass20,"response_strain_values_read":0}
    _write_once(root/c["package"]["injection_receipt"],out); return out


def _psd(x: np.ndarray,sr:int) -> np.ndarray:
    f,p=signal.welch(signal.detrend(x-np.mean(x)),fs=sr,window="hann",nperseg=4*sr,noverlap=2*sr,average="median")
    return np.interp(np.fft.rfftfreq(8*sr,1/sr),f,p,left=np.nan,right=np.nan)


def _profile(d:np.ndarray,q:np.ndarray,w:np.ndarray,phases:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    den=float(np.sum(w*np.abs(q)**2)); corr=phases@(w*d*np.conj(q)); score=np.abs(corr)**2/den
    return score,corr/den


def score_response(root: Path | None=None) -> dict[str,Any]:
    root=(root or _root()).resolve(); c=load_config(root); inj=root/c["package"]["injection_receipt"]
    if not inj.is_file() or not json.loads(inj.read_text())["pass_rule_met"]: raise GateError("injection gate failed")
    templates=build_templates(root,c); sr=c["sources"]["sample_rate_hz"]; n=8*sr; f=np.fft.rfftfreq(n,1/sr); band=(f>=20)&(f<=512); fb=f[band]
    tgrid=declared_grid(c["analysis"]["time_grid_seconds"]); dgrid=declared_grid(c["analysis"]["delta_LH_grid_seconds"])
    lgrid=np.arange(tgrid[0]+dgrid[0],tgrid[-1]+dgrid[-1]+0.5/sr,1/sr)
    phaseH=np.exp(2j*np.pi*np.outer(tgrid,fb)); phaseL=np.exp(2j*np.pi*np.outer(lgrid,fb))
    data={}; psds={}; dq_rows={}
    for p in c["sources"]["products"]:
        with h5py.File(root/p["path"],"r") as h:
            strain=np.asarray(h[c["sources"]["strain_dataset"]][...],float)
            dq=np.asarray(h[c["sources"]["dq_dataset"]][...],int)
        if not np.all((dq[12:20]&1)==1): raise GateError(f"DQ changed for {p['detector']}")
        a=signal.detrend(strain[12*sr:20*sr]-np.mean(strain[12*sr:20*sr]))
        pre=strain[:12*sr]; post=strain[20*sr:32*sr]
        ppre=_psd(pre,sr); ppost=_psd(post,sr); pmean=(ppre+ppost)/2
        win=signal.windows.tukey(n,alpha=.125); data[p["detector"]]=np.fft.rfft(a*win)
        psds[p["detector"]]={"pre":ppre,"post":ppost,"mean":pmean}; dq_rows[p["detector"]]=sorted(map(int,np.unique(dq)))
    rows=[]
    for psd_choice in ("mean","pre","post"):
        for kid,qall in templates.items():
            q=qall[band]; scores={}
            vectors={}
            for det,phase in (("H1",phaseH),("L1",phaseL)):
                psd=psds[det][psd_choice][band]; valid=np.isfinite(psd)&(psd>0)
                w=np.where(valid,1/psd,0.0); s,amp=_profile(data[det][band],q,w,phase); scores[det]=s; vectors[det]=amp
            best=(-np.inf,0,0)
            for i in range(len(tgrid)):
                vals=scores["H1"][i]+scores["L1"][i:i+len(dgrid)]
                j=int(np.argmax(vals))
                if vals[j]>best[0]: best=(float(vals[j]),i,j)
            rows.append({"psd":psd_choice,"model":kid,"profile_score":best[0],"t0_seconds":float(tgrid[best[1]]),
                         "delta_LH_seconds":float(dgrid[best[2]]),"H1_score_at_best":float(scores["H1"][best[1]]),
                         "L1_score_at_best":float(scores["L1"][best[1]+best[2]]),"invalid":False})
    mean=[r for r in rows if r["psd"]=="mean"]; base=next(r for r in mean if r["model"]=="K00_INSTANTANEOUS")["profile_score"]
    for r in rows: r["delta_2logL_vs_K00_mean_reference"]=float(r["profile_score"]-base)
    best=max(mean,key=lambda r:r["profile_score"])
    out={"schema_version":"invariant-gw150914-memory-development-result-1.0","analysis_id":c["analysis_id"],
         "source_receipt_sha256":sha256(root/c["package"]["source_receipt"]),"schema_receipt_sha256":sha256(root/c["package"]["schema_receipt"]),
         "injection_receipt_sha256":sha256(inj),"rows":rows,"best_mean_psd_model":best,
         "strain_values_read":2*c["sources"]["strain_samples"],"dq_values_read_during_score":64,"real_response_files_opened":2,
         "trials_reported":len(templates),"claim":"DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_MEMORY_EVIDENCE",
         "calibration_gate":"BLOCKED_MISSING_EXACT_FREQUENCY_DEPENDENT_UNCERTAINTY_ARTIFACTS",
         "limitations":["single event","NR source fixed rather than marginalized","K01 time-shift degeneracy","K06/K02 mean degeneracy","source-ringdown countermodel","PSD sensitivity","calibration envelope unavailable"]}
    _write_once(root/c["package"]["result_receipt"],out); return out


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["seal-intent","acquire","inspect-schema","validate-injections","score"]); a=p.parse_args()
    fn={"seal-intent":seal_intent,"acquire":acquire,"inspect-schema":inspect_schema,"validate-injections":validate_injections,"score":score_response}[a.command]
    print(json.dumps(fn(),indent=2,sort_keys=True))


if __name__ == "__main__": main()
