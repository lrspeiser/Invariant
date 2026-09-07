"""Manufactured controls only; actual measurement arrays are never accessed."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr, erfc
from mond_atlas_aperture_scorer import training_fit, freeze_predictions, evaluate

R = Path(__file__).resolve().parents[1]
P = R/'work/gravity-first-principles/mond-atlas-aperture-scorer-001'
edges = np.linspace(-105, 105, 43)
truth = np.array([7.2, 8.6, 1.1])
centers = np.linspace(-45, 45, 15)
flux = np.linspace(12, 24, 15)


def model(indices, parameters):
    index = np.array(indices)
    offset = (edges[None, :]-centers[index, None]-parameters[0])/parameters[1]
    return parameters[2]*flux[index, None]*np.diff(ndtr(offset), axis=1)


def run():
    out = P/'run001'; out.mkdir(exist_ok=False)
    mean = .025*np.sin(np.arange(42))
    cov = .03*np.exp(-abs(np.subtract.outer(np.arange(42), np.arange(42)))/3)+.01*np.eye(42)
    # Independent erfc expression for the manufactured observations.
    offset = (edges[None, :]-centers[:, None]-truth[0])/(truth[1]*np.sqrt(2))
    spectra = -.5*truth[2]*flux[:, None]*np.diff(erfc(offset), axis=1)
    data = spectra+mean
    train, held = tuple(range(10)), tuple(range(10, 15))
    fit = training_fit(model, train, data[:10], mean, cov)
    (out/'training-fit.json').write_text(json.dumps(fit, indent=2)+'\n')
    frozen = freeze_predictions(model, fit, held)
    (out/'predictions-before-evaluation.json').write_text(json.dumps(dict(indices=held, values=frozen.values.tolist()), indent=2)+'\n')
    parameter_error = np.max(abs(np.array(fit['parameters'])-truth))
    prediction_error = np.linalg.norm(frozen.values-spectra[10:])/np.linalg.norm(spectra[10:])
    first = evaluate(frozen, data[10:], mean, cov)
    original_bytes = frozen.values.tobytes()
    disturbance = .02*np.cos(np.arange(210).reshape(5, 42))
    changed = evaluate(frozen, data[10:]+disturbance, mean, cov)
    residual = np.asarray(changed['residuals'])
    direct = np.einsum('ic,cd,id->i', residual, np.linalg.inv(cov), residual)/42
    score_error = np.max(abs(direct-changed['per_aperture_q_per_channel']))
    deficient = training_fit(lambda indices,p:model(indices,[p[0],8.6,p[2]]), train, data[:10], mean, cov)
    invalid_rejected = False
    try:
        evaluate(frozen, np.full((5,42),np.nan), mean, cov)
    except ValueError:
        invalid_rejected = True
    checks = dict(parameter_error_lt_1e_5=bool(parameter_error<1e-5),
        held_prediction_relative_lt_1e_6=bool(prediction_error<1e-6),
        dense_score_error_lt_1e_11=bool(score_error<1e-11),
        rank_deficient_rejected=deficient['evaluation_allowed'] is False,
        frozen_predictions_unchanged=original_bytes==frozen.values.tobytes(),
        nonfinite_rejected=invalid_rejected,
        altered_evaluation_changes_score=changed['mean_q_per_channel']>first['mean_q_per_channel'],
        western_mean_accounted=first['mean_q_per_channel']<1e-12)
    result = dict(status='MANUFACTURED_SCORER_CONTROLS_ONLY', all_pass=all(checks.values()), checks=checks,
        max_parameter_error=float(parameter_error), held_relative_error=float(prediction_error),
        dense_score_max_error=float(score_error), rank_deficient_fit=deficient,
        source_region_reads=0, observed_gravity_scores=0,
        bindings={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [Path(__file__),R/'scripts/mond_atlas_aperture_scorer.py',P/'PREFLIGHT.md']})
    (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(all_pass=result['all_pass'], checks=checks, parameter_error=float(parameter_error), prediction_error=float(prediction_error), score_error=float(score_error))))
    assert result['all_pass']


if __name__=='__main__':run()
