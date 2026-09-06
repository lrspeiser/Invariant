"""Fixed subtraction of a previously defined observational warning mask."""
import numpy as np


def exclude_support(train, test, expanded, east, north, config):
    train, test, expanded = [np.asarray(a, bool) for a in (train, test, expanded)]
    east, north = np.asarray(east), np.asarray(north)
    if any(a.shape != train.shape for a in [test, expanded, east, north]) or np.any(train & test):
        raise ValueError('invalid spatial masks')
    new_train, new_test = train & ~expanded, test & ~expanded
    quadrants = (east[new_test]>=0).astype(int)+2*(north[new_test]>=0).astype(int)
    counts = np.bincount(quadrants, minlength=4)
    gates = dict(calibration_pixels=int(new_train.sum())>=config['minimum_calibration_pixels'],
        validation_pixels=int(new_test.sum())>=config['minimum_validation_pixels'],
        validation_quadrants=bool(np.all(counts>=config['diagnostic_gates']['minimum_quadrant_pixels'])))
    return new_train, new_test, dict(calibration_pixels=int(new_train.sum()), validation_pixels=int(new_test.sum()),
        removed_calibration_pixels=int((train & expanded).sum()), removed_validation_pixels=int((test & expanded).sum()),
        validation_quadrant_counts=counts.tolist(), support_gates=gates, sufficient_support=all(gates.values()))
