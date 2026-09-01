from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from sigma_theory_compiler import open_gravity_void_correlation_development_release_v3 as release


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


rows = release._synthetic_permutation_rows()
y, sigma, luminosity, path, directions, exposure, identifiers = release.v1._primary_arrays(
    rows, "L_void_Mpc"
)
labels = release.v1.executor_v3.distance_strata(luminosity, identifiers, 10)
groups = [
    sorted(
        (index for index, label in enumerate(labels) if label == stratum),
        key=lambda index: int(identifiers[index]),
    )
    for stratum in range(10)
]
assert all(groups) and sum(map(len, groups)) == len(identifiers)

generator = np.random.Generator(np.random.PCG64(902104729))
observed = float(
    release.v1.executor_v3.profile_grid(y, sigma, path, directions, exposure, identifiers)[
        "one_sided_statistic"
    ]
)
statistics: list[float] = []
order_hashes: list[str] = []
for _ in range(10_000):
    orders = release.v1.executor_v3._pcg64_permutation_orders(
        generator, [len(group) for group in groups]
    )
    order_hashes.append(sha256(canonical([list(order) for order in orders])))
    permuted = [float(value) for value in exposure]
    for stratum, indexes in enumerate(groups):
        values = [float(exposure[index]) for index in indexes]
        for target_position, source_position in enumerate(orders[stratum]):
            permuted[indexes[target_position]] = values[source_position]
    statistic = float(
        release.v1.executor_v3.profile_grid(y, sigma, path, directions, permuted, identifiers)[
            "one_sided_statistic"
        ]
    )
    assert math.isfinite(statistic)
    statistics.append(statistic)

tail = sum(value >= observed for value in statistics)
p_value = (1 + tail) / 10_001
order_root = sha256(b"".join(bytes.fromhex(value) for value in order_hashes))
package = release.regenerate_permutations_from_rows(rows, 10_000)

assert package["observed"] == observed
assert package["permutation_statistics"] == statistics
assert package["tail_count"] == tail
assert package["p_value"] == p_value
assert package["order_hashes"] == order_hashes
assert package["order_root_sha256"] == order_root

forged = {
    "observed": observed,
    "permutation_statistics": [1.0] * 10_000,
    "tail_count": sum(1.0 >= observed for _ in range(10_000)),
    "p_value": (1 + sum(1.0 >= observed for _ in range(10_000))) / 10_001,
}
forgery_rejected = False
try:
    release._exact_validate_regenerated(rows, forged, order_hashes, permutations=10_000)
except release.DevelopmentReleaseV3Error as error:
    forgery_rejected = "statistic mismatch" in str(error)
assert forgery_rejected

print(
    json.dumps(
        {
            "numpy_version": np.__version__,
            "bit_generator": type(generator.bit_generator).__name__,
            "seed": 902104729,
            "strata": 10,
            "permutations": 10_000,
            "observed_hex": observed.hex(),
            "tail_count": tail,
            "p_value_hex": p_value.hex(),
            "order_root_sha256": order_root,
            "statistics_sha256": sha256(canonical([value.hex() for value in statistics])),
            "coherent_all_ones_forgery_rejected": forgery_rejected,
        },
        sort_keys=True,
        indent=2,
    )
)
