from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    """Powers of length and time in the non-relativistic static sector."""

    length: int = 0
    time: int = 0

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.length + other.length, self.time + other.time)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.length - other.length, self.time - other.time)

    def __pow__(self, power: int) -> "Dimension":
        return Dimension(self.length * power, self.time * power)

    @property
    def is_dimensionless(self) -> bool:
        return self == Dimension()

    def as_dict(self) -> dict[str, int]:
        return {"L": self.length, "T": self.time}


DIMENSIONLESS = Dimension()
LENGTH = Dimension(length=1)
ACCELERATION = Dimension(length=1, time=-2)


def normalized_invariant_dimensions() -> dict[str, Dimension]:
    """Dimensions of the frozen v18 MVP invariants.

    x = D^2/a_sigma^2
    q = L_sigma^2 (partial D)^2/a_sigma^2
    z = Z_b^2/Z_0^2
    """

    displacement = ACCELERATION
    displacement_gradient = displacement / LENGTH
    state = Dimension(time=-2)
    return {
        "x": displacement**2 / ACCELERATION**2,
        "q": LENGTH**2 * displacement_gradient**2 / ACCELERATION**2,
        "z": state**2 / state**2,
    }


def assert_dimensionless_invariants(dimensions: dict[str, Dimension]) -> None:
    bad = {name: dim.as_dict() for name, dim in dimensions.items() if not dim.is_dimensionless}
    if bad:
        raise ValueError(f"Grammar contains dimensionful normalized invariants: {bad}")

