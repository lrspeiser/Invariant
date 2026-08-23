import Std.Tactic

namespace Invariant

/-- The discovered candidate `(1/2)*x0*x1^2` is deliberately changed to
    `(1/2)*x0*x1^2 + 1`. The sealed holdout input `x0 = x1 = 0` exposes residual `1`. -/
theorem externalOpenStaxKineticUnitOffsetFalse :
    (1 / 2 : Rat) * 0 * 0 ^ 2 + 1 = 0 := by
  rfl

end Invariant
