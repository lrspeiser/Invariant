import Std.Tactic

namespace Invariant

/-- The discovered candidate `x0*(x0+1)*(2*x0+1)/6` is deliberately changed to
    `x0*(x0+1)*(2*x0+1)/6 + 1`. The sealed holdout input `x0 = 0` exposes residual `1`. -/
theorem externalNistSumSquaresUnitOffsetFalse :
    (0 : Rat) * (0 + 1) * (2 * 0 + 1) / 6 + 1 = 0 := by
  rfl

end Invariant
