import Std.Tactic

namespace Invariant

theorem recoveredKineticNormalForm (mass velocity : Rat) :
    ((1 / 2) * mass) * velocity * velocity =
      (mass * velocity * velocity) * (1 / 2) := by
  ac_rfl

theorem recoveredSumSquaresNormalForm (n : Nat) :
    n * (n + 1) * (2 * n + 1) =
      (n * n) * (2 * n) + n * (2 * n) + (n * n + n) := by
  simp only [Nat.mul_add, Nat.add_mul, Nat.mul_one]

theorem externalKnownFormulaControls :
    (∀ mass velocity : Rat,
      ((1 / 2) * mass) * velocity * velocity =
        (mass * velocity * velocity) * (1 / 2)) ∧
    (∀ n : Nat,
      n * (n + 1) * (2 * n + 1) =
        (n * n) * (2 * n) + n * (2 * n) + (n * n + n)) :=
  ⟨recoveredKineticNormalForm, recoveredSumSquaresNormalForm⟩

end Invariant

#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN"
#eval IO.println "target=Invariant.externalKnownFormulaControls"
#eval IO.println "dependency=Invariant.recoveredKineticNormalForm"
#eval IO.println "dependency=Invariant.recoveredSumSquaresNormalForm"
#eval IO.println "dependency=Nat.add_mul"
#eval IO.println "dependency=Nat.mul_add"
#eval IO.println "dependency=Nat.mul_one"
#eval IO.println "dependency=Lean.Parser.Tactic.ac_rfl"
#eval IO.println "result=checked"
#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_END"
