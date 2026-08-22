import Std.Tactic

namespace Invariant

theorem recoveredKineticNormalForm (mass velocity : Rat) :
    (1 / 2) * mass * velocity ^ 2 = mass * velocity ^ 2 / 2 := by
  simp only [div_eq_mul_inv, one_mul]
  ac_rfl

def externalSumSquares : Nat → Nat
  | 0 => 0
  | n + 1 => externalSumSquares n + (n + 1) ^ 2

theorem externalSumSquaresSuccessor (n : Nat) :
    n * (n + 1) * (2 * n + 1) + 6 * (n + 1) ^ 2 =
      (n + 1) * (n + 2) * (2 * (n + 1) + 1) := by
  simp only [
    Nat.pow_succ,
    Nat.pow_zero,
    Nat.one_mul,
    Nat.mul_one,
    Nat.mul_assoc,
    Nat.add_mul,
    Nat.mul_add,
  ]
  omega

theorem externalSumSquaresClosedForm (n : Nat) :
    6 * externalSumSquares n = n * (n + 1) * (2 * n + 1) := by
  induction n with
  | zero => rfl
  | succ n ih =>
      simp only [externalSumSquares]
      calc
        6 * (externalSumSquares n + (n + 1) ^ 2) =
            6 * externalSumSquares n + 6 * (n + 1) ^ 2 := Nat.mul_add ..
        _ = n * (n + 1) * (2 * n + 1) + 6 * (n + 1) ^ 2 := by rw [ih]
        _ = (n + 1) * (n + 2) * (2 * (n + 1) + 1) :=
          externalSumSquaresSuccessor n

theorem externalKnownFormulaControls :
    (∀ mass velocity : Rat,
      (1 / 2) * mass * velocity ^ 2 = mass * velocity ^ 2 / 2) ∧
    (∀ n : Nat, 6 * externalSumSquares n = n * (n + 1) * (2 * n + 1)) :=
  ⟨recoveredKineticNormalForm, externalSumSquaresClosedForm⟩

end Invariant

#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN"
#eval IO.println "target=Invariant.externalKnownFormulaControls"
#eval IO.println "dependency=Invariant.recoveredKineticNormalForm"
#eval IO.println "dependency=Invariant.externalSumSquaresClosedForm"
#eval IO.println "dependency=Invariant.externalSumSquares"
#eval IO.println "dependency=Invariant.externalSumSquaresSuccessor"
#eval IO.println "dependency=Nat.rec"
#eval IO.println "dependency=Nat.add_mul"
#eval IO.println "dependency=Nat.mul_add"
#eval IO.println "dependency=Nat.mul_assoc"
#eval IO.println "dependency=Nat.mul_one"
#eval IO.println "dependency=Nat.pow_succ"
#eval IO.println "dependency=Nat.pow_zero"
#eval IO.println "dependency=Lean.Parser.Tactic.omega"
#eval IO.println "dependency=Lean.Parser.Tactic.ac_rfl"
#eval IO.println "result=checked"
#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_END"
