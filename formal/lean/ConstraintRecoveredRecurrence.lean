import Std.Tactic

namespace Invariant

def constraintRecoveredSequence : Nat → Nat
  | 0 => 7
  | n + 1 => constraintRecoveredSequence n + (6 * n ^ 2 + 10 * n + 5)

private theorem recoveredPolynomialSuccessor (n : Nat) :
    2 * n ^ 3 + 2 * n ^ 2 + n + 7 + (6 * n ^ 2 + 10 * n + 5) =
      2 * (n + 1) ^ 3 + 2 * (n + 1) ^ 2 + (n + 1) + 7 := by
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

theorem constraintRecoveredSequenceClosedForm (n : Nat) :
    constraintRecoveredSequence n = 2 * n ^ 3 + 2 * n ^ 2 + n + 7 := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [constraintRecoveredSequence, ih]
      exact recoveredPolynomialSuccessor n

end Invariant

#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN"
#eval IO.println "target=Invariant.constraintRecoveredSequenceClosedForm"
#eval IO.println "dependency=Invariant.constraintRecoveredSequence"
#eval IO.println "dependency=Invariant.recoveredPolynomialSuccessor"
#eval IO.println "dependency=Nat.rec"
#eval IO.println "dependency=Nat.add_mul"
#eval IO.println "dependency=Nat.mul_add"
#eval IO.println "dependency=Nat.mul_assoc"
#eval IO.println "dependency=Nat.mul_one"
#eval IO.println "dependency=Nat.pow_succ"
#eval IO.println "dependency=Nat.pow_zero"
#eval IO.println "dependency=Lean.Parser.Tactic.omega"
#eval IO.println "result=checked"
#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_END"
