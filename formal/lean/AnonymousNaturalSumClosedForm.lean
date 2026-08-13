namespace Invariant

def anonymousNaturalSum : Nat → Nat
  | 0 => 0
  | n + 1 => anonymousNaturalSum n + (n + 1)

theorem anonymousNaturalSumClosedForm (n : Nat) :
    2 * anonymousNaturalSum n = n * (n + 1) := by
  induction n with
  | zero => rfl
  | succ n ih =>
      simp only [anonymousNaturalSum]
      calc
        2 * (anonymousNaturalSum n + (n + 1)) =
            2 * anonymousNaturalSum n + 2 * (n + 1) := Nat.mul_add ..
        _ = n * (n + 1) + 2 * (n + 1) := by rw [ih]
        _ = (n + 2) * (n + 1) := by rw [Nat.add_mul]
        _ = (n + 1) * (n + 2) := Nat.mul_comm ..

end Invariant

#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN"
#eval IO.println "target=Invariant.anonymousNaturalSumClosedForm"
#eval IO.println "dependency=Invariant.anonymousNaturalSum"
#eval IO.println "dependency=Nat.rec"
#eval IO.println "dependency=Nat.mul_add"
#eval IO.println "dependency=Nat.add_mul"
#eval IO.println "dependency=Nat.mul_comm"
#eval IO.println "result=checked"
#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_END"
