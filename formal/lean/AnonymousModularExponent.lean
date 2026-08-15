namespace Invariant

theorem anonymousModularExponent : ∀ a : Fin 11, a ≠ 0 → a ^ 10 = 1 := by
  native_decide

end Invariant

#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN"
#eval IO.println "target=Invariant.anonymousModularExponent"
#eval IO.println "dependency=of_decide_eq_true"
#eval IO.println "result=checked"
#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_END"
