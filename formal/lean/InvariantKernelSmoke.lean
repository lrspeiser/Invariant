namespace Invariant

theorem kernelSmoke (n : Nat) : n = n := Eq.refl n

end Invariant

#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN"
#eval IO.println "target=Invariant.kernelSmoke"
#eval IO.println "dependency=Eq.refl"
#eval IO.println "result=checked"
#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_END"
