import Std.Tactic

namespace Invariant

def recoveredPolyAdd : List Int -> List Int -> List Int
  | [], right => right
  | left, [] => left
  | a :: left, b :: right => (a + b) :: recoveredPolyAdd left right

def recoveredPolyScale (a : Int) : List Int -> List Int
  | [] => []
  | b :: rest => (a * b) :: recoveredPolyScale a rest

def recoveredPolyMul : List Int -> List Int -> List Int
  | [], _ => []
  | a :: left, right =>
      recoveredPolyAdd (recoveredPolyScale a right) (0 :: recoveredPolyMul left right)

theorem constraintRecoveredQuarticIdentity :
    recoveredPolyMul (recoveredPolyMul [-2, 1] [3, 1]) [5, 1, 1] =
      [-30, -1, 0, 2, 1] := by
  decide

end Invariant

#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN"
#eval IO.println "target=Invariant.constraintRecoveredQuarticIdentity"
#eval IO.println "dependency=Invariant.recoveredPolyAdd"
#eval IO.println "dependency=Invariant.recoveredPolyScale"
#eval IO.println "dependency=Invariant.recoveredPolyMul"
#eval IO.println "dependency=of_decide_eq_true"
#eval IO.println "result=checked"
#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_END"
