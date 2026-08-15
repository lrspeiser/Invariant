import Std.Tactic

/-!
Engine-discovered Collatz halving relation, in its honest conditional form.

The Invariant conjecture engine, given only the raw total stopping times
`sigma(1..64)`, proposed `sigma(2n) = sigma(n) + 1` and confirmed it on 26 held-out
rows.  That empirical statement presupposes both stopping times exist, which is the
open part of the Collatz conjecture and is NOT claimed here.

What IS provable without any termination assumption is the conditional form: if `n`
reaches 1 in exactly `k` steps, then `2 * n` reaches 1 in exactly `k + 1` steps.
Reachability is stated inductively, so no partial function and no well-founded
recursion on the open conjecture is needed.

This file proves nothing about whether any `n` reaches 1 at all.
-/

namespace Invariant

/-- One Collatz step: halve when even, otherwise `3n + 1`. -/
def collatzStep (n : Nat) : Nat :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

/-- `CollatzReaches n k`: starting from `n`, exactly `k` steps arrive at 1. -/
inductive CollatzReaches : Nat → Nat → Prop
  | done : CollatzReaches 1 0
  | step {n k : Nat} (ne_one : n ≠ 1)
      (tail : CollatzReaches (collatzStep n) k) : CollatzReaches n (k + 1)

/-- Doubling then stepping returns to the start: `collatzStep (2n) = n`. -/
theorem collatzStep_double (n : Nat) : collatzStep (2 * n) = n := by
  unfold collatzStep
  rw [if_pos (Nat.mul_mod_right 2 n)]
  omega

/-- The engine-discovered relation, conditionally: reaching 1 from `n` in `k`
    steps forces reaching 1 from `2 * n` in `k + 1` steps. -/
theorem collatzReaches_double {n k : Nat} (pos : 1 ≤ n)
    (reaches : CollatzReaches n k) : CollatzReaches (2 * n) (k + 1) := by
  have ne_one : 2 * n ≠ 1 := by omega
  have back : collatzStep (2 * n) = n := collatzStep_double n
  exact CollatzReaches.step ne_one (back ▸ reaches)

/-- Corollary chain at a concrete point: 1 -> 2 -> 4 reach 1 in 0, 1, 2 steps. -/
theorem collatzReaches_four : CollatzReaches 4 2 := by
  have one : CollatzReaches 1 0 := CollatzReaches.done
  have two : CollatzReaches 2 1 := by
    simpa using collatzReaches_double (Nat.le_refl 1) one
  simpa using collatzReaches_double (by omega : 1 ≤ 2) two

end Invariant

#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN"
#eval IO.println "target=Invariant.collatzReaches_double"
#eval IO.println "dependency=Invariant.collatzStep"
#eval IO.println "dependency=Invariant.CollatzReaches"
#eval IO.println "dependency=Invariant.collatzStep_double"
#eval IO.println "dependency=Nat.mul_mod_right"
#eval IO.println "dependency=Lean.Parser.Tactic.omega"
#eval IO.println "result=checked"
#eval IO.println "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_END"
