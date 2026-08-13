# Chronological rediscovery of an anonymous finite-sum formula

> **Verdict:** `proved`
> **Disclosure:** Historical-style reconstruction generated from sealed machine receipts. It is not an authentic historical document, private model reasoning, or a replacement for the cited receipts.

## Problem stated without the answer

Let $S(0)=0$ and let the anonymous sequence satisfy

$$S(n+1)-S(n)=n+1.
$$

We seek a closed form using only the declared quadratic rational grammar. The familiar name of the theorem is withheld until the end.

Evidence: `runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json`

## Finite discovery experiment

Start with $q(n)=an^2+bn+c$. The bounded search examined 46,656 raw coefficient triples, reduced them to 12,167 canonical classes, and left 1 survivor on the public examples. The survivor then passed 59 additional exact counterexample points. These tests identify a candidate; they do not prove it.

Evidence: `runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json`

## The conjectured formula

The surviving coefficients are $a=1/2$, $b=1/2$, and $c=0$, hence

$$q(n)=\frac{n^2+n}{2}=\frac{n(n+1)}{2}.
$$

Evidence: `runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json`

## Proof

**Base case.** $q(0)=0=S(0)$.

**Successor step.** Exact polynomial arithmetic gives

$$q(n+1)-q(n)=\frac{(n+1)^2+(n+1)-n^2-n}{2}=n+1.
$$

Assume $q(n)=S(n)$. The defining recurrence and the displayed identity imply

$$q(n+1)=q(n)+(n+1)=S(n)+(n+1)=S(n+1).
$$

Therefore $q(n)=S(n)$ for every nonnegative integer $n$ by induction.

Evidence: `runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json`

## Chronological unsealing

The candidate catalog, counterexample record, and induction proof were sealed before the withheld reference was read. Only afterward was the result compared with the conventional natural-sum identity, and the forms matched exactly. This demonstrates bounded rediscovery mechanics; it is not a novelty claim.

Evidence: `runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json`

## Claim ledger

- **proved:** The discovered quadratic equals the anonymous recurrence-defined sequence for every nonnegative integer.
- **scope_limit:** Only the declared finite quadratic rational grammar was exhausted; no unbounded formula-space or novelty claim follows.

## Receipt bindings

- `runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json` — file `3158d6031ad0dbf1c3cb955c956af319f39c58d50982d2193a07b0f46c83e685`, content `05b7ab12f3876513216394dd578258ea6af242d0b8e6a4911e281a4d713bd5be`

## Limits

- enumeration_exhausts_only_the_declared_finite_quadratic_rational_grammar
- public_examples_are_not_the_universal_proof
- file_read_guards_are_process_local_not_an_operating_system_sandbox
- the_benchmark_rediscovers_a_withheld_known_theorem_and_claims_no_novelty

Notebook content seal: `6de04c91cdc3bc940aa4e578c812b4a083a513ba0466cf69b3686def9464b52e`
