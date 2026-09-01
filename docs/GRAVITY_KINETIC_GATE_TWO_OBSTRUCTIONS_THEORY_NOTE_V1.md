# Two exact local obstructions for derivative-gated two-scalar theories

## Status and scope

This is a theorem-note candidate for expert review, not a claim of a successful theory of gravity.
It studies the local high-frequency scalar block of

\[
P(X,Y)=P_0(X)+Z(X)Y
\]

on aligned homogeneous timelike backgrounds. Here \(X>0\), \(Y>0\), and \(Z>0\). The result does
not establish a causal paradox, a full covariant no-go theorem, a realistic cosmology, or
observational support. Its purpose is to identify two exact design constraints that any theory using
this particular derivative gate must address.

## Principal matrices

Let

\[
C=P_{0,X}+Y Z_X.
\]

For the two scalar perturbations, the local kinetic and spatial-gradient matrices are

\[
K=\begin{pmatrix}
C+2X(P_{0,XX}+Y Z_{XX}) & 2Z_X\sqrt{XY}\\
2Z_X\sqrt{XY} & Z
\end{pmatrix},
\qquad
G=\begin{pmatrix}C&0\\0&Z\end{pmatrix}.
\]

The squared characteristic speeds are the generalized eigenvalues satisfying
\(Gv=c_s^2Kv\). We restrict the theorem to backgrounds on which both \(K\) and \(G\) are symmetric
positive definite.

## Theorem 1: unavoidable metric-cone straddling

If \(X>0\), \(Y>0\), \(Z_X\ne0\), and both \(K\) and \(G\) are positive definite, then the two
generalized scalar speeds obey

\[
0<c_-^2<1<c_+^2.
\]

### Proof

The difference of the two principal matrices has the exact determinant

\[
\det(K-G)=-4XYZ_X^2<0.
\]

Let \(F(s)=\det(G-sK)\), where \(s=c_s^2\). Positive definiteness gives
\(F(0)=\det G>0\), while the identity above gives \(F(1)<0\). Its leading coefficient is
\(\det K>0\), so \(F(s)>0\) at sufficiently large positive \(s\). The generalized eigenvalues are
real and positive because \(K\) and \(G\) are positive definite. Therefore one root lies strictly in
\((0,1)\) and the other lies strictly in \((1,\infty)\). The determinant identity contains no
\(P_0\), so changing \(P_0\) can move the speeds but cannot remove the straddling while the gate is
active. ∎

This theorem says that the scalar cones straddle the metric light cone. It does not, by itself, say
that the faster scalar creates a causal paradox; k-essence can possess a consistent emergent causal
structure on suitable backgrounds. It does say that metric-cone subluminality cannot be obtained in
this architecture merely by choosing a different \(P_0\).

## Theorem 2: finite healthy-mixing range for a growing gate

Let \(u=\beta X^2>0\), define

\[
q(u)=\frac{d\ln Z}{d\ln u},
\]

and isolate the active mixing combination

\[
M=Z(Z_X+2XZ_{XX})-4XZ_X^2.
\]

The exact identity

\[
M=\frac{2Z^2}{X}\left(4\frac{dq}{d\ln u}-q-4q^2\right)
\]

implies the following. If \(q(u_0)=q_0>0\) and \(M\ge0\) throughout a closed interval
\([u_0,U]\), then

\[
\frac{U}{u_0}<\left(1+\frac{1}{4q_0}\right)^4.
\]

### Proof sketch

For \(X>0\), \(M\ge0\) gives the Riccati inequality

\[
\frac{dq}{d\ln u}\ge\frac{q}{4}+q^2.
\]

The equality solution starting from \(q_0\) develops a pole after

\[
\Delta\ln u=4\ln\left(1+\frac{1}{4q_0}\right).
\]

Comparison then forbids a finite \(C^2\) growing gate from reaching that endpoint while preserving
the hypotheses. ∎

For \(Z=(1+u)^p\), the sign boundary is exact:

\[
M\ge0 \quad\Longleftrightarrow\quad 0<u\le\frac{3}{1+4p}.
\]

## What the two theorems jointly imply

The two results are complementary. The second theorem says that a smooth growing gate cannot keep
its active mixing contribution favorable over an unlimited dynamic range. The first theorem says
that even inside a bounded interval where the local kinetic and gradient blocks are healthy, an
active gate necessarily places one scalar characteristic outside the metric light cone.

An exact coupled Einstein-scalar FLRW witness already realizes the bounded case for
\(Z=(1+u)^2\) with \(u<1/3\). Its scalar block is positive, but its largest squared scalar speed is
approximately \(1.598\), as Theorem 1 requires. This is a consistency witness for the mathematics,
not a realistic cosmological model.

## Escape architectures

The strict cone result switches off when \(X=0\), \(Y=0\), or \(Z_X=0\). It also does not apply
when additional derivative operators change both principal matrices. A previously derived split-gate
architecture makes the \(\chi\) kinetic coefficient constant and moves the environmental dependence
into a lower-order mass term. That removes the second-order derivative cross block, although it
introduces separate amplitude, first-derivative mixing, source, and phenomenology constraints.

These are escape routes from the theorem hypotheses, not proven healthy theories.

## Literature boundary

General multi-field perturbation matrices are established in arXiv:0801.1085 and arXiv:0806.0336.
The causal interpretation of superluminal k-essence characteristics is treated in arXiv:0708.0561
and arXiv:gr-qc/0607055. Multiple independent kinetic fields, disformally coupled two-field models,
field-dependent kinetic metrics, and recent multi-field screening appear in arXiv:0806.4368,
arXiv:1510.01650, arXiv:2304.12364, and arXiv:2603.13986v2. A targeted search did not locate the
two explicit results above, but that does not establish historical novelty.

## Required work before a preprint

1. Independent expert verification of the principal matrices, hypotheses, and both proofs.
2. A broader human literature review, including citation chaining and non-arXiv sources.
3. A precise statement of the causal question using the joint scalar, metric, and matter cones.
4. Lower-order perturbation and strong-hyperbolicity analysis on nonconstant backgrounds.
5. Radiative-stability and cutoff analysis for any proposed action.
6. Separation between the mathematical theorem note and any later phenomenological model.

No observational dataset is needed to prove these local mathematical statements. Any later 3D or
observational builder must separately bind a real public source dataset, its primary paper, and an
independent analytic or published solver benchmark before response scoring.
