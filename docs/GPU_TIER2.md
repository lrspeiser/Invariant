# GPU Tier 2 plan

## Verified local capability

The local hardware probe found:

| Component | Verified value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| GPU memory | 32,607 MiB |
| Compute capability | 12.0 |
| Driver | 580.88 |
| CUDA toolkit/runtime | 12.9 |
| CuPy | 13.5.1, one CUDA device, successful device-array reduction |
| PyTorch | 2.7.1 CPU-only build |
| JAX | CPU-only backend |
| Numba CUDA | Runtime detected, but the device-object probe raised a Windows access violation |

CuPy with CUDA C/RawKernel is therefore the frozen implementation route. PyTorch, JAX, and Numba are not trusted GPU dependencies in the current environment.

## Appropriate division of work

The Rust enumerator should remain on CPU. It already traverses the complete 1.088-billion-action grammar in 94.1 seconds, and its combinatorial decoding plus SHA-256 commitments are not dense linear algebra.

The GPU becomes useful after Rust screening, for the 17,540,440 sampled-static survivors:

1. Rust writes deterministic fixed-width survivor records by absolute checkpoint block. No JSON row is emitted per survivor.
2. CuPy loads block batches and gathers precomputed per-basis derivative/principal-symbol tensors.
3. Candidate matrices are sparse signed reductions of at most six basis tensors.
4. CUDA evaluates eigenvalues and characteristic-polynomial sentinels across large background/orientation tiles.
5. Only candidates safely separated from zero by a frozen numerical margin may be rejected on GPU.
6. Near-zero, non-finite, ill-conditioned, or precision-sensitive rows return to CPU FP64, interval arithmetic, and exact symbolic checks.

## Numerical safety rule

Consumer GPUs prioritize lower precision. GPU throughput must not become a source of false theory rejection.

For a gate quantity `lambda_min` with an independently bounded numerical error `delta`:

```text
lambda_min + delta < 0     -> safe rejection
lambda_min - delta > 0     -> numerical survivor, still not a proof
otherwise                  -> mandatory CPU/exact recheck
```

The error margin must include floating-point roundoff, kernel reduction order, condition number, and comparison against an FP64 CPU control batch. The GPU may accelerate a kill gate; it may not redefine the gate.

## Required next artifact

The next implementation should add a versioned compact survivor-block format to Generator v2 and a CuPy reader with:

- protocol/config/basis hashes in every block header;
- term IDs, sign mask, ordinal, and candidate hash;
- checksum and processed/survivor counts;
- deterministic batch order;
- CPU-versus-GPU golden matrices and negative controls;
- no observational fields.

After the format is verified, the first GPU run should expand the static background grid and orientation coverage. Covariant characteristic-cone kernels must wait for a genuine tensor-action basis; the current scalar-invariant basis cannot support claims about full relativistic modes.

