# ADR 0002: Define-by-run autograd with fused kernels on NumPy

Status: accepted

## Context

The project trains its model with NumPy only. Training must finish on a shared desktop
CPU within minutes, and every gradient must be checkable.

## Decision

`Tensor` records, per operation, its parents and a closure from the output gradient to
the input gradients (a define-by-run graph, as in PyTorch). Only a small set of
primitives is differentiated generically: elementwise arithmetic with broadcasting,
`matmul`, reshape, transpose, indexing, `sum` and `mean`. The operations that dominate
a transformer step are written as fused kernels with hand-derived backward passes:
softmax, log-softmax, GELU, layer norm, embedding gather, cross-entropy, dropout and
masked fill.

Backward builds the topological order with an explicit stack and accumulates gradients
in a dictionary, so a tensor used several times gets the sum and long chains do not hit
Python's recursion limit.

## Consequences

Composing layer norm from primitives would create about ten graph nodes and keep each
intermediate array alive until backward; the fused version keeps two arrays and runs one
vectorized expression. GELU was rewritten with in-place operations after profiling
showed it was the most expensive operation of the step; together with float32 dropout
masks this cut a 1.1M-parameter step from 0.36 s to 0.29 s.

Every fused kernel is a place where a hand-derived formula can be wrong, so each one is
checked against central finite differences in float64 (`lyra.autograd.gradcheck`), and
Hypothesis generates random broadcastable shapes for the binary operations.

There is no GPU path, no mixed precision and no graph-level optimization. That is
acceptable for a 1.1M-parameter model and would not be for a large one.
