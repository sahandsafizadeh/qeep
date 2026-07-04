# Speed up qeep CPU execution (profile-driven)

## Context

Profiling the MNIST example (`examples/04-mnist/cpu.prof`, a 355s / 5-epoch training run) shows the
CPU time is dominated by three avoidable inefficiencies in the CPU tensor backend. This plan targets
them with **pure-Go, single-threaded, dependency-free** changes (no parallelism, no BLAS). The goal is
a large wall-clock reduction on the existing training loop with minimal risk to correctness.

### What the profile says (326s sampled total)

| Cost | % total | Where | Root cause |
|---|---|---|---|
| `matMul` | **42%** (137s) | `operators.go:185` | Naive triple loop; **52s on a single load** (`operators.go:212`) because `Transpose()` is lazy (`shape_modifiers.go:9`, swaps strides + reuses data), so backprop matmuls (`gradients.go:1016,1029`) stride through memory → cache misses |
| element-wise framework | **~40%** | `operators.go:248-262`, `initializers.go:8` | `applyUnary/BinaryFuncOnTensorsElemWise` recompute a flat index via `at()` (15%, `accessors.go:12`) and run `updateElementWiseIndex` (12%, `initializers.go:141`) **plus a `defer`** on *every element*, instead of walking the contiguous `data` slice |
| `math.Pow` | **~8%** | `operators.go:16` | `Pow(2)`/`Pow(0.5)`/`Pow(-1)`/`Pow(0)` (squaring, sqrt, reciprocal, ones) all routed through general `math.Pow`; Adam runs `delta.Pow(2)` + `Pow(0.5)` every step (`optimizers/adam.go:83,111`) |

---

## Change 1 — Specialize `pow` for common exponents (easiest, ~8%)

**File:** `tensor/internal/cputensor/operators.go` (`pow`, line 15)

Replace the unconditional `math.Pow(a, u)` with a switch selecting a cheap kernel by exponent:

```go
func (t *CPUTensor) pow(u float64) *CPUTensor {
	switch u {
	case 0:
		return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return 1 })
	case 1:
		return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return a })
	case 2:
		return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return a * a })
	case 0.5:
		return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Sqrt(a) })
	case -1:
		return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return 1 / a })
	default:
		return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Pow(a, u) })
	}
}
```

Covers every `Pow` call site found: Adam/AdamW (`Pow(2)`, `Pow(0.5)`), BatchNorm (`Pow(0.5)`), Sigmoid
(`Pow(-1)`), MSE loss/metric (`Pow(2)`), and gradient helpers (`Pow(0)` ones-tensor,
`gradients.go:540` `Pow(a-1)`). `math.Pow(a,2)`≈20× slower than `a*a`; `math.Pow(a,0.5)` slower than
`math.Sqrt`. Numerically identical or better for these exponents.

---

## Change 2 — Contiguous fast path for element-wise ops (biggest, ~40%)

**Files:** `tensor/internal/cputensor/operators.go` (`applyUnaryFuncOnTensorElemWise` line 248,
`applyBinaryFuncOnTensorsElemWise` line 256); helper in `accessors.go`.

The current closures call `t.at(index)` (full stride dot-product) + `defer updateElementWiseIndex(...)`
per element. For a **contiguous** tensor (strides == `util.DimsToStrides(dims)`), the element at
iteration `i` is exactly `data[i]`, so all of that machinery collapses to a linear walk.

1. Add a helper in `accessors.go`:
   ```go
   func (t *CPUTensor) isContiguous() bool {
   	// true when strides match the standard packed layout for dims
   	s := util.DimsToStrides(t.dims)
   	return slices.Equal(t.strd, s)
   }
   ```
2. Rewrite the unary apply to allocate the output contiguously and branch:
   ```go
   func applyUnaryFuncOnTensorElemWise(t *CPUTensor, suf scalarUnaryFunc) *CPUTensor {
   	o := newContiguousTensor(t.dims) // dims/strd/data, no per-elem init
   	if t.isContiguous() {
   		for i, v := range t.data {
   			o.data[i] = suf(v)
   		}
   		return o
   	}
   	// fallback: strided walk WITHOUT defer
   	index := make([]int, len(t.dims))
   	for i := range o.data {
   		o.data[i] = suf(t.at(index))
   		updateElementWiseIndex(index, t.dims)
   	}
   	return o
   }
   ```
   Binary version mirrors this: fast path when **both** operands are contiguous
   (`for i := range o.data { o.data[i] = sbf(t1.data[i], t2.data[i]) }`); otherwise the defer-free
   strided fallback. Broadcasting already materializes operands to matching contiguous dims before
   these are called, so the fast path covers the overwhelming majority of calls (Relu, softmax exp,
   Add/Sub/Mul, BatchNorm, optimizer math, losses).
3. Factor a small `newContiguousTensor(dims)` (dims copy + `DimsToStrides` + `make` data) out of
   `newTensorWithElementWiseInit` so both share allocation logic. Keep
   `newTensorWithElementWiseInit` for the genuine generator cases (`constTensor`, `eyeMatrix`,
   random initializers, `broadcast`), but **remove its `defer` usage** in `broadcast`
   (`shape_modifiers.go:52`) the same way.

This removes `at()` (15%), `updateElementWiseIndex` (12%), the per-element `defer`, and the closure
`fn()` indirection from the hot path.

---

## Change 3 — Pack the strided operand in `matMul` (~30% of the 42%)

**File:** `tensor/internal/cputensor/operators.go` (`matMul`, lines 185-240)

The inner loop reads `t2.data[t2row + j*t2cs]`. In the forward pass `t2cs==1` (contiguous, fast), but
in backprop `t2` is a lazily-transposed tensor so `t2cs` is large → the inner loop cache-thrashes
(the 52s line). Fix: when `t2cs != 1`, **pack the current 2-D `n×k` block of `t2` into a small
contiguous scratch buffer once per batch element**, then run the kernel with unit column stride:

```go
// inside the per-batch-element loop, before kernel:
if t2cs != 1 {
	// copy t2's [n×k] slice at t2ofst into packed[n*k] with row-major layout,
	// then point the kernel at packed with stride 1
}
```

The pack is O(n·k); the kernel is O(m·n·k), so the copy cost is negligible and the inner loop becomes
sequential memory access. Keep the existing fast path untouched when `t2cs == 1`.

(If this proves fiddly with the batched-offset structure, an equally valid variant: detect the pure
2-D transposed case and iterate `t2` in its natural row-major order into the packed buffer.)

---

## Verification

1. **Correctness — unit tests must stay green:**
   `go test ./tensor/... ./component/... ./model/...`
   The cputensor package has existing tests (`operators_test.go`, etc.); Change 1/2/3 must not alter
   numeric results. Add a couple of assertions for the `pow` special cases and a non-contiguous
   element-wise case (transpose then unary op) to exercise the fallback branch.
2. **Speed — before/after on the same workload:** re-run the MNIST example with profiling and compare
   `Duration` and the `-top` breakdown:
   `go tool pprof -top examples/04-mnist/cpu.prof`
   Expect `at`, `updateElementWiseIndex`, and `math.Pow` to largely disappear from the top, and
   `matMul`'s flat time to drop substantially. For a faster iteration loop than the full 355s run,
   add a small `go test -bench` over `matMul` and `Add`/`Pow` on representative shapes
   (e.g. `[64,256]×[256,128]`) and compare `ns/op`.
3. Confirm final MNIST accuracy is unchanged (~0.98) to prove the optimizations are behavior-preserving.

## Notes / out of scope (per user)

- No goroutine parallelism and no BLAS/cgo backend this round — these were the largest *additional*
  multipliers but were explicitly deferred to keep the change pure-Go and low-risk.
- `gonum` is already a dependency (used for random init only); no new modules are added.
