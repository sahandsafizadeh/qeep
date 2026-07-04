# Model Forward-Traversal: Merge Component + Comprehensive Tests

## Context

We recently fixed backpropagation (`tensor/internal/gradtrack/back_propagation.go`): the new
algorithm counts *incoming edges* per node (`unconsumed`) and only propagates a node's gradient
upstream once **all** contributions have arrived — a proper topological order instead of a naive BFS.

The model's **forward pass** has the *same* flaw the old backprop had.
[`model/traverse.go`](model/traverse.go) `traverseBFS` is a plain BFS over the node graph with
**no visited set and no dependency gating**:

- `node.Forward()` reads each parent's cached `p.result` (`model/internal/node/node.go:54-68`).
  A node is therefore only safe to forward **after every parent** has forwarded.
- On an **unequal-depth fan-in (merge)** graph, `traverseBFS` dequeues the merge node before its
  deeper parent has produced a result → it forwards with a `nil` parent tensor → error.
- On a **symmetric diamond**, the shared merge node is enqueued once per parent → it is forwarded
  **multiple times** (wasteful; and outright wrong for the `optimize` traversal that shares this code).

Only chains and depth-balanced diamonds happen to work today.

**This task delivers the executable spec** (per decision): a merge component to make fan-in graphs
buildable, plus a comprehensive forward-traversal test suite that is **RED** against the current
`traverseBFS`. We intentionally **do not** fix `traverse.go` here — that fix (rewrite `traverseBFS`
as Kahn's algorithm: in-degree = `len(node.Parents())`, process each node once after all parents)
is a follow-up, mirroring how backprop was fixed after its tests existed.

## Why a new component is required

Every existing layer rejects >1 input (e.g. `Tanh.toValidInputs` at
`component/layers/activations/tanh.go:36-44`, `Input` requires 0). The node graph already supports
fan-in — `stream.NewStream` wires N parents via `AddParent`/`AddChild`
(`model/stream/stream.go:58-68`) and `node.Forward` feeds all parent results as the variadic `xs` —
but there is **no layer that consumes multiple inputs**, so a merge/diamond graph cannot be built
through the public `stream` API without one.

## Deliverables

### 1. Element-wise `Add` merge component
New file `component/layers/add.go`, following the exact shape of
[`component/layers/activations/tanh.go`](component/layers/activations/tanh.go)
(`Forward` wrapper → validation helper → inner `forward`):

- `type Add struct{}` + `func NewAdd() *Add`.
- `Forward(xs ...tensor.Tensor)`: validate `len(xs) >= 2` (error message in the existing style,
  e.g. `"expected at least two input tensors: got (%d)"`), then element-wise sum by
  loop-accumulating `acc, err = acc.Add(xs[i])` over `tensor.Tensor.Add`. Shape-preserving, so
  branch/merge math stays trivial to assert.
- Plain `Layer` (no weights), so `disableGrad`/`optimize` treat it as a no-op — `Predict` works with
  a bare `&model.ModelConfig{}` (no loss/optimizer needed).
- Small component-level test `component/layers/add_test.go` mirroring the activation tests: valid
  2- and 3-input sums, and the `< 2` inputs validation error.

### 2. `stream.Add()` wrapper
Add to [`model/stream/built_in_layers.go`](model/stream/built_in_layers.go), following the `Tanh()`
pattern exactly:
```go
func Add() StreamFunc {
    initf := func() (contract.Layer, error) { return layers.NewAdd(), nil }
    return NewStreamFunc(initf)
}
```
Usage: `merged := stream.Add()(branchA, branchB)`.

### 3. Comprehensive forward-traversal tests (the spec)
New file `model/traverse_test.go`, package `model_test` (consistent with existing
`model/model_test.go`). Black-box via the public API: build streams → `NewModel` /
`NewMultiInputModel` → `Predict` → assert the output tensor, all inside
`tensor.RunTestLogicOnDevices(func(dev tensor.Device){ ... })`.

**Trick for predictable math:** feed **positive** inputs and use `stream.Relu()` as an
identity op (`Relu(x)=x` for `x>=0`), so every branch is a pure copy and `stream.Add()` merges are
exact sums — the forward analog of the `Scale`/`Add` arithmetic in the backprop tests. Under the
broken `traverseBFS`, the asymmetric-merge cases surface a `nil`-parent error from `Predict`
(→ RED); once `traverseBFS` is made topological they return the expected sums (→ GREEN).

Scenarios (mirroring the backprop coverage in
`tensor/tensor_test/backward_test/back_propagation_test.go`):

| # | Scenario | Graph sketch (Relu = identity) | Expected | Current impl |
|---|----------|--------------------------------|----------|--------------|
| 1 | Simple chain (baseline) | `in→Relu→Relu→Relu` | `x` | GREEN |
| 2 | Symmetric diamond | `x1=Relu(in); x2=Relu(in); Add(x1,x2)` | `2x` | value GREEN* |
| 3 | **Asymmetric diamond** | `p1=Relu(in); b=Relu(Relu(in)); Add(p1,b)` | `2x` | RED (nil parent) |
| 4 | Residual/skip over a non-leaf | `x=Relu(in); f=Relu(Relu(x)); Add(f,x)` | `2x` | RED |
| 5 | 3-way merge, unequal depths | `Add(Relu(in), Relu²(in), Relu³(in))` | `3x` | RED |
| 6 | Nested merges at two depths | `h=Relu(in); p1=Relu(h); p2=Relu(p1); s=Add(p2,h); Add(s,p1)` | `3x` | RED |
| 7 | Long pending chain before a shared merge | `c=Relu(in); deep=Relu⁴(c); Add(deep,c)` | `2x` | RED |
| 8 | Multi-input model merge (unequal branches) | `in1,in2; Add(Relu(in1), Relu²(in2))` via `NewMultiInputModel` | `x1+x2` | RED |
| 9 | Output past the merge | any of the above with a trailing `Relu(merge)` | same | RED |
| 10 | **Forward-exactly-once** (dedup) | symmetric diamond whose merge is a counting spy layer | spy `calls==1` | RED (calls==2) |

\*Scenario 2's value assertion passes even today (both branches equal depth); its purpose is a
control against #3. Scenario 10 turns the same shape into a structural assertion that catches the
duplicate-forward defect that value assertions cannot.

**Counting spy layer (scenario 10):** a test-local
`type spyAdd struct { calls int }` implementing `contract.Layer.Forward` (increments `calls`, then
sums like `Add`). Wire it through the public API with
`stream.NewStreamFunc(func() (contract.Layer, error) { return sp, nil })` — `NewStreamFunc` is
exported and a func literal is assignable to its unexported `layerInitFunc` param; `model_test`
(living in the `model/` dir) may import the `model/internal/contract` internal package. Capture `sp`
in the closure, run `Predict`, then assert `sp.calls == 1`.

### 4. Leave `model/traverse.go` unchanged
The tests stand as the RED spec for the follow-up `traverseBFS` topological fix.

## Critical files

- New: `component/layers/add.go`, `component/layers/add_test.go`, `model/traverse_test.go`.
- Edit: `model/stream/built_in_layers.go` (add `Add()` wrapper).
- Reference patterns: `component/layers/activations/tanh.go` (component shape),
  `model/model_test.go` (test/build/assert style, `RunTestLogicOnDevices`, initializer usage),
  `model/internal/node/node.go:54-68` (why parent-readiness matters),
  `model/model.go:56-72,179-205` (`Predict`→`disableGrad`→`feed`→`seed`→`forward`).

## Verification

- `go build ./...` — the new component and stream wrapper compile.
- `go test ./component/layers/...` — the `Add` component tests pass (component is independently correct).
- `go test ./model/...` — the forward-traversal suite runs. Expected outcome **with `traverse.go`
  unchanged**: scenario 1 and scenario 2's value assertion pass; scenarios 3–10 **fail**
  (asymmetric cases via a `Predict` error mentioning a nil/invalid parent input; scenario 10 via
  `calls == 2`). This RED result is the deliverable and confirms the tests actually exercise the flaw.
- Sanity check that the suite is *correct* (not just red): temporarily prototype the topological
  `traverseBFS` (in-degree `len(node.Parents())`, Kahn's algorithm) locally and confirm the whole
  suite goes GREEN — then discard the prototype, since fixing `traverse.go` is out of scope here.
