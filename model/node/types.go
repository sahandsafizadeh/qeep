package node

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Node struct {
	parents   []*Node
	children  []*Node
	component qc.Component
	output    qt.Tensor
}
