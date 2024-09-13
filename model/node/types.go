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

type Stream struct {
	cursor *Node
	errCtx []error
}

type StreamFunc1 func(x1 *Stream) *Stream
type StreamFunc2 func(x1, x2 *Stream) *Stream
type StreamFunc3 func(x1, x2, x3 *Stream) *Stream
type StreamFunc4 func(x1, x2, x3, x4 *Stream) *Stream

type componentInitializerFunc func() (qc.Component, error)
