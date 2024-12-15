package stream

import (
	"github.com/sahandsafizadeh/qeep/model/internal/contract"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
)

type Stream struct {
	nlayer int
	cursor *node.Node
	errCtx []error
}

type StreamFunc func(xs ...*Stream) *Stream
type layerInitFunc func() (contract.Layer, error)
