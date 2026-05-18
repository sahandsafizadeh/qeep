package stream

import (
	"github.com/sahandsafizadeh/qeep/model/internal/contract"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
)

// Stream represents a node in the computational graph, wrapping a layer and its connections.
// Compose streams to define the network topology.
type Stream struct {
	cursor *node.Node
	errCtx []error
}

type StreamFunc func(xs ...*Stream) *Stream
type layerInitFunc func() (contract.Layer, error)
