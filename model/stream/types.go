package stream

import (
	"github.com/sahandsafizadeh/qeep/model/internal/contract"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
)

type Stream struct {
	cursor *node.Node
}

type StreamFunc func(xs ...*Stream) *Stream
type LayerInitFunc func() (contract.Layer, error)
