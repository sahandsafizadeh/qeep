package stream

import (
	"github.com/sahandsafizadeh/qeep/model/internal/contract"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
)

type Stream struct {
	cursor *node.Node
}

type Func func(xs ...*Stream) *Stream
type LayerInitFunc func() (contract.Layer, error)
