package stream

import (
	"github.com/sahandsafizadeh/qeep/model/internal/node"
	"github.com/sahandsafizadeh/qeep/model/internal/types"
)

type Stream struct {
	cursor *node.Node
}

type Func func(xs ...*Stream) *Stream
type ForwarderInitFunc func() (types.Layer, error)
