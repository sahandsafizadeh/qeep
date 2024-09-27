package stream

import "github.com/sahandsafizadeh/qeep/model/internal/node"

type Stream struct {
	cursor *node.Node
	errCtx []error
}

type StreamFunc1 func(x1 *Stream) *Stream
type StreamFunc2 func(x1, x2 *Stream) *Stream
type StreamFunc3 func(x1, x2, x3 *Stream) *Stream
type StreamFunc4 func(x1, x2, x3, x4 *Stream) *Stream

func NewStream(forwarderInitFunc node.ForwarderInitFunc, xs []*Stream) (y *Stream) {
	errCtx := make([]error, 0)

	cursor, err := node.NewNode(forwarderInitFunc)
	if err != nil {
		errCtx = append(errCtx, err)
	}

	for _, x := range xs {
		x.cursor.AddChild(cursor)
		cursor.AddParent(x.cursor)
		errCtx = append(errCtx, x.errCtx...)
	}

	return &Stream{
		cursor: cursor,
		errCtx: errCtx,
	}
}

func NextStreamFunc1(forwarderInitFunc node.ForwarderInitFunc) StreamFunc1 {
	return func(x1 *Stream) *Stream {
		return NewStream(forwarderInitFunc, []*Stream{x1})
	}
}

func NextStreamFunc2(forwarderInitFunc node.ForwarderInitFunc) StreamFunc2 {
	return func(x1, x2 *Stream) *Stream {
		return NewStream(forwarderInitFunc, []*Stream{x1, x2})
	}
}

func NextStreamFunc3(forwarderInitFunc node.ForwarderInitFunc) StreamFunc3 {
	return func(x1, x2, x3 *Stream) *Stream {
		return NewStream(forwarderInitFunc, []*Stream{x1, x2, x3})
	}
}

func NextStreamFunc4(forwarderInitFunc node.ForwarderInitFunc) StreamFunc4 {
	return func(x1, x2, x3, x4 *Stream) *Stream {
		return NewStream(forwarderInitFunc, []*Stream{x1, x2, x3, x4})
	}
}
