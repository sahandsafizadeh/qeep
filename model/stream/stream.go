package stream

import "github.com/sahandsafizadeh/qeep/model/internal/node"

func (s *Stream) Cursor() *node.Node {
	return s.cursor
}

func NewStream(initFunc LayerInitFunc, xs []*Stream) (y *Stream) {
	var err error
	defer func() {
		if err != nil {
			panic(err)
		}
	}()

	forwarder, err := initFunc()
	if err != nil {
		return
	}

	cursor, err := node.NewNode(forwarder)
	if err != nil {
		return
	}

	for _, x := range xs {
		err = x.cursor.AddChild(cursor)
		if err != nil {
			return
		}

		err = cursor.AddParent(x.cursor)
		if err != nil {
			return
		}
	}

	return &Stream{cursor: cursor}
}

func NewStreamFunc(initFunc LayerInitFunc) Func {
	return func(xs ...*Stream) *Stream { return NewStream(initFunc, xs) }
}
