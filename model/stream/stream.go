package stream

import (
	"errors"
	"strings"

	"github.com/sahandsafizadeh/qeep/model/internal/node"
)

func (s *Stream) Cursor() *node.Node {
	return s.cursor
}

func (s *Stream) Error() error {
	if len(s.errCtx) == 0 {
		return nil
	}

	var errs = s.errCtx
	var chained strings.Builder

	chained.WriteString(errs[0].Error())
	for i := 1; i < len(s.errCtx); i++ {
		chained.WriteString(": ")
		chained.WriteString(errs[i].Error())
	}

	return errors.New(chained.String())
}

func NewStream(initFunc layerInitFunc, xs []*Stream) (y *Stream) {

	// CHANGE WITH CAUTION: this function does not recover nil value of 'layer'

	errCtx := make([]error, 0)

	layer, err := initFunc()
	if err != nil {
		errCtx = append(errCtx, err)
	}

	cursor := node.NewNode(layer)

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

func NewStreamFunc(initFunc layerInitFunc) StreamFunc {
	return func(xs ...*Stream) *Stream { return NewStream(initFunc, xs) }
}
