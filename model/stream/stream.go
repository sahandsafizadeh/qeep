package stream

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/sahandsafizadeh/qeep/model/internal/node"
)

func (s *Stream) Cursor() any {
	/*
		stream package is exposed while node package is not; therefore,
		'any' type is returned so that node functionalities stay internal
	*/

	return s.cursor
}

func (s *Stream) Error() error {
	if len(s.errCtx) == 0 {
		return nil
	}

	errs := s.errCtx
	msgs := make([]string, len(errs))

	for i, err := range errs {
		msgs[i] = err.Error()
	}

	slices.Sort(msgs)

	var chained strings.Builder
	for _, msg := range msgs {
		chained.WriteString("\n")
		chained.WriteString(msg)
	}

	return errors.New(chained.String())
}

func NewStream(initFunc layerInitFunc, xs []*Stream) (y *Stream) {

	/*
		CHANGE WITH CAUTION:
		- this function only handles error of 'initFunc'.
		- this function is the only place where nodes are initialized.
		- 'err' value is handled after for loop.
	*/

	nlayer := -1
	errCtx := make([]error, 0)

	layer, err := initFunc()
	cursor := node.NewNode(layer)

	for _, x := range xs {
		if x.nlayer >= nlayer {
			nlayer = x.nlayer
		}

		x.cursor.AddChild(cursor)
		cursor.AddParent(x.cursor)
		errCtx = append(errCtx, x.errCtx...)
	}

	nlayer++

	if err != nil {
		err = fmt.Errorf("(Layer %d): %w", nlayer, err)
		errCtx = append(errCtx, err)
	}

	return &Stream{
		nlayer: nlayer,
		cursor: cursor,
		errCtx: errCtx,
	}
}

func NewStreamFunc(initFunc layerInitFunc) StreamFunc {
	return func(xs ...*Stream) *Stream { return NewStream(initFunc, xs) }
}
