package stream

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/sahandsafizadeh/qeep/model/internal/node"
)

// Cursor returns the internal node for this stream. Used by the model package to traverse the graph.
// The return type is any because the node package is internal.
func (s *Stream) Cursor() any {
	/*
		stream package is exposed while node package is not; therefore,
		'any' type is returned so that node functionalities stay internal
	*/

	return s.cursor
}

// Error returns a combined error if any layer in this stream's subtree failed during construction.
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

// NewStream creates a new stream by running initFunc to get a layer and connecting it to the given input streams.
// Callers typically use stream constructors (Input, FC, Tanh, etc.) instead of calling NewStream directly.
func NewStream(initFunc layerInitFunc, xs []*Stream) (y *Stream) {
	/*
		CHANGE WITH CAUTION:
		- this function only handles error of 'initFunc'.
		- this function is the only place where nodes are initialized.
		- 'err' value is handled after for loop.
	*/

	layer, err := initFunc()
	cursor := node.NewNode(layer)

	nlayer := -1
	errCtx := make([]error, 0)

	for _, x := range xs {
		parent := x.cursor

		if pnl := parent.NLayer(); pnl >= nlayer {
			nlayer = pnl
		}

		parent.AddChild(cursor)
		cursor.AddParent(parent)
		errCtx = append(errCtx, x.errCtx...)
	}

	nlayer++
	cursor.SetNLayer(nlayer)

	if err != nil {
		err = fmt.Errorf("(Layer %d): %w", cursor.NLayer(), err)
		errCtx = append(errCtx, err)
	}

	return &Stream{
		cursor: cursor,
		errCtx: errCtx,
	}
}

// NewStreamFunc wraps a layer init function into a StreamFunc (e.g. used by FC, Tanh).
func NewStreamFunc(initFunc layerInitFunc) StreamFunc {
	return func(xs ...*Stream) *Stream { return NewStream(initFunc, xs) }
}
