package node

func NewStream(compInitFunc componentInitializerFunc, xs []*Stream) (y *Stream) {
	errCtx := make([]error, 0)

	cursor, err := NewNode(compInitFunc)
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

func NextStreamFunc1(compInitFunc componentInitializerFunc) StreamFunc1 {
	return func(x1 *Stream) *Stream {
		return NewStream(compInitFunc, []*Stream{x1})
	}
}

func NextStreamFunc2(compInitFunc componentInitializerFunc) StreamFunc2 {
	return func(x1, x2 *Stream) *Stream {
		return NewStream(compInitFunc, []*Stream{x1, x2})
	}
}

func NextStreamFunc3(compInitFunc componentInitializerFunc) StreamFunc3 {
	return func(x1, x2, x3 *Stream) *Stream {
		return NewStream(compInitFunc, []*Stream{x1, x2, x3})
	}
}

func NextStreamFunc4(compInitFunc componentInitializerFunc) StreamFunc4 {
	return func(x1, x2, x3, x4 *Stream) *Stream {
		return NewStream(compInitFunc, []*Stream{x1, x2, x3, x4})
	}
}
