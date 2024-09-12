package queue

import "fmt"

type Queue[T any] struct {
	data []T
}

func (q *Queue[T]) IsEmpty() (is bool) {
	return len(q.data) == 0
}

func (q *Queue[T]) Enqueue(values []T) {
	q.data = append(q.data, values...)
}

func (q *Queue[T]) Dequeue() (value T, err error) {
	if q.IsEmpty() {
		err = fmt.Errorf("can not dequeue as queue is empty")
		return
	}

	value = q.data[0]
	q.data = q.data[1:]

	return value, nil
}
