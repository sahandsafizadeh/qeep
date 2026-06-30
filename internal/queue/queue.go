package queue

import "fmt"

type Queue[T any] struct {
	head *node[T]
	tail *node[T]
}

type node[T any] struct {
	value T
	next  *node[T]
}

func NewQueue[T any]() *Queue[T] {
	return new(Queue[T])
}

func (q *Queue[T]) IsEmpty() bool {
	return q.head == nil && q.tail == nil
}

func (q *Queue[T]) Enqueue(values ...T) {
	for _, value := range values {
		q.enqueue(value)
	}
}

func (q *Queue[T]) enqueue(value T) {
	n := &node[T]{value, nil}

	if q.IsEmpty() {
		q.head = n
		q.tail = n
	} else {
		q.tail.next = n
		q.tail = n
	}
}

func (q *Queue[T]) Dequeue() (value T, err error) {
	if q.IsEmpty() {
		return value, fmt.Errorf("can not dequeue as queue is empty")
	}

	value = q.head.value
	q.head = q.head.next

	if q.head == nil {
		q.tail = nil
	}

	return value, nil
}
