package queue_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/internal/queue"
)

func TestQueue(t *testing.T) {

	// ============================== main paths ==============================

	t.Run("new queue / IsEmpty() / returns true", func(t *testing.T) {
		q := queue.NewQueue[int]()
		if !q.IsEmpty() {
			t.Fatal("expected queue to be empty")
		}
	})

	t.Run("queue with one enqueued item / IsEmpty() / returns false", func(t *testing.T) {
		q := queue.NewQueue[int]()
		q.Enqueue(1)
		if q.IsEmpty() {
			t.Fatal("expected queue not to be empty")
		}
	})

	t.Run("queue with one enqueued item / Dequeue() / returns that item", func(t *testing.T) {
		q := queue.NewQueue[int]()
		q.Enqueue(1)
		if val := q.Dequeue(); val != 1 {
			t.Fatalf("expected (1) as dequeue value: got %d", val)
		}
	})

	t.Run("queue with multiple batches enqueued / Dequeue() repeatedly / returns items in FIFO order", func(t *testing.T) {
		q := queue.NewQueue[int]()
		q.Enqueue(2, 3, 4)
		q.Enqueue(5, 6)
		q.Enqueue(7)

		expected := []int{2, 3, 4, 5, 6, 7}
		for _, exp := range expected {
			if value := q.Dequeue(); value != exp {
				t.Fatalf("expected (%d) as dequeue value: got %d", exp, value)
			}
		}
	})

	t.Run("drained queue / IsEmpty() / returns true", func(t *testing.T) {
		q := queue.NewQueue[int]()
		q.Enqueue(1)
		q.Dequeue()
		if !q.IsEmpty() {
			t.Fatal("expected queue to be empty")
		}
	})

	// ============================== error handling ==============================

	t.Run("empty queue / Dequeue() / panics", func(t *testing.T) {
		q := queue.NewQueue[int]()

		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic as queue is empty")
			} else if r != "can not dequeue as queue is empty" {
				t.Fatal("unexpected panic message")
			}
		}()

		q.Dequeue()
	})
}
