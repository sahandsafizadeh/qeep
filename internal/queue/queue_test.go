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
		if val, err := q.Dequeue(); err != nil {
			t.Fatal(err)
		} else if val != 1 {
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
			if value, err := q.Dequeue(); err != nil {
				t.Fatal(err)
			} else if value != exp {
				t.Fatalf("expected (%d) as dequeue value: got %d", exp, value)
			}
		}
	})

	t.Run("drained queue / IsEmpty() / returns true", func(t *testing.T) {
		q := queue.NewQueue[int]()
		q.Enqueue(1)
		_, err := q.Dequeue()
		if err != nil {
			t.Fatal(err)
		} else if !q.IsEmpty() {
			t.Fatal("expected queue to be empty")
		}
	})

	// ============================== error handling ==============================

	t.Run("empty queue / Dequeue() / returns error", func(t *testing.T) {
		q := queue.NewQueue[int]()
		_, err := q.Dequeue()
		if err == nil {
			t.Fatal("expected error as queue is empty")
		} else if err.Error() != "can not dequeue as queue is empty" {
			t.Fatal("unexpected error message returned")
		}
	})
}
