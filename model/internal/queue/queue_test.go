package queue_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/model/internal/queue"
)

func TestQueue(t *testing.T) {

	/* ------------------------------ */

	q := queue.NewQueue[int]()

	/* ------------------------------ */

	if !q.IsEmpty() {
		t.Fatalf("expected queue to be empty")
	}

	_, err := q.Dequeue()
	if err == nil {
		t.Fatalf("expected error as queue is empty")
	}

	/* ------------------------------ */

	q.Enqueue([]int{1})

	if q.IsEmpty() {
		t.Fatalf("expected queue not to be empty")
	}

	value, err := q.Dequeue()
	if err != nil {
		t.Fatal(err)
	} else if value != 1 {
		t.Fatalf("expected (1) as dequeue value: got %d", value)
	}

	/* ------------------------------ */

	q.Enqueue([]int{2, 3, 4})
	q.Enqueue([]int{5, 6})
	q.Enqueue([]int{7})

	value, err = q.Dequeue()
	if err != nil {
		t.Fatal(err)
	} else if value != 2 {
		t.Fatalf("expected (2) as dequeue value: got %d", value)
	}

	value, err = q.Dequeue()
	if err != nil {
		t.Fatal(err)
	} else if value != 3 {
		t.Fatalf("expected (3) as dequeue value: got %d", value)
	}

	value, err = q.Dequeue()
	if err != nil {
		t.Fatal(err)
	} else if value != 4 {
		t.Fatalf("expected (4) as dequeue value: got %d", value)
	}

	value, err = q.Dequeue()
	if err != nil {
		t.Fatal(err)
	} else if value != 5 {
		t.Fatalf("expected (5) as dequeue value: got %d", value)
	}

	value, err = q.Dequeue()
	if err != nil {
		t.Fatal(err)
	} else if value != 6 {
		t.Fatalf("expected (6) as dequeue value: got %d", value)
	}

	value, err = q.Dequeue()
	if err != nil {
		t.Fatal(err)
	} else if value != 7 {
		t.Fatalf("expected (7) as dequeue value: got %d", value)
	}

	/* ------------------------------ */

	_, err = q.Dequeue()
	if err == nil {
		t.Fatalf("expected error as queue is empty")
	}

	if !q.IsEmpty() {
		t.Fatalf("expected queue to be empty")
	}

	/* ------------------------------ */

}
