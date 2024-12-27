package stream_test

import (
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
	"github.com/sahandsafizadeh/qeep/model/stream"
)

func TestStream(t *testing.T) {

	/* ------------------------------ */

	input := stream.Input()

	fc1 := stream.FC(&layers.FCConfig{Inputs: 16, Outputs: 4})(input)

	tanh := stream.Tanh()(fc1)

	fc2 := stream.FC(&layers.FCConfig{Inputs: 4, Outputs: 2})(tanh)

	output := stream.Softmax(nil)(fc2)

	/* --------------- */

	if err := input.Error(); err != nil {
		t.Fatal(err)
	}

	if err := fc1.Error(); err != nil {
		t.Fatal(err)
	}

	if err := tanh.Error(); err != nil {
		t.Fatal(err)
	}

	if err := fc2.Error(); err != nil {
		t.Fatal(err)
	}

	if err := output.Error(); err != nil {
		t.Fatal(err)
	}

	/* ------------------------------ */

	ninput := input.Cursor().(*node.Node)
	nfc1 := fc1.Cursor().(*node.Node)
	ntanh := tanh.Cursor().(*node.Node)
	nfc2 := fc2.Cursor().(*node.Node)
	nsoftmax := output.Cursor().(*node.Node)

	/* --------------- */

	_, ok := ninput.Layer().(*layers.Input)
	if !ok {
		t.Fatalf("expected stream's cursor to have 'Input' layer")
	}

	_, ok = nfc1.Layer().(*layers.FC)
	if !ok {
		t.Fatalf("expected stream's cursor to have 'FC' layer")
	}

	_, ok = ntanh.Layer().(*activations.Tanh)
	if !ok {
		t.Fatalf("expected stream's cursor to have 'Tanh' layer")
	}

	_, ok = nfc2.Layer().(*layers.FC)
	if !ok {
		t.Fatalf("expected stream's cursor to have 'FC' layer")
	}

	_, ok = nsoftmax.Layer().(*activations.Softmax)
	if !ok {
		t.Fatalf("expected stream's cursor to have 'Softmax' layer")
	}

	/* --------------- */

	if !slices.Equal(ninput.Parents(), nil) {
		t.Fatalf("unexpected node Parents")
	}

	if !slices.Equal(nfc1.Parents(), []*node.Node{ninput}) {
		t.Fatalf("unexpected node Parents")
	}

	if !slices.Equal(ntanh.Parents(), []*node.Node{nfc1}) {
		t.Fatalf("unexpected node Parents")
	}

	if !slices.Equal(nfc2.Parents(), []*node.Node{ntanh}) {
		t.Fatalf("unexpected node Parents")
	}

	if !slices.Equal(nsoftmax.Parents(), []*node.Node{nfc2}) {
		t.Fatalf("unexpected node Parents")
	}

	/* --------------- */

	if !slices.Equal(ninput.Children(), []*node.Node{nfc1}) {
		t.Fatalf("unexpected node Children")
	}

	if !slices.Equal(nfc1.Children(), []*node.Node{ntanh}) {
		t.Fatalf("unexpected node Children")
	}

	if !slices.Equal(ntanh.Children(), []*node.Node{nfc2}) {
		t.Fatalf("unexpected node Children")
	}

	if !slices.Equal(nfc2.Children(), []*node.Node{nsoftmax}) {
		t.Fatalf("unexpected node Children")
	}

	if !slices.Equal(nsoftmax.Children(), nil) {
		t.Fatalf("unexpected node Children")
	}

	/* ------------------------------ */

	if ninput.NLayer() != 0 {
		t.Fatalf("unexpected node NLayer")
	}

	if nfc1.NLayer() != 1 {
		t.Fatalf("unexpected node NLayer")
	}

	if ntanh.NLayer() != 2 {
		t.Fatalf("unexpected node NLayer")
	}

	if nfc2.NLayer() != 3 {
		t.Fatalf("unexpected node NLayer")
	}

	if nsoftmax.NLayer() != 4 {
		t.Fatalf("unexpected node NLayer")
	}

	/* ------------------------------ */

}

func TestStreamWithError(t *testing.T) {

	/* ------------------------------ */

	input1 := stream.Input()
	input2 := stream.Input()

	tanh1 := stream.Tanh()(input1)
	tanh2 := stream.Tanh()(input2)

	fc11 := stream.FC(nil)(tanh1)
	fc12 := stream.FC(nil)(tanh2)

	sigmoid1 := stream.Sigmoid()(fc11)
	sigmoid2 := stream.Sigmoid()(fc12)

	fc21 := stream.FC(&layers.FCConfig{Inputs: -1, Outputs: 1})(sigmoid1)
	fc22 := stream.FC(&layers.FCConfig{Inputs: -1, Outputs: 1})(sigmoid2)

	output := stream.Softmax(&activations.SoftmaxConfig{Dim: -2})(fc21, fc22)

	/* ------------------------------ */

	if err := input1.Error(); err != nil {
		t.Fatal(err)
	}

	if err := input2.Error(); err != nil {
		t.Fatal(err)
	}

	/* --------------- */

	if err := tanh1.Error(); err != nil {
		t.Fatal(err)
	}

	if err := tanh2.Error(); err != nil {
		t.Fatal(err)
	}

	/* --------------- */

	if err := fc11.Error(); err == nil {
		t.Fatalf("expected error in the stream")
	} else {
		act := err.Error()
		exp := `
(Layer 2): FC config data validation failed: expected config not to be nil`

		if act != exp {
			t.Fatal("unexpected error message returned")
		}
	}

	if err := fc12.Error(); err == nil {
		t.Fatalf("expected error in the stream")
	} else {
		act := err.Error()
		exp := `
(Layer 2): FC config data validation failed: expected config not to be nil`

		if act != exp {
			t.Fatal("unexpected error message returned")
		}
	}

	/* --------------- */

	if err := sigmoid1.Error(); err == nil {
		t.Fatalf("expected error in the stream")
	} else {
		act := err.Error()
		exp := `
(Layer 2): FC config data validation failed: expected config not to be nil`

		if act != exp {
			t.Fatal("unexpected error message returned")
		}
	}

	if err := sigmoid2.Error(); err == nil {
		t.Fatalf("expected error in the stream")
	} else {
		act := err.Error()
		exp := `
(Layer 2): FC config data validation failed: expected config not to be nil`

		if act != exp {
			t.Fatal("unexpected error message returned")
		}
	}

	/* --------------- */

	if err := fc21.Error(); err == nil {
		t.Fatalf("expected error in the stream")
	} else {
		act := err.Error()
		exp := `
(Layer 2): FC config data validation failed: expected config not to be nil
(Layer 4): FC config data validation failed: expected 'Inputs' to be positive: got (-1)`

		if act != exp {
			t.Fatal("unexpected error message returned")
		}
	}

	if err := fc22.Error(); err == nil {
		t.Fatalf("expected error in the stream")
	} else {
		act := err.Error()
		exp := `
(Layer 2): FC config data validation failed: expected config not to be nil
(Layer 4): FC config data validation failed: expected 'Inputs' to be positive: got (-1)`

		if act != exp {
			t.Fatal("unexpected error message returned")
		}
	}

	/* --------------- */

	if err := output.Error(); err == nil {
		t.Fatalf("expected error in the stream")
	} else {
		act := err.Error()
		exp := `
(Layer 2): FC config data validation failed: expected config not to be nil
(Layer 2): FC config data validation failed: expected config not to be nil
(Layer 4): FC config data validation failed: expected 'Inputs' to be positive: got (-1)
(Layer 4): FC config data validation failed: expected 'Inputs' to be positive: got (-1)
(Layer 5): Softmax config data validation failed: expected 'Dim' not to be negative: got (-2)`

		if act != exp {
			t.Fatal("unexpected error message returned")
		}
	}

	/* ------------------------------ */

}

func TestBuiltInStreams(t *testing.T) {

	/* ------------------------------ */

	x := stream.Input()
	x = stream.Tanh()(x)
	x = stream.Sigmoid()(x)
	x = stream.Softmax(nil)(x)
	x = stream.Relu()(x)
	x = stream.LeakyRelu(nil)(x)
	x = stream.FC(&layers.FCConfig{Inputs: 1, Outputs: 1})(x)

	if err := x.Error(); err != nil {
		t.Fatal(err)
	}

	/* ------------------------------ */

}
