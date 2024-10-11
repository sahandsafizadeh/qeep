package forward_test

import (
	"math"
	"testing"

	qti "github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func TestScale(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Scale(0.)

		exp, err := qti.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([]float64{-1., 0., 1.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Scale(-1.)

		exp, err = qti.TensorOf([]float64{1., 0., -1.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([][]float64{
			{-5., 2.},
			{3., -4.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Scale(0.5)

		exp, err = qti.TensorOf([][]float64{
			{-2.5, 1.},
			{1.5, -2.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Scale(5.)

		exp, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestPow(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf(math.E, conf)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Pow(0.)

		exp, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([]float64{-1., 0., 1.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Pow(1.)

		exp, err = qti.TensorOf([]float64{-1., 0., 1.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([][]float64{
			{-2., 2.},
			{-1., 0.5},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Pow(-2)

		exp, err = qti.TensorOf([][]float64{
			{0.25, 0.25},
			{1., 4.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Pow(1000)

		exp, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestExp(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Exp()

		exp, err := qti.TensorOf(math.E, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([]float64{-1., 0., 1.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Exp()

		exp, err = qti.TensorOf([]float64{1 / math.E, 1., math.E}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([][]float64{
			{1., -1.},
			{-2., 0.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Exp()

		exp, err = qti.TensorOf([][]float64{
			{math.E, 1 / math.E},
			{1 / (math.E * math.E), 1.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Exp()

		exp, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestLog(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf(-1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Log()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !math.IsNaN(val) {
			t.Fatalf("expected scalar tensors value to be (NaN): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Log()

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !math.IsInf(val, -1) {
			t.Fatalf("expected scalar tensors value to be (-Inf): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(math.E, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Log()

		exp, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([]float64{1., math.E, math.E * math.E}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Log()

		exp, err = qti.TensorOf([]float64{0., 1., 2.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([][]float64{{math.E, 1.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Log()

		exp, err = qti.TensorOf([][]float64{{1., 0.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Log()

		exp, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestSin(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Sin()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.-1e-10 < val && val < 0.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (0): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(math.Pi/6, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Sin()

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.5-1e-10 < val && val < 0.5+1e-10) {
			t.Fatalf("expected scalar tensors value to be (0.5): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(math.Pi/2, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Sin()

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(1.-1e-10 < val && val < 1.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (1): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Sin()

		exp, err := qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestCos(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Cos()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(1.-1e-10 < val && val < 1.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (1): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(math.Pi/3, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Cos()

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.5-1e-10 < val && val < 0.5+1e-10) {
			t.Fatalf("expected scalar tensors value to be (0.5): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(math.Pi/2, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Cos()

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.-1e-10 < val && val < 0.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (0): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Cos()

		exp, err := qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestTan(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Tan()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.-1e-10 < val && val < 0.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (0): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(math.Pi/4, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Tan()

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(1.-1e-10 < val && val < 1.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (1): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Tan()

		exp, err := qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestSinh(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Sinh()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.-1e-10 < val && val < 0.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (0): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Sinh()
		c := (math.E*math.E - 1) / (2 * math.E)

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(c-1e-10 < val && val < c+1e-10) {
			t.Fatalf("expected scalar tensors value to be ((e^2-1)/2e): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Sinh()

		exp, err := qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestCosh(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Cosh()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(1.-1e-10 < val && val < 1.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (1): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Cosh()
		c := (math.E*math.E + 1) / (2 * math.E)

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(c-1e-10 < val && val < c+1e-10) {
			t.Fatalf("expected scalar tensors value to be ((e^2+1)/2e): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Cosh()

		exp, err := qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestTanh(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Tanh()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.-1e-10 < val && val < 0.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (0): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(math.Inf(-1), conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Tanh()

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(-1.-1e-10 < val && val < -1.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (-1): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(math.Inf(+1), conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Tanh()

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(1.-1e-10 < val && val < 1.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (1): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Tanh()
		c := (math.E*math.E - 1) / (math.E*math.E + 1)

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(c-1e-10 < val && val < c+1e-10) {
			t.Fatalf("expected scalar tensors value to be ((e^2-1)/(e^2+1)): got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Tanh()

		exp, err := qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestEq(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Eq(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{0., math.E + 1e-10}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{0., math.E}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Eq(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{1., 0.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Eq(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{0., 0., 0.},
			{0., 0., 0.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Eq(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestNe(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Ne(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{0., math.E + 1e-10}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{0., math.E}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ne(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{0., 1.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ne(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{1., 1., 1.},
			{1., 1., 1.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ne(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestGt(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Gt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{0., math.E + 1e-10}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{0., math.E}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Gt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{0., 1.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Gt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{1., 1., 1.},
			{0., 0., 0.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Gt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestGe(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Ge(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{0., math.E + 1e-10}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{0., math.E}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ge(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{1., 1.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ge(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{1., 1., 1.},
			{0., 0., 0.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ge(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestLt(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Lt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{0., math.E + 1e-10}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{0., math.E}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Lt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{0., 0.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Lt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{0., 0., 0.},
			{1., 1., 1.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Lt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestLe(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Le(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{0., math.E + 1e-10}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{0., math.E}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Le(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{1., 0.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Le(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{0., 0., 0.},
			{1., 1., 1.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Le(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestElMax(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.ElMax(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{0., math.E + 1e-10}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{0., math.E}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.ElMax(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{0., math.E + 1e-10}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.ElMax(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{1., 2., 3.},
			{1., 2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.ElMax(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestElMin(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.ElMin(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{0., math.E + 1e-10}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{0., math.E}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.ElMin(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{0., math.E}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.ElMin(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{-1., -2., -3.},
			{-1., -2., -3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.ElMin(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestAdd(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(4., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(2., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Add(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(6., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{1., 2., 3.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{4., 5., 6.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Add(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{5., 7., 9.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][][]float64{
			{{-5.}, {1.}},
			{{-9.}, {2.}},
			{{2.}, {-1.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][][]float64{
			{{-4.}, {7.}},
			{{6.}, {2.}},
			{{-3.}, {4.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Add(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][][]float64{
			{{-9.}, {8.}},
			{{-3.}, {4.}},
			{{-1.}, {3.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{3, 1, 5, 1, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 2, 3, 4, 1, 6, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Add(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Ones([]int32{1, 2, 3, 4, 5, 6, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestSub(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(4., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(2., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Sub(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(2., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{1., 2., 3.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{4., 5., 6.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Sub(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{-3., -3., -3.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][][]float64{
			{{-5.}, {1.}},
			{{-9.}, {2.}},
			{{2.}, {-1.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][][]float64{
			{{-4.}, {7.}},
			{{6.}, {2.}},
			{{-3.}, {4.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Sub(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][][]float64{
			{{-1.}, {-6.}},
			{{-15.}, {0.}},
			{{5.}, {-5.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{3, 1, 5, 1, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Zeros([]int32{1, 2, 3, 4, 1, 6, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Sub(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Ones([]int32{1, 2, 3, 4, 5, 6, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestMul(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(4., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(2., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Mul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(8., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{1., 2., 3.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{4., 5., 6.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Mul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{4., 10., 18.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][][]float64{
			{{-5.}, {1.}},
			{{-9.}, {2.}},
			{{2.}, {-1.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][][]float64{
			{{-4.}, {7.}},
			{{6.}, {2.}},
			{{-3.}, {4.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Mul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][][]float64{
			{{20.}, {7.}},
			{{-54.}, {4.}},
			{{-6.}, {-4.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{3, 1, 5, 1, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Zeros([]int32{1, 2, 3, 4, 1, 6, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Mul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 2, 3, 4, 5, 6, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestDiv(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf(4., conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf(2., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Div(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(2., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{4., 5., 6.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{1., 2., 3.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Div(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([]float64{4., 2.5, 2.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][][]float64{
			{{0.}, {1.}},
			{{2.}, {3.}},
			{{4.}, {5.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][][]float64{
			{{-1.}, {1.}},
			{{-2.}, {2.}},
			{{-4.}, {5.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Div(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][][]float64{
			{{0.}, {1.}},
			{{-1.}, {1.5}},
			{{-1.}, {1.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{3, 1, 5, 1, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 2, 3, 4, 1, 6, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Div(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 2, 3, 4, 5, 6, 7}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestDot(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf([]float64{5.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf([]float64{2.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Dot(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(10., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([]float64{0., 1., 2.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([]float64{3., 4., 5.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Dot(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf(14., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{5, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{5, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Dot(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Ones([]int32{5}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{5, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{5, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Dot(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Full([]int32{5}, 4., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{4, 5, 1, 7, 8}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 2, 3, 4, 1, 6, 1, 8}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Dot(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Full([]int32{1, 2, 3, 4, 5, 6, 7}, 8., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestMatMul(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.TensorOf([][]float64{{5.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.TensorOf([][]float64{{2.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf([][]float64{{10.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{{8.}, {4.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{{2.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{{16.}, {8.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{{2.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{{8., 4.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{{16., 8.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{
			{0., 1.},
			{2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{
			{4., 5.},
			{6., 7.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{6., 7.},
			{26., 31.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.TensorOf([][]float64{
			{1., 1., 1.},
			{2., 2., 2.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.TensorOf([][]float64{
			{3., 3.},
			{4., 4.},
			{5., 5.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{12., 12.},
			{24., 24.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{5, 1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{5, 1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Ones([]int32{5, 1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{4, 2, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{4, 1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Ones([]int32{4, 2, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{4, 1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{4, 1, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Ones([]int32{4, 1, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{5, 4, 2, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{5, 4, 2, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Full([]int32{5, 4, 2, 2}, 2., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = qti.Ones([]int32{7, 6, 1, 4, 3, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Ones([]int32{1, 5, 1, 2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Full([]int32{7, 6, 5, 4, 3, 3}, 2., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestValidationBinaryOperators(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.Zeros([]int32{6, 5, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.Zeros([]int32{6, 4, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		_, err = t1.Eq(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Ne(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Gt(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Ge(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Lt(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Le(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.ElMax(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.ElMin(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Equals(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		/* --------------- */

		_, err = t1.Eq(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Ne(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Gt(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Ge(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Lt(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Le(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.ElMax(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.ElMin(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Equals(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{1, 5, 2, 4, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Zeros([]int32{6, 4, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Add(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Sub(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Mul(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Div(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		/* --------------- */

		_, err = t1.Add(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (2)")
		} else if err.Error() != "tensors' broadcasting failed: input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (2): got shape (6)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Sub(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (2)")
		} else if err.Error() != "tensors' broadcasting failed: input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (2): got shape (6)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Mul(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (2)")
		} else if err.Error() != "tensors' broadcasting failed: input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (2): got shape (6)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Div(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (2)")
		} else if err.Error() != "tensors' broadcasting failed: input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (2): got shape (6)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationDot(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.Zeros([]int32{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Dot(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Dot(t1)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than 1 dimension")
		} else if err.Error() != "tensors' dimension validation failed: expected tensors to have at least (1) dimension for dot product: got (0) and (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.Dot(t2)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than 1 dimension")
		} else if err.Error() != "tensors' dimension validation failed: expected tensors to have at least (1) dimension for dot product: got (0) and (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t2.Dot(t1)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than 1 dimension")
		} else if err.Error() != "tensors' dimension validation failed: expected tensors to have at least (1) dimension for dot product: got (1) and (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Zeros([]int32{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Dot(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at last dimension")
		} else if err.Error() != "tensors' dimension validation failed: expected sizes to match at last dimensions: (3) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{8, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Zeros([]int32{5, 2, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Dot(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "tensors' broadcasting failed: input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (8)" {
			t.Fatal("unexpected error message returned")
		}

		/* -------------------- */

	})
}

func TestValidationMatMul(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		t1, err := qti.Zeros([]int32{3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := qti.Zeros([]int32{2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.MatMul(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "tensors' device validation failed: expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.MatMul(t1)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than (2) dimensions")
		} else if err.Error() != "tensors' dimension validation failed: expected tensors to have at least (2) dimensions for matrix multiplication: got (1) and (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t1.MatMul(t2)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than (2) dimensions")
		} else if err.Error() != "tensors' dimension validation failed: expected tensors to have at least (2) dimensions for matrix multiplication: got (1) and (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = t2.MatMul(t1)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than (2) dimensions")
		} else if err.Error() != "tensors' dimension validation failed: expected tensors to have at least (2) dimensions for matrix multiplication: got (2) and (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{3, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Zeros([]int32{2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.MatMul(t2)
		if err == nil {
			t.Fatalf("expected error because of size incompatiblity in the last 2 dimensions for matrix multiplication")
		} else if err.Error() != "tensors' dimension validation failed: expected dimension (1) of first tensor to be equal to dimension (0) of second tensor for matrix multiplication: (3) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{5, 5, 3, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Zeros([]int32{5, 3, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.MatMul(t2)
		if err == nil {
			t.Fatalf("expected error because of size incompatiblity in the last 2 dimensions for matrix multiplication")
		} else if err.Error() != "tensors' dimension validation failed: expected dimension (3) of first tensor to be equal to dimension (1) of second tensor for matrix multiplication: (2) != (3)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = qti.Zeros([]int32{6, 4, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = qti.Zeros([]int32{6, 2, 5, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.MatMul(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (2)")
		} else if err.Error() != "tensors' broadcasting failed: input shape validation failed: expected target shape to be (5) or source size to be (1) at dimension (2): got shape (6)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
