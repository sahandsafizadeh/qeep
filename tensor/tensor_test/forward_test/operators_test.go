package forward_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func TestScale(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Scale(0.)

		exp, err := tinit.TensorOf(conf, 0.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{-1., 0., 1.})
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Scale(-1.)

		exp, err = tinit.TensorOf(conf, []float64{1., 0., -1.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][]float64{
			{-5., 2.},
			{3., -4.},
		})
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Scale(0.5)

		exp, err = tinit.TensorOf(conf, [][]float64{
			{-2.5, 1.},
			{1.5, -2.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Scale(5.)

		exp, err = tinit.Zeros(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, math.E)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Pow(0.)

		exp, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{-1., 0., 1.})
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Pow(1.)

		exp, err = tinit.TensorOf(conf, []float64{-1., 0., 1.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][]float64{
			{-2., 2.},
			{-1., 0.5},
		})
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Pow(-2)

		exp, err = tinit.TensorOf(conf, [][]float64{
			{0.25, 0.25},
			{1., 4.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Pow(1000)

		exp, err = tinit.Ones(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		act := ten.Exp()

		exp, err := tinit.TensorOf(conf, math.E)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{-1., 0., 1.})
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Exp()

		exp, err = tinit.TensorOf(conf, []float64{1 / math.E, 1., math.E})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][]float64{
			{1., -1.},
			{-2., 0.},
		})
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Exp()

		exp, err = tinit.TensorOf(conf, [][]float64{
			{math.E, 1 / math.E},
			{1 / (math.E * math.E), 1.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Exp()

		exp, err = tinit.Ones(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, -1.)
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

		ten, err = tinit.TensorOf(conf, 0.)
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

		ten, err = tinit.TensorOf(conf, math.E)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Log()

		exp, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{1., math.E, math.E * math.E})
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Log()

		exp, err = tinit.TensorOf(conf, []float64{0., 1., 2.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][]float64{{math.E, 1.}})
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Log()

		exp, err = tinit.TensorOf(conf, [][]float64{{1., 0.}})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Log()

		exp, err = tinit.Zeros(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 0.)
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

		ten, err = tinit.TensorOf(conf, math.Pi/6)
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

		ten, err = tinit.TensorOf(conf, math.Pi/2)
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

		ten, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Sin()

		exp, err := tinit.Zeros(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 0.)
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

		ten, err = tinit.TensorOf(conf, math.Pi/3)
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

		ten, err = tinit.TensorOf(conf, math.Pi/2)
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

		ten, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Cos()

		exp, err := tinit.Ones(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 0.)
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

		ten, err = tinit.TensorOf(conf, math.Pi/4)
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

		ten, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Tan()

		exp, err := tinit.Zeros(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 0.)
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

		ten, err = tinit.TensorOf(conf, 1.)
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

		ten, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Sinh()

		exp, err := tinit.Zeros(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 0.)
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

		ten, err = tinit.TensorOf(conf, 1.)
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

		ten, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Cosh()

		exp, err := tinit.Ones(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 0.)
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

		ten, err = tinit.TensorOf(conf, math.Inf(-1))
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

		ten, err = tinit.TensorOf(conf, math.Inf(+1))
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

		ten, err = tinit.TensorOf(conf, 1.)
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

		ten, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act = ten.Tanh()

		exp, err := tinit.Zeros(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Eq(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, []float64{0., math.E + 1e-10})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, []float64{0., math.E})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Eq(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{1., 0.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Eq(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{0., 0., 0.},
			{0., 0., 0.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Eq(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Ones(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Ne(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 0.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, []float64{0., math.E + 1e-10})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, []float64{0., math.E})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ne(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{0., 1.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ne(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{1., 1., 1.},
			{1., 1., 1.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ne(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Gt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 0.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, []float64{0., math.E + 1e-10})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, []float64{0., math.E})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Gt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{0., 1.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Gt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{1., 1., 1.},
			{0., 0., 0.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Gt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Ge(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, []float64{0., math.E + 1e-10})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, []float64{0., math.E})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ge(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{1., 1.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ge(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{1., 1., 1.},
			{0., 0., 0.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Ge(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Lt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 0.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, []float64{0., math.E + 1e-10})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, []float64{0., math.E})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Lt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{0., 0.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Lt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{0., 0., 0.},
			{1., 1., 1.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Lt(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Ones(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Le(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, []float64{0., math.E + 1e-10})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, []float64{0., math.E})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Le(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{1., 0.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][]float64{
			{1., 2., 3.},
			{-1., -2., -3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][]float64{
			{-1., -2., -3.},
			{1., 2., 3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Le(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{0., 0., 0.},
			{1., 1., 1.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Le(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Ones(conf, 1, 2, 3, 4)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, 4.)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Add(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 6.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, []float64{1., 2., 3.})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, []float64{4., 5., 6.})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Add(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{5., 7., 9.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][][]float64{
			{{-5.}, {1.}},
			{{-9.}, {2.}},
			{{2.}, {-1.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][][]float64{
			{{-4.}, {7.}},
			{{6.}, {2.}},
			{{-3.}, {4.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Add(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
			{{-9.}, {8.}},
			{{-3.}, {4.}},
			{{-1.}, {3.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 3, 1, 5, 1, 7)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 2, 3, 4, 1, 6, 7)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Add(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Ones(conf, 1, 2, 3, 4, 5, 6, 7)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, 4.)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Sub(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, []float64{1., 2., 3.})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, []float64{4., 5., 6.})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Sub(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{-3., -3., -3.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][][]float64{
			{{-5.}, {1.}},
			{{-9.}, {2.}},
			{{2.}, {-1.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][][]float64{
			{{-4.}, {7.}},
			{{6.}, {2.}},
			{{-3.}, {4.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Sub(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
			{{-1.}, {-6.}},
			{{-15.}, {0.}},
			{{5.}, {-5.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 3, 1, 5, 1, 7)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 1, 2, 3, 4, 1, 6, 7)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Sub(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Ones(conf, 1, 2, 3, 4, 5, 6, 7)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, 4.)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Mul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 8.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, []float64{1., 2., 3.})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, []float64{4., 5., 6.})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Mul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{4., 10., 18.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][][]float64{
			{{-5.}, {1.}},
			{{-9.}, {2.}},
			{{2.}, {-1.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][][]float64{
			{{-4.}, {7.}},
			{{6.}, {2.}},
			{{-3.}, {4.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Mul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
			{{20.}, {7.}},
			{{-54.}, {4.}},
			{{-6.}, {-4.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 3, 1, 5, 1, 7)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 1, 2, 3, 4, 1, 6, 7)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Mul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 1, 2, 3, 4, 5, 6, 7)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, 4.)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Div(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, []float64{4., 5., 6.})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, []float64{1., 2., 3.})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Div(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{4., 2.5, 2.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][][]float64{
			{{0.}, {1.}},
			{{2.}, {3.}},
			{{4.}, {5.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][][]float64{
			{{-1.}, {1.}},
			{{-2.}, {2.}},
			{{-4.}, {5.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Div(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
			{{0.}, {1.}},
			{{-1.}, {1.5}},
			{{-1.}, {1.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 3, 1, 5, 1, 7)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 2, 3, 4, 1, 6, 7)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Div(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 1, 2, 3, 4, 5, 6, 7)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, []float64{5.})
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, []float64{2.})
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Dot(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 10.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, []float64{0., 1., 2.})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, []float64{3., 4., 5.})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Dot(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, 14.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 5, 1)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 5, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Dot(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Ones(conf, 5)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 5, 4)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 5, 4)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Dot(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 4., 5)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 4, 5, 1, 7, 8)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 2, 3, 4, 1, 6, 1, 8)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Dot(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 8., 1, 2, 3, 4, 5, 6, 7)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.TensorOf(conf, [][]float64{{5.}})
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.TensorOf(conf, [][]float64{{2.}})
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, [][]float64{{10.}})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][]float64{{8.}, {4.}})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][]float64{{2.}})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{{16.}, {8.}})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][]float64{{2.}})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][]float64{{8., 4.}})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{{16., 8.}})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][]float64{
			{0., 1.},
			{2., 3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][]float64{
			{4., 5.},
			{6., 7.},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{6., 7.},
			{26., 31.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][]float64{
			{1., 1., 1.},
			{2., 2., 2.},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.TensorOf(conf, [][]float64{
			{3., 3.},
			{4., 4.},
			{5., 5.},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{12., 12.},
			{24., 24.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 5, 1, 1)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 5, 1, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Ones(conf, 5, 1, 1)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 4, 2, 1)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 4, 1, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Ones(conf, 4, 2, 1)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 4, 1, 1)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 4, 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Ones(conf, 4, 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 5, 4, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 5, 4, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 2., 5, 4, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Ones(conf, 7, 6, 1, 4, 3, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 5, 1, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.MatMul(t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 2., 7, 6, 5, 4, 3, 3)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.Zeros(conf, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Eq(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (0)")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 6, 5, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 6, 4, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Eq(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 6, 4, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Eq(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (2)")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 5, 1)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 6, 4, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Add(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 1, 5, 2, 4, 1)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 6, 4, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Add(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (2)")
		}

		/* ------------------------------ */

	})
}

func TestValidationDot(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Dot(t1)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than 1 dimension")
		}

		_, err = t1.Dot(t2)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than 1 dimension")
		}

		_, err = t2.Dot(t1)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than 1 dimension")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Dot(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at last dimension")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 8, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 5, 1, 4)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Dot(t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at last dimension")
		}

		/* -------------------- */

	})
}

func TestValidationMatMul(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.Zeros(conf, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.Zeros(conf, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.MatMul(t1)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than 2 dimensions")
		}

		_, err = t1.MatMul(t2)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than 2 dimensions")
		}

		_, err = t2.MatMul(t1)
		if err == nil {
			t.Fatalf("expected error because of tensors having less than 2 dimensions")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 3, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.MatMul(t2)
		if err == nil {
			t.Fatalf("expected error because of size incompatiblity in the last 2 dimensions for matrix multiplication")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 5, 3, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 5, 3, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.MatMul(t2)
		if err == nil {
			t.Fatalf("expected error because of size incompatiblity in the last 2 dimensions for matrix multiplication")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 6, 4, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 6, 1, 5, 4, 4)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.MatMul(t2)
		if err == nil {
			t.Fatalf("expected error because of size incompatiblity in the last 2 dimensions for matrix multiplication")
		}

		/* ------------------------------ */

	})
}
