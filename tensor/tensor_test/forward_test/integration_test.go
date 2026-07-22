package forward_test

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"sync"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestIntegrationAccessorsShapeModifiers(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Of([2,3,4]) values 0..23 / Slice([1,2),[],[2,4)) then Reshape([3,2]) / compacts non-contiguous slice row-major", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 1, To: 2}, {}, {From: 2, To: 4}})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Reshape([]int{3, 2})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{14., 15.},
				{18., 19.},
				{22., 23.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,4]) values 0..23 / Slice([],[1,3),[0,2)) then Transpose() / order matters", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{}, {From: 1, To: 3}, {From: 0, To: 2}})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Transpose()
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{4., 8.},
					{5., 9.},
				},
				{
					{16., 20.},
					{17., 21.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,4]) values 0..23 / Slice([1,2),[0,2),[1,3)) then UnSqueeze(0) / reshape delegate reaches fallback", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 1, To: 2}, {From: 0, To: 2}, {From: 1, To: 3}})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.UnSqueeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][][]float64{
				{
					{
						{13., 14.},
						{17., 18.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,4]) values 0..23 / Slice([1,2),[],[]) then Squeeze(0) / reshape delegate reaches fallback", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 1, To: 2}, {}, {}})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Squeeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{12., 13., 14., 15.},
				{16., 17., 18., 19.},
				{20., 21., 22., 23.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,4]) values 0..23 / Slice([],[1,3),[1,3)) then Flatten(1) / reshape delegate reaches fallback", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{}, {From: 1, To: 3}, {From: 1, To: 3}})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Flatten(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{5., 6., 9., 10.},
				{17., 18., 21., 22.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,4]) values 0..23 / Slice([1,2),[0,2),[2,4)) then Broadcast([3,2,2]) / broadcasts a sliced size-1 leading dim", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 1, To: 2}, {From: 0, To: 2}, {From: 2, To: 4}})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Broadcast([]int{3, 2, 2})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{14., 15.},
					{18., 19.},
				},
				{
					{14., 15.},
					{18., 19.},
				},
				{
					{14., 15.},
					{18., 19.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,4]) values 0..23 / Transpose() then Slice([],[1,3),[0,2)) / order matters", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Slice([]tensor.Range{{}, {From: 1, To: 3}, {From: 0, To: 2}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{1., 5.},
					{2., 6.},
				},
				{
					{13., 17.},
					{14., 18.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([3,4]) values 0..11 / Transpose() then Reshape([2,6]) / reshape reflects transposed order", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{0., 1., 2., 3.},
				{4., 5., 6., 7.},
				{8., 9., 10., 11.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Reshape([]int{2, 6})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., 4., 8., 1., 5., 9.},
				{2., 6., 10., 3., 7., 11.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([3,4]) values 0..11 / Transpose() then Flatten(0) / flatten reflects transposed order", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{0., 1., 2., 3.},
				{4., 5., 6., 7.},
				{8., 9., 10., 11.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Flatten(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{0., 4., 8., 1., 5., 9., 2., 6., 10., 3., 7., 11.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([3,4]) values 0..11 / Transpose() then Broadcast([2,4,3]) / broadcasts a transposed view over a new leading dim", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{0., 1., 2., 3.},
				{4., 5., 6., 7.},
				{8., 9., 10., 11.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Broadcast([]int{2, 4, 3})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{0., 4., 8.},
					{1., 5., 9.},
					{2., 6., 10.},
					{3., 7., 11.},
				},
				{
					{0., 4., 8.},
					{1., 5., 9.},
					{2., 6., 10.},
					{3., 7., 11.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([3]) values 1,2,3 / Broadcast([2,4,3]) then Slice([0,1),[1,3),[]) / new leading dim then slice", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{2, 4, 3})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Slice([]tensor.Range{{From: 0, To: 1}, {From: 1, To: 3}, {}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{1., 2., 3.},
					{1., 2., 3.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([1,3]) values 10,20,30 / Broadcast([4,3]) then Transpose() / transposes a broadcasted view", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{{10., 20., 30.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{4, 3})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Transpose()
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{10., 10., 10., 10.},
				{20., 20., 20., 20.},
				{30., 30., 30., 30.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([1,3]) values 10,20,30 / Broadcast([4,3]) then Reshape([12]) / numElems>len feeds compaction", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{{10., 20., 30.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{4, 3})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Reshape([]int{12})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{10., 20., 30., 10., 20., 30., 10., 20., 30., 10., 20., 30.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([4,5]) / Slice then Reshape then Transpose then UnSqueeze then Slice / alternating reshape and copy-forcing ops", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{0., 1., 2., 3., 4.},
				{5., 6., 7., 8., 9.},
				{10., 11., 12., 13., 14.},
				{15., 16., 17., 18., 19.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 1, To: 4}, {From: 1, To: 4}})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Reshape([]int{3, 3})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.UnSqueeze(1)
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Slice([]tensor.Range{{}, {}, {From: 0, To: 2}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{{6., 11.}},
				{{7., 12.}},
				{{8., 13.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,4]) / Slice then Transpose then Reshape then UnSqueeze then Squeeze / round-trips through the compaction path", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{}, {From: 1, To: 3}, {From: 1, To: 3}})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Reshape([]int{2, 4})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.UnSqueeze(0)
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Squeeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{5., 9., 6., 10.},
				{17., 21., 18., 22.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,4]) / Transpose then Slice then Squeeze then Reshape then Transpose / distinct ordering", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Slice([]tensor.Range{{}, {From: 0, To: 2}, {From: 0, To: 1}})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Squeeze(2)
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Reshape([]int{2, 2})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Transpose()
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., 12.},
				{1., 13.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([3,4]) values 0..11 / Transpose then Reshape([1,12]) then Broadcast([3,12]) then Slice([0,2),[2,5)) / broadcasts a reshaped transpose then slices", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{0., 1., 2., 3.},
				{4., 5., 6., 7.},
				{8., 9., 10., 11.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Reshape([]int{1, 12})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Broadcast([]int{3, 12})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Slice([]tensor.Range{{From: 0, To: 2}, {From: 2, To: 5}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{8., 1., 5.},
				{8., 1., 5.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,4]) / Transpose then Reshape then Transpose then Reshape / hits compact on an already-compacted intermediate", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Reshape([]int{4, 6})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Reshape([]int{3, 8})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., 2., 12., 14., 4., 6., 16., 18.},
				{8., 10., 20., 22., 1., 3., 13., 15.},
				{5., 7., 17., 19., 9., 11., 21., 23.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([3,4]) values 0..11 / Transpose then Reshape([2,6]) then Transpose then Broadcast([2,6,2]) / broadcasts a doubly-transposed reshape", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{0., 1., 2., 3.},
				{4., 5., 6., 7.},
				{8., 9., 10., 11.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Reshape([]int{2, 6})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Broadcast([]int{2, 6, 2})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{0., 2.},
					{4., 6.},
					{8., 10.},
					{1., 3.},
					{5., 7.},
					{9., 11.},
				},
				{
					{0., 2.},
					{4., 6.},
					{8., 10.},
					{1., 3.},
					{5., 7.},
					{9., 11.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([1,3]) values 7,8,9 / Broadcast([4,3]) then Transpose() then Reshape([12]) / zero-stride survives two views", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{{7., 8., 9.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{4, 3})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Reshape([]int{12})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{7., 7., 7., 7., 8., 8., 8., 8., 9., 9., 9., 9.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,4]) / Broadcast then Slice then Transpose then Flatten / broadcast feeds a long non-contiguous chain", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{2, 2, 3, 4})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Slice([]tensor.Range{{From: 1, To: 2}, {}, {From: 1, To: 3}, {From: 0, To: 2}})
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			act, err = act.Flatten(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{4., 8., 5., 9., 16., 20., 17., 21.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})
	})
}

// TestDataSharingConcurrency stresses the interaction between the CUDA backend's
// shareCUDATensorData (increments a shared buffer's ref-count when a view is
// created) and freeCUDATensorData (decrements it on GC, releasing device memory
// only at zero). Many goroutines create and drop zero-copy views of the same
// buffer while the GC reclaims collected tensors. The invariants asserted
// throughout — regardless of non-deterministic cleanup timing — are:
//   - a shared buffer is never freed while a tensor still references it
//     (surviving tensors keep returning correct data), and
//   - a buffer is never freed twice (the run completes without the
//     "reached negative value for cudaAllocMem" panic from a double free).
//
// Bases are Full(sentinel) so every element of any view equals a known value;
// validation samples the all-zeros element, which detects both a premature free
// (value changes when the device buffer is reused) and gross corruption. The big
// pressure tensors are sized to push CUDA allocation past allocatedMemThreshold
// (0.75), tripping the internal runtime.GC() and exercising freeCUDATensorData
// concurrently with shareCUDATensorData.
//
// The test runs on CPU (via RunTestLogicOnDevices, best paired with -race) and,
// when built with -tags cuda, on the actual shared-buffer/ref-count path.
func TestDataSharingConcurrency(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		const (
			baseDim   = 32      // square base so Transpose/Reshape stay valid
			bigLen    = 1 << 20 // ~8 MB per pressure tensor
			sentinelA = 3.5
		)

		cfg := &tensor.Config{Device: dev}

		// forceGC runs cleanups: AddCleanup callbacks fire asynchronously after a
		// collection, so we GC twice with a yield between to give them a chance to
		// run before we assert.
		forceGC := func() {
			runtime.GC()
			runtime.Gosched()
			runtime.GC()
		}

		// validateSentinel samples the all-zeros element of a view; for a
		// Full(want) base every element (through any view) must equal want.
		validateSentinel := func(v tensor.Tensor, want float64) error {
			idx := make([]int, len(v.Shape()))
			got, err := v.At(idx...)
			if err != nil {
				return fmt.Errorf("reading element: %w", err)
			}
			if got != want {
				return fmt.Errorf("expected element %v, got %v (buffer freed or corrupted)", want, got)
			}
			return nil
		}

		// makeView returns a zero-copy view of a square [baseDim, baseDim] base,
		// picking one of four sharing ops. All four share the base buffer on CUDA
		// (the base is contiguous), so each increments the ref-count.
		makeView := func(base tensor.Tensor, which int) (tensor.Tensor, error) {
			switch which % 4 {
			case 0:
				return base.Slice([]tensor.Range{{From: 0, To: 1}, {}})
			case 1:
				return base.Transpose()
			case 2:
				return base.Broadcast([]int{2, baseDim, baseDim})
			default:
				return base.Reshape([]int{baseDim * baseDim})
			}
		}

		// ===== scenario 1: single base, many concurrent readers =====

		t.Run("single base shared by many concurrent readers / views dropped, base survives GC", func(t *testing.T) {
			const workers = 128

			base, err := tensor.Full([]int{baseDim, baseDim}, sentinelA, cfg)
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func(w int) {
					defer wg.Done()

					view, err := makeView(base, w)
					if err != nil {
						t.Errorf("worker %d: making view: %v", w, err)
						return
					}
					if err := validateSentinel(view, sentinelA); err != nil {
						t.Errorf("worker %d: %v", w, err)
					}
					// view goes out of scope here and becomes collectible
				}(w)
			}
			wg.Wait()
			forceGC()

			if err := validateSentinel(base, sentinelA); err != nil {
				t.Fatalf("base freed/corrupted while still referenced: %v", err)
			}
			runtime.KeepAlive(base)
		})

		// ===== scenario 2: concurrent shares under GC pressure =====

		t.Run("concurrent shares while allocating pressure tensors / internal GC must not free live base", func(t *testing.T) {
			const (
				workers = 64
				rounds  = 8
			)

			base, err := tensor.Full([]int{baseDim, baseDim}, sentinelA, cfg)
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func(w int) {
					defer wg.Done()

					for r := 0; r < rounds; r++ {
						view, err := makeView(base, w+r)
						if err != nil {
							t.Errorf("worker %d round %d: making view: %v", w, r, err)
							return
						}
						if err := validateSentinel(view, sentinelA); err != nil {
							t.Errorf("worker %d round %d: %v", w, r, err)
							return
						}

						// Allocate a big tensor and immediately drop it so it is
						// eligible for the internal threshold-triggered GC.
						if _, err := tensor.Full([]int{bigLen}, 0., cfg); err != nil {
							t.Errorf("worker %d round %d: pressure alloc: %v", w, r, err)
							return
						}
						runtime.KeepAlive(view)
					}
				}(w)
			}
			wg.Wait()
			forceGC()

			if err := validateSentinel(base, sentinelA); err != nil {
				t.Fatalf("base freed/corrupted under GC pressure: %v", err)
			}
			runtime.KeepAlive(base)
		})

		// ===== scenario 3: chained views mixing the share and reshape-copy paths =====

		t.Run("chained views per goroutine / mixes share path and reshape-copy path", func(t *testing.T) {
			const workers = 64

			base, err := tensor.Full([]int{baseDim, baseDim}, sentinelA, cfg)
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func(w int) {
					defer wg.Done()

					// Slice -> Transpose -> Broadcast are all shares of the base
					// buffer; the final Reshape of a non-contiguous view cannot
					// share and falls back to a fresh copy (newCUDATensor).
					v1, err := base.Slice([]tensor.Range{{From: 0, To: baseDim / 2}, {}})
					if err != nil {
						t.Errorf("worker %d: slice: %v", w, err)
						return
					}
					v2, err := v1.Transpose()
					if err != nil {
						t.Errorf("worker %d: transpose: %v", w, err)
						return
					}
					v3, err := v2.Broadcast([]int{2, baseDim, baseDim / 2})
					if err != nil {
						t.Errorf("worker %d: broadcast: %v", w, err)
						return
					}
					v4, err := v3.Reshape([]int{baseDim * baseDim})
					if err != nil {
						t.Errorf("worker %d: reshape: %v", w, err)
						return
					}

					for _, v := range []tensor.Tensor{v1, v2, v3, v4} {
						if err := validateSentinel(v, sentinelA); err != nil {
							t.Errorf("worker %d: %v", w, err)
							return
						}
					}
				}(w)
			}
			wg.Wait()
			forceGC()

			if err := validateSentinel(base, sentinelA); err != nil {
				t.Fatalf("base freed/corrupted after chained views: %v", err)
			}
			runtime.KeepAlive(base)
		})

		// ===== scenario 4: base dropped while views survive =====

		t.Run("base dropped while views survive in goroutines / buffer survives via view refs", func(t *testing.T) {
			const workers = 64

			base, err := tensor.Full([]int{baseDim, baseDim}, sentinelA, cfg)
			if err != nil {
				t.Fatal(err)
			}

			views := make([]tensor.Tensor, workers)
			for w := 0; w < workers; w++ {
				v, err := makeView(base, w)
				if err != nil {
					t.Fatal(err)
				}
				views[w] = v
			}

			// Drop the only direct reference to the base. Its buffer must survive
			// because the views still hold references (ref-count > 0).
			base = nil
			forceGC()

			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func(w int) {
					defer wg.Done()
					if err := validateSentinel(views[w], sentinelA); err != nil {
						t.Errorf("view %d after base dropped: %v", w, err)
					}
				}(w)
			}
			wg.Wait()

			// Now drop every view and GC: the buffer must be freed exactly once
			// (no negative-cudaAllocMem panic from a double free).
			for w := range views {
				views[w] = nil
			}
			forceGC()

			_ = base
		})

		// ===== scenario 5: randomized churn with canary tensors =====

		t.Run("randomized churn across goroutines / canaries stay intact", func(t *testing.T) {
			const (
				canaryCount = 6
				workers     = 32
				iters       = 40
			)

			canaries := make([]tensor.Tensor, canaryCount)
			sentinels := make([]float64, canaryCount)
			for i := range canaries {
				// Distinct sentinel per canary so a wrongly-freed-and-reused
				// buffer is detected as a value mismatch.
				sentinels[i] = float64(i)*1000. + 7.
				c, err := tensor.Full([]int{baseDim, baseDim}, sentinels[i], cfg)
				if err != nil {
					t.Fatal(err)
				}
				canaries[i] = c
			}

			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func(w int) {
					defer wg.Done()

					for it := 0; it < iters; it++ {
						i := rand.IntN(canaryCount)

						view, err := makeView(canaries[i], rand.IntN(4))
						if err != nil {
							t.Errorf("worker %d iter %d: making view: %v", w, it, err)
							return
						}
						if err := validateSentinel(view, sentinels[i]); err != nil {
							t.Errorf("worker %d iter %d canary %d: %v", w, it, i, err)
							return
						}

						if rand.IntN(4) == 0 {
							if _, err := tensor.Full([]int{bigLen}, 0., cfg); err != nil {
								t.Errorf("worker %d iter %d: pressure alloc: %v", w, it, err)
								return
							}
						}
						if rand.IntN(8) == 0 {
							runtime.GC()
						}
					}
				}(w)
			}
			wg.Wait()
			forceGC()

			for i := range canaries {
				if err := validateSentinel(canaries[i], sentinels[i]); err != nil {
					t.Fatalf("canary %d freed/corrupted: %v", i, err)
				}
				runtime.KeepAlive(canaries[i])
			}
		})

		// ===== scenario 6: cross-goroutine ownership handoff =====

		t.Run("cross-goroutine ownership handoff / shared on producer, freed on consumer", func(t *testing.T) {
			const (
				producers   = 8
				consumers   = 8
				perProducer = 32
			)

			base, err := tensor.Full([]int{baseDim, baseDim}, sentinelA, cfg)
			if err != nil {
				t.Fatal(err)
			}

			ch := make(chan tensor.Tensor, 64)

			var pwg sync.WaitGroup
			for p := 0; p < producers; p++ {
				pwg.Add(1)
				go func(p int) {
					defer pwg.Done()
					for i := 0; i < perProducer; i++ {
						v, err := makeView(base, p+i)
						if err != nil {
							t.Errorf("producer %d: making view: %v", p, err)
							return
						}
						ch <- v
					}
				}(p)
			}
			go func() {
				pwg.Wait()
				close(ch)
			}()

			var cwg sync.WaitGroup
			for c := 0; c < consumers; c++ {
				cwg.Add(1)
				go func() {
					defer cwg.Done()
					// A view is shared on a producer goroutine but dropped (and
					// thus freed via cleanup) on this consumer goroutine.
					for v := range ch {
						if err := validateSentinel(v, sentinelA); err != nil {
							t.Errorf("consumer: %v", err)
						}
						if rand.IntN(4) == 0 {
							runtime.GC()
						}
					}
				}()
			}
			cwg.Wait()
			forceGC()

			if err := validateSentinel(base, sentinelA); err != nil {
				t.Fatalf("base freed/corrupted after ownership handoff: %v", err)
			}
			runtime.KeepAlive(base)
		})
	})
}
