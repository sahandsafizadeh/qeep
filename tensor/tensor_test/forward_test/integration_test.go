package forward_test

import (
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
