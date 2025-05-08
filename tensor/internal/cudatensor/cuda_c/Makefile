libcudatensor:
	@nvcc -Xcompiler -fPIC -shared -o libcudatensor.so *.cu
	@mv libcudatensor.so ..
	@cp ../libcudatensor.so ../../../tensor_test/forward_test
	@cp ../libcudatensor.so ../../../tensor_test/backward_test