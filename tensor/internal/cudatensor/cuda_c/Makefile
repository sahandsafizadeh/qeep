libcudatensor:
	@nvcc -Xcompiler -fPIC -shared -o libcudatensor.so *.cu
	@mv libcudatensor.so ..
	@cp ../libcudatensor.so ../../../tensor_test/forward_test