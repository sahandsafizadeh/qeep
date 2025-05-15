libcudatensor:
	@nvcc -Xcompiler -fPIC -shared -o libcudatensor.so *.cu
	@mv libcudatensor.so ..

copylib:
	@find ../../../../ -type d -exec cp ../libcudatensor.so {} \;

cleanlib:
	@find ../../../../ -type f -name libcudatensor.so -exec rm {} \;
