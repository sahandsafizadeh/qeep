libgoutil:
	go build -buildmode=c-shared -o goutil.so goutil.go

libcudatensor:
	nvcc -Xcompiler -fPIC -shared -o libcudatensor.so *.cu
	mv libcudatensor.so ..