CUDA_LIB_FILE := libcudatensor.so
CUDA_SRC_PATH := ./tensor/internal/cudatensor/cuda_c

cuda:
	@nvcc -Xcompiler -fPIC -shared -o $(CUDA_LIB_FILE) $(CUDA_SRC_PATH)/*.cu
	@mkdir -p $(QEEP_LIB_PATH)
	@mv $(CUDA_LIB_FILE) $(QEEP_LIB_PATH)

clean:
	@rm -r $(QEEP_LIB_PATH)