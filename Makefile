QEEP_CUDA_LIB_FILE := libcudatensor.so
QEEP_CUDA_SRC_PATH := ./tensor/internal/cudatensor/cuda_c
QEEP_CUDA_LIB_PATH := ./tensor/internal/cudatensor/lib

cuda:
	@nvcc -Xcompiler -fPIC -shared -o $(QEEP_CUDA_LIB_FILE) $(QEEP_CUDA_SRC_PATH)/*.cu
	@mkdir -p $(QEEP_CUDA_LIB_PATH)
	@mv $(QEEP_CUDA_LIB_FILE) $(QEEP_CUDA_LIB_PATH)

clean:
	@rm -rf $(QEEP_CUDA_LIB_PATH)