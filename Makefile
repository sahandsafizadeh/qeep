# -------------------- DEV --------------------
QEEP_BUILD_TAGS := cuda

fmt:
	@./scripts/gofmt.sh

lint:
	@./scripts/golint.sh $(QEEP_BUILD_TAGS)

gosec:
	@./scripts/gosec.sh $(QEEP_BUILD_TAGS)

cover:
	@./scripts/gotest.sh $(QEEP_BUILD_TAGS)

install_linters:
	@./scripts/install_linters.sh

# -------------------- CUDA --------------------
QEEP_CUDA_LIB_FILE := libcudatensor.so
QEEP_CUDA_SRC_PATH := ./tensor/internal/cudatensor/cuda_c
QEEP_CUDA_LIB_PATH := ./tensor/internal/cudatensor/lib

cuda:
	@nvcc -Xcompiler -fPIC -shared -o $(QEEP_CUDA_LIB_FILE) $(QEEP_CUDA_SRC_PATH)/*.cu
	@mkdir -p $(QEEP_CUDA_LIB_PATH)
	@mv $(QEEP_CUDA_LIB_FILE) $(QEEP_CUDA_LIB_PATH)

clean:
	@rm -rf $(QEEP_CUDA_LIB_PATH)