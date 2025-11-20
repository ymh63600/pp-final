# ----------------------------------------------------
# 核心變數
# ----------------------------------------------------
CXX = g++
# 基礎編譯標誌
CXXFLAGS = -std=c++17 -Wall -Wextra -O3
# SIMD 特有的編譯標誌
SIMD_FLAGS = -mavx2

NVCC = nvcc
NVCCFLAGS = -std=c++17 -O3

MPICXX = mpicxx
MPICXXFLAGS = -std=c++17 -Wall -Wextra -O3

# ----------------------------------------------------
# 目標檔案
# ----------------------------------------------------
TARGET_SERIAL = serial
SRC_SERIAL = serial.cpp

TARGET_PTHREAD = pthread
SRC_PTHREAD = pthread.cpp

TARGET_OPENMP = openmp
SRC_OPENMP = openmp.cpp

TARGET_SIMD = simd
SRC_SIMD = simd.cpp

TARGET_COMPARE = compare
SRC_COMPARE = compare.cpp

TARGET_CUDA    = cuda
SRC_CUDA       = cuda.cu

TARGET_MPI     = mpi
SRC_MPI        = mpi.cpp

# ----------------------------------------------------
# 指令
# ----------------------------------------------------
.PHONY: all clean run_serial run_pthread run_openmp run_simd run_compare run_cuda run_mpi

# all: 編譯所有目標
all: $(TARGET_SERIAL) $(TARGET_PTHREAD) $(TARGET_OPENMP) $(TARGET_SIMD) $(TARGET_COMPARE) $(TARGET_CUDA) $(TARGET_MPI)

# ---------------- Serial ----------------
$(TARGET_SERIAL): $(SRC_SERIAL)
	$(CXX) $(CXXFLAGS) $< -o $@

run_serial: $(TARGET_SERIAL)
	@echo "--- Running $(TARGET_SERIAL) ---"
	run -- ./$(TARGET_SERIAL)

# ---------------- Pthread ----------------
$(TARGET_PTHREAD): $(SRC_PTHREAD)
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread

run_pthread: $(TARGET_PTHREAD)
	@echo "--- Running $(TARGET_PTHREAD) ---"
	run -c 4 -- ./$(TARGET_PTHREAD)

# ---------------- OpenMP ----------------
$(TARGET_OPENMP): $(SRC_OPENMP)
	$(CXX) $(CXXFLAGS) $< -o $@ -fopenmp

run_openmp: $(TARGET_OPENMP)
	@echo "--- Running $(TARGET_OPENMP) ---"
	run -c 4 -- ./$(TARGET_OPENMP)

# ---------------- SIMD ----------------
$(TARGET_SIMD): $(SRC_SIMD)
	$(CXX) $(CXXFLAGS) $(SIMD_FLAGS) $< -o $@

run_simd: $(TARGET_SIMD)
	@echo "--- Running $(TARGET_SIMD) ---"
	run -- ./$(TARGET_SIMD)

# ---------------- CUDA ----------------
$(TARGET_CUDA): $(SRC_CUDA)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run_cuda: $(TARGET_CUDA)
	@echo "--- Running $(TARGET_CUDA) ---"
	run -- ./$(TARGET_CUDA)

# ---------------- MPI ----------------
$(TARGET_MPI): $(SRC_MPI)
	$(MPICXX) $(MPICXXFLAGS) $< -o $@

# 用 NP 指定 process 數，例如：make run_mpi NP=8
run_mpi: $(TARGET_MPI)
	@echo "--- Running $(TARGET_MPI) ---"
	run --mpi=pmix -N 4 -n 4 -- ./$(TARGET_MPI)

# ---------------- Compare CSV ----------------
$(TARGET_COMPARE): $(SRC_COMPARE)
	$(CXX) $(CXXFLAGS) $< -o $@

# 動態傳入 CSV 參數
run_compare:
	@echo "--- Running $(TARGET_COMPARE) ---"
	@if [ "$$CSV1" = "" ] || [ "$$CSV2" = "" ]; then \
	    echo "Usage: make run_compare CSV1=file1.csv CSV2=file2.csv"; \
	else \
	    ./$(TARGET_COMPARE) $$CSV1 $$CSV2; \
	fi

# ---------------- Clean ----------------
clean:
	@echo "Cleaning up generated files..."
	rm -f $(TARGET_SERIAL) $(TARGET_PTHREAD) $(TARGET_OPENMP) $(TARGET_SIMD) $(TARGET_COMPARE) $(TARGET_CUDA) $(TARGET_MPI) *.o
