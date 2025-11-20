# ----------------------------------------------------
# 核心變數
# ----------------------------------------------------
CXX = g++
# 基礎編譯標誌
CXXFLAGS = -std=c++17 -Wall -Wextra -O3
# SIMD 特有的編譯標誌
SIMD_FLAGS = -mavx2

# ----------------------------------------------------
# 目標檔案
# ----------------------------------------------------
TARGET_SERIAL = serial
SRC_SERIAL = serial.cpp

TARGET_PTHREAD = pthread
SRC_PTHREAD = pthread.cpp

TARGET_OPENMP = openmp
SRC_OPENMP = openmp.cpp

# 新增 SIMD 變數
TARGET_SIMD = simd
SRC_SIMD = simd.cpp


# ----------------------------------------------------
# 指令
# ----------------------------------------------------
.PHONY: all clean run_serial run_pthread run_openmp run_simd

# all: 編譯所有目標
all: $(TARGET_SERIAL) $(TARGET_PTHREAD) $(TARGET_OPENMP) $(TARGET_SIMD)

# ---------------- Serial ----------------
$(TARGET_SERIAL): $(SRC_SERIAL)
	$(CXX) $(CXXFLAGS) $< -o $@

run_serial: $(TARGET_SERIAL)
	@echo "--- Running $(TARGET_SERIAL) ---"
	./$(TARGET_SERIAL)

# ---------------- Pthread ----------------
$(TARGET_PTHREAD): $(SRC_PTHREAD)
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread

run_pthread: $(TARGET_PTHREAD)
	@echo "--- Running $(TARGET_PTHREAD) ---"
	./$(TARGET_PTHREAD)

# ---------------- OpenMP ----------------
$(TARGET_OPENMP): $(SRC_OPENMP)
	$(CXX) $(CXXFLAGS) $< -o $@ -fopenmp

run_openmp: $(TARGET_OPENMP)
	@echo "--- Running $(TARGET_OPENMP) ---"
	./$(TARGET_OPENMP)

# ---------------- SIMD ----------------
$(TARGET_SIMD): $(SRC_SIMD)
	$(CXX) $(CXXFLAGS) $(SIMD_FLAGS) $< -o $@

run_simd: $(TARGET_SIMD)
	@echo "--- Running $(TARGET_SIMD) ---"
	./$(TARGET_SIMD)

# ---------------- Clean ----------------
clean:
	@echo "Cleaning up generated files..."
	rm -f $(TARGET_SERIAL) $(TARGET_PTHREAD) $(TARGET_OPENMP) $(TARGET_SIMD) *.o
