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

TARGET_SIMD = simd
SRC_SIMD = simd.cpp

TARGET_COMPARE = compare
SRC_COMPARE = compare.cpp

# ----------------------------------------------------
# 指令
# ----------------------------------------------------
.PHONY: all clean run_serial run_pthread run_openmp run_simd run_compare

# all: 編譯所有目標
all: $(TARGET_SERIAL) $(TARGET_PTHREAD) $(TARGET_OPENMP) $(TARGET_SIMD) $(TARGET_COMPARE)

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
	rm -f $(TARGET_SERIAL) $(TARGET_PTHREAD) $(TARGET_OPENMP) $(TARGET_SIMD) $(TARGET_COMPARE) *.o
