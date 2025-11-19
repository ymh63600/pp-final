# ----------------------------------------------------
# 核心變數
# ----------------------------------------------------
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3

# ----------------------------------------------------
# 目標檔案
# ----------------------------------------------------
TARGET_SERIAL = serial
SRC_SERIAL = serial.cpp

TARGET_PTHREAD = pthread
SRC_PTHREAD = pthread.cpp

TARGET_OPENMP = openmp
SRC_OPENMP = openmp.cpp

# ----------------------------------------------------
# 指令
# ----------------------------------------------------
.PHONY: all clean run_serial run_pthread run_openmp

all: $(TARGET_SERIAL) $(TARGET_PTHREAD) $(TARGET_OPENMP)

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

# ---------------- Clean ----------------
clean:
	@echo "Cleaning up generated files..."
	rm -f $(TARGET_SERIAL) $(TARGET_PTHREAD) $(TARGET_OPENMP) *.o
