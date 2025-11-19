# ----------------------------------------------------
# 核心變數
# ----------------------------------------------------
CXX = g++
# 必須使用 C++17 或更高版本來支援 std::filesystem 和 chrono
CXXFLAGS = -std=c++17 -Wall -Wextra -O3
TARGET_BIN = serial
SRC_FILE = serial.cpp

# ----------------------------------------------------
# 核心指令
# ----------------------------------------------------

.PHONY: all clean run

# 預設目標：編譯執行檔
all: $(TARGET_BIN)

# 編譯規則
$(TARGET_BIN): $(SRC_FILE)
	$(CXX) $(CXXFLAGS) $< -o $@

# 執行指令
run: $(TARGET_BIN)
	@echo "--- Running $(TARGET_BIN) ---"
	./$(TARGET_BIN)

# 清理指令
clean:
	@echo "Cleaning up generated files..."
	rm -f $(TARGET_BIN) *.o
