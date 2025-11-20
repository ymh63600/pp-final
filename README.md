# TF-IDF Parallelization (Serial / SIMD/ Pthread / OpenMP)

本專案示範如何使用 **Serial、SIMD、Pthread、OpenMP** 實作加速版的 TF-IDF，並比較不同並行方法的效能。


## 📦 Build

使用 Makefile 進行編譯：

```bash
make
```

成功後會生成以下可執行檔：

* `serial`   — 單執行緒版本
* `simd`     — SIMD 向量化版本 (需要支援 AVX2 的 CPU)
* `pthread`  — Pthread 平行化版本
* `openmp`   — OpenMP 平行化版本
* `compare`  — CSV 檔案比對工具

---

## 🚀 Run

### 1. Serial 版本

不需提供參數。

```bash
./serial
```

---

### 2. SIMD 版本

不需提供參數。

```bash
./simd
```

---

### 3. Pthread 版本（可指定 thread 數）

語法：

```bash
./pthread <thread-num>
```

例如指定 6 threads：

```bash
./pthread 6
```

若不帶參數，預設使用 **8 threads**：

```bash
./pthread
```

---

### 4. OpenMP 版本（可指定 thread 數）

語法：

```bash
./openmp <thread-num>
```

例如：

```bash
./openmp 8
```

若不帶參數，預設使用 **8 threads**：

```bash
./openmp
```

---

### 5. Compare CSV 版本

用來比對兩個 CSV 檔案（TF-IDF 結果）。

#### 5.1 直接執行

```bash
./compare file1.csv file2.csv
```

#### 5.2 使用 Makefile 動態傳參

```bash
make run_compare CSV1=file1.csv CSV2=file2.csv
```

* `CSV1`、`CSV2` 為要比對的兩個 CSV 檔案
* 若檔案內容相同），會輸出：

```
The CSV files are identical.
```

