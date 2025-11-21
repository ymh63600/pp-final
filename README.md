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
* `mpi`   — MPI 平行化版本
* `cuda`   — CUDA 平行化版本
* `compare`  — CSV 檔案比對工具

---

## 🚀 Run

### 1. Serial 版本

不需提供參數。

```bash
run -- ./serial
```

---

### 2. SIMD 版本

不需提供參數。

```bash
run -- ./simd
```

---

### 3. Pthread 版本（可指定 thread 數）

語法：

```bash
run -c <thread-num> -- ./pthread <thread-num>
```

例如指定 6 threads：

```bash
run -c 6 -- ./pthread 6
```

若不帶參數，預設使用 **4 threads**：

```bash
run -c 4 -- ./pthread
```

---

### 4. OpenMP 版本（可指定 thread 數）

語法：

```bash
run -c <thread-num> -- ./openmp <thread-num>
```

例如：

```bash
run -c 8 -- ./openmp 8
```

若不帶參數，預設使用 **4 threads**：

```bash
run -c 4 -- ./openmp
```

### 5. MPI 版本

語法：

```bash
run --mpi=pmix -N <nodes> -n <processes> -- ./mpi
```

例如：

```bash
run --mpi=pmix -N 2 -n 4 -- ./mpi
```

### 6. CUDA 版本

語法：

```bash
run -- ./cuda
```

### 7. python scikit-learn 版本

語法：

```bash
run -- python3 sklearn_tfidf.py
```

---

### 8. Compare CSV 版本

用來比對兩個 CSV 檔案（TF-IDF 結果）。

#### 8.1 直接執行

```bash
run ./compare file1.csv file2.csv
```

#### 8.2 使用 Makefile 動態傳參

```bash
make run_compare CSV1=file1.csv CSV2=file2.csv
```

* `CSV1`、`CSV2` 為要比對的兩個 CSV 檔案
* 若檔案內容相同），會輸出：

```
The CSV files are identical.
```

