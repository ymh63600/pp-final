# TF-IDF Parallelization (Serial / Pthread / OpenMP)

本專案示範如何使用 **Serial、Pthread、OpenMP** 實作加速版的 TF-IDF，並比較不同並行方法的效能。

---

## 📦 Build

使用 Makefile 進行編譯：

```bash
make
```

成功後會生成以下可執行檔：

* `serial`   — 單執行緒版本
* `pthread`  — Pthread 平行化版本
* `openmp`   — OpenMP 平行化版本

---

## 🚀 Run

### 1. Serial 版本

不需提供參數。

```bash
./serial
```

---

### 2. Pthread 版本（可指定 thread 數）

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

### 3. OpenMP 版本（可指定 thread 數）

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