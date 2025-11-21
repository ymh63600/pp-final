# scikit_transformer_serial_like.py
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_20newsgroups(root_dir):
    documents = []
    paths = []
    root = Path(root_dir)
    for p in root.rglob("*.txt"):
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            documents.append(text)
            paths.append(str(p))
    return documents, paths


def main():
    dataset_path = "dataset"

    t0 = time.perf_counter()
    documents, paths = load_20newsgroups(dataset_path)
    t1 = time.perf_counter()

    print("--- Serial-like TF-IDF Timing Report (sklearn TfidfTransformer) ---")
    print(f"Total Documents Loaded: {len(documents)}")
    print(f"Document Loading Time: {t1 - t0:.6f} seconds")

    if len(documents) == 0:
        print(f'Error: No documents found. Please check path "{dataset_path}"')
        return

    # Tokenization and vocabulary building
    # Match serial.cpp tokenize(): split by whitespace, keep original case/punctuation
    t2 = time.perf_counter()
    vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False
    )
    X_counts = vectorizer.fit_transform(documents)  # CSR, shape (N_docs, V_raw)
    feature_names = vectorizer.get_feature_names_out()
    t3 = time.perf_counter()
    print(f"Tokenization and CountVectorizer Time: {t3 - t2:.6f} seconds")

    N_docs, V_raw = X_counts.shape

    # Force lexicographic (dictionary) order like std::map for stable output
    t4 = time.perf_counter()
    sort_idx = np.argsort(feature_names)
    feature_names = feature_names[sort_idx]

    # Column reorder: CSR -> CSC -> reorder -> CSR
    X_csc = X_counts.tocsc()
    X_csc = X_csc[:, sort_idx]
    X_counts = X_csc.tocsr()
    t5 = time.perf_counter()
    print(f"Sort vocab and reorder matrix Time: {t5 - t4:.6f} seconds (Vocabulary Size: {V_raw})")

    # TF-IDF using sklearn TfidfTransformer
    # Important settings:
    # norm=None: no L1/L2 normalization (serial.cpp has none)
    # use_idf=True: use IDF
    # smooth_idf=False: do NOT add 1 to df, but sklearn still adds +1 after log
    # sublinear_tf=False: TF = raw count / doc_len (handled internally)
    t6 = time.perf_counter()
    transformer = TfidfTransformer(
        norm=None,
        use_idf=True,
        smooth_idf=False,
        sublinear_tf=False
    )
    X_tfidf = transformer.fit_transform(X_counts)
    t7 = time.perf_counter()
    print(f"TfidfTransformer fit_transform Time: {t7 - t6:.6f} seconds")

    total_time = t7 - t0
    print("------------------------------------------")
    print(f"Total Execution Time (including load): {total_time:.6f} seconds")
    print("------------------------------------------")

    # Output serial.csv with same format as C++
    out_path = "serial.csv"
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write("document_id,word,tfidf_value\n")
        X_csr = X_tfidf.tocsr()
        for d in range(N_docs):
            row_start = X_csr.indptr[d]
            row_end = X_csr.indptr[d + 1]
            indices = X_csr.indices[row_start:row_end]
            data = X_csr.data[row_start:row_end]
            for j, val in zip(indices, data):
                fout.write(f"{d},{feature_names[j]},{val}\n")

    print(f"TF-IDF saved to {out_path}")


if __name__ == "__main__":
    main()
