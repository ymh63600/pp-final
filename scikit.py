# serial_sklearn_tfidf.py
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

    # ---------------------------
    # Phase 1: Load documents
    # ---------------------------
    t0 = time.perf_counter()
    documents, paths = load_20newsgroups(dataset_path)
    t1 = time.perf_counter()

    print("--- Serial-like TF-IDF Timing Report (sklearn built-in) ---")
    print(f"Total Documents Loaded: {len(documents)}")

    if len(documents) == 0:
        print(f'Error: No documents found. Please check path "{dataset_path}"')
        return

    # ---------------------------
    # Phase 2: Tokenization + Count (CountVectorizer)
    # Match C++ tokenize(): split by whitespace, keep original case/punctuation
    # ---------------------------
    t2 = time.perf_counter()
    count_vec = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False
    )
    X_counts = count_vec.fit_transform(documents)  # (N_docs, V)
    t3 = time.perf_counter()

    N_docs, V = X_counts.shape
    feature_names = count_vec.get_feature_names_out()

    # ---------------------------
    # Phase 3: IDF (TfidfTransformer.fit)
    # IDF = log(N / df), no smoothing, no norm
    # ---------------------------
    tfidf_tr = TfidfTransformer(
        use_idf=True,
        smooth_idf=False,
        norm=None,
        sublinear_tf=False
    )

    t4 = time.perf_counter()
    tfidf_tr.fit(X_counts)  # compute idf only
    t5 = time.perf_counter()

    # ---------------------------
    # Phase 4: TF time (manual, for apples-to-apples timing)
    # TF = count / total_words_in_doc
    # Note: final tfidf still uses sklearn transformer
    # ---------------------------
    t6 = time.perf_counter()
    doc_len = X_counts.sum(axis=1).A1.astype(np.float64)
    doc_len[doc_len == 0] = 1.0
    inv_doc_len = 1.0 / doc_len
    _X_tf = X_counts.multiply(inv_doc_len[:, None])
    t7 = time.perf_counter()

    # ---------------------------
    # Phase 5: TF-IDF (TfidfTransformer.transform)
    # ---------------------------
    t8 = time.perf_counter()
    X_tfidf = tfidf_tr.transform(X_counts)
    t9 = time.perf_counter()

    print(f"Document Loading Time: {t1 - t0:.6f} seconds")
    print(f"Tokenization and CountVectorizer Time: {t3 - t2:.6f} seconds")
    print(f"Compute IDF Time (Transformer.fit): {t5 - t4:.6f} seconds (Vocabulary Size: {V})")
    print(f"Compute TF Time (manual, from counts): {t7 - t6:.6f} seconds")
    print(f"Compute All TF-IDFs Time (Transformer.transform): {t9 - t8:.6f} seconds")

    total_time = (t1 - t0) + (t3 - t2) + (t5 - t4) + (t7 - t6) + (t9 - t8)
    print("------------------------------------------")
    print(f"Total Execution Time (including load): {total_time:.6f} seconds")
    print("------------------------------------------")

    # ---------------------------
    # Phase 6: Save to CSV (same format as C++)
    # ---------------------------
    out_path = "serial.csv"
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write("document_id,word,tfidf_value\n")
        X_csr = X_tfidf.tocsr()
        for i in range(N_docs):
            row_start = X_csr.indptr[i]
            row_end = X_csr.indptr[i + 1]
            indices = X_csr.indices[row_start:row_end]
            data = X_csr.data[row_start:row_end]
            for j, val in zip(indices, data):
                fout.write(f"{i},{feature_names[j]},{val}\n")

    print(f"TF-IDF saved to {out_path}")


if __name__ == "__main__":
    main()
