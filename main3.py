#!/usr/bin/env python3
"""
Hindi Fake News Detection (CONSTRAINT-2021, Hindi subset)
End-to-end replication of the paper's CNN-LSTM using FastText Hindi embeddings.

What this script does
- Loads cleaned CSVs (id, text, label) for train/val and test (no label)
- Optional: loads raw XLSX instead (same columns as the official release)
- Cleans text (URLs, emojis, extra spaces), optional stopword removal
- Tokenizes + pads to max_len=300
- Loads FastText Hindi vectors (cc.hi.300.vec) and builds embedding matrix
- Builds CNN-LSTM: Embedding -> Conv1D(64,k=5) -> MaxPool(2) -> LSTM(64) -> Dense(32) -> Sigmoid
- Trains with Adam(1e-3), batch_size=64, epochs=10
- Evaluates on val, saves metrics + confusion matrix
- Predicts on test, saves predictions
- Exports sentence-level embeddings (LSTM output) for train/val/test
- Saves all artifacts into an output folder with a timestamp

Updated Hindi Fake News Detection training script (CNN-LSTM) — fixes and improvements
- Monitors val_loss + AUC, adds ReduceLROnPlateau
- Optional upsampling of minority class (--upsample)
- Optional focal loss (--use_focal)
- Improved model: Conv1D -> MaxPool -> Bidirectional LSTM -> Dropout -> Dense
- Better FastText lookup (tries lowercased tokens) and reports zero embedding rows
- Reports AUC and PR-AUC, saves artifacts
- Keeps original functionality: saves predictions, embeddings, tokenizer, history

Usage (example):
python train_cnn_lstm_hindi_fixed.py \
  --train_csv train_clean.csv \
  --val_csv val_clean.csv \
  --test_csv test_clean.csv \
  --fasttext_vec cc.hi.300.vec \
  --outdir outputs \
  --upsample

local usage: (only for reference)
python "E:\constraint2021_implement\main3.py" `
>>   --train_csv "E:\constraint2021_implement\train_clean.csv" `
>>   --val_csv   "E:\constraint2021_implement\val_clean.csv" `
>>   --test_csv  "E:\constraint2021_implement\test_clean.csv" `
>>   --fasttext_vec "E:\constraint2021_implement\cc.hi.300.vec" `
>>   --outdir "E:\constraint2021_implement\outputs"

Author: you (patched)
"""

import argparse
import os
import json
import random
from datetime import datetime
import re
from typing import List, Optional

import numpy as np
import pandas as pd

# TensorFlow/Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Embedding, Conv1D, MaxPooling1D, LSTM,
                                     Dense, Dropout, Bidirectional)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import matplotlib.pyplot as plt

# ------------------------
# Reproducibility
# ------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------
# Text cleaning
# ------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F]"
    "|[\U0001F300-\U0001F5FF]"
    "|[\U0001F680-\U0001F6FF]"
    "|[\U0001F1E0-\U0001F1FF]",
    flags=re.UNICODE,
)
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[\u2000-\u206F\u2E00-\u2E7F\\'!\"#\$%&\(\)\*\+,\-\./:;<=>\?@\[\]^_`\{\|\}~]")


def load_stopwords(path: Optional[str]) -> set:
    if not path:
        return set()
    sw = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                sw.add(w)
    return sw


def clean_text(text: str, stopwords: set, lowercase: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    t = text
    t = URL_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = PUNCT_RE.sub(" ", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    if lowercase:
        t = t.lower()
    if stopwords:
        tokens = [tok for tok in t.split() if tok not in stopwords]
        t = " ".join(tokens)
    return t

# ------------------------
# Data loading
# ------------------------

def map_label_generic(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).lower()
    return 1 if "fake" in s or s.strip() in {"1","true","t","y","yes"} else 0


def read_dataframe(csv: Optional[str], xlsx: Optional[str], expect_label: bool) -> pd.DataFrame:
    if csv:
        df = pd.read_csv(csv)
    elif xlsx:
        df = pd.read_excel(xlsx)
    else:
        raise ValueError("Provide either CSV or XLSX path")

    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    id_col = pick("id", "unique id", "uid")
    text_col = pick("text", "post", "headline", "body")
    label_col = pick("label", "labels set", "target") if expect_label else None

    if not id_col or not text_col:
        raise ValueError(f"Could not find id/text columns in {csv or xlsx}. Columns: {list(df.columns)}")

    out = pd.DataFrame({"id": df[id_col], "text": df[text_col]})
    if expect_label and label_col:
        out["label"] = df[label_col].apply(map_label_generic)
    elif expect_label:
        raise ValueError("Expected labels but couldn't find a label column.")

    return out

# ------------------------
# Embeddings
# ------------------------

def load_fasttext_vec(vec_path: str, vocab: dict, embedding_dim: int, vocab_size: int) -> np.ndarray:
    """Loads FastText .vec file and builds an embedding matrix aligned to capped vocab indices [0..vocab_size-1].
    Tries to match exact token and lowercased token to increase coverage.
    """
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    found = 0
    vocab_lookup = vocab  # tokenizer.word_index mapping
    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        parts = first.strip().split()
        header_like = len(parts) == 2 and parts[1].isdigit()
        if not header_like:
            # first line is a vector line
            word = parts[0]
            vec = np.asarray(parts[1:1 + embedding_dim], dtype=np.float32)
            idx = vocab_lookup.get(word) or vocab_lookup.get(word.lower())
            if idx is not None and idx < vocab_size:
                embedding_matrix[idx] = vec
                found += 1
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < embedding_dim + 1:
                continue
            word = parts[0]
            try:
                vec = np.asarray(parts[1:1 + embedding_dim], dtype=np.float32)
            except Exception:
                continue
            idx = vocab_lookup.get(word)
            if idx is None:
                idx = vocab_lookup.get(word.lower())
            if idx is not None and idx < vocab_size:
                if not embedding_matrix[idx].any():
                    embedding_matrix[idx] = vec
                    found += 1
    coverage = 100.0 * found / max(1, vocab_size)
    print(f"FastText coverage: {found}/{vocab_size} tokens = {coverage:.2f}%")
    zero_rows = np.where(~embedding_matrix.any(axis=1))[0]
    print(f"Zero-embedding rows: {len(zero_rows)} / {embedding_matrix.shape[0]}")
    return embedding_matrix

# ------------------------
# Model
# ------------------------

def build_model_improved(vocab_size: int, embedding_dim: int, max_len: int,
                         embedding_matrix: Optional[np.ndarray], embed_trainable: bool, lr: float) -> Model:
    model = Sequential()
    if embedding_matrix is not None:
        model.add(
            Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                      weights=[embedding_matrix], input_length=max_len,
                      trainable=embed_trainable, name="embedding")
        )
    else:
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, trainable=True, name="embedding"))

    model.add(Conv1D(filters=128, kernel_size=5, activation="relu", name="conv1"))
    model.add(MaxPooling1D(pool_size=2, name="pool1"))
    model.add(Bidirectional(LSTM(128, activation="tanh"), name="bilstm"))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu", name="dense1"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid", name="out"))
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def build_encoder(model: Model) -> Model:
    # Return a model that outputs the LSTM (bidirectional) representation if present
    # Try to locate a layer named 'bilstm' or fallback to 'lstm'
    try:
        lstm_layer = model.get_layer("bilstm")
    except Exception:
        try:
            lstm_layer = model.get_layer("lstm")
        except Exception:
            raise RuntimeError("No LSTM layer named 'bilstm' or 'lstm' found in model")
    return Model(inputs=model.inputs, outputs=lstm_layer.output, name="encoder")

# ------------------------
# Plotting
# ------------------------

def plot_confusion(cm: np.ndarray, labels: List[str], path: str, title: str):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# ------------------------
# Helper: focal loss
# ------------------------

def focal_loss(gamma=2., alpha=0.25):
    def fl(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        loss = -alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)
    return fl

# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    # Data inputs
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--train_xlsx", type=str, default=None)
    parser.add_argument("--val_xlsx", type=str, default=None)
    parser.add_argument("--test_xlsx", type=str, default=None)

    # Text processing
    parser.add_argument("--stopwords_txt", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--num_words", type=int, default=50000)
    parser.add_argument("--lowercase", action="store_true", default=True, help="lowercase text during cleaning")

    # Embeddings
    parser.add_argument("--fasttext_vec", type=str, required=True)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--embed_trainable", type=lambda x: str(x).lower() in {"1","true","t","yes","y"}, default=True)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--upsample", action="store_true", help="upsample minority class in training data")
    parser.add_argument("--use_focal", action="store_true", help="use focal loss instead of BCE")

    # Output
    parser.add_argument("--outdir", type=str, default="outputs")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"run_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # Load stopwords
    stopwords = load_stopwords(args.stopwords_txt)
    print(f"Loaded {len(stopwords)} stopwords.")

    # Load data
    train_df = read_dataframe(args.train_csv, args.train_xlsx, expect_label=True)
    val_df = read_dataframe(args.val_csv, args.val_xlsx, expect_label=True)
    test_df = read_dataframe(args.test_csv, args.test_xlsx, expect_label=False)

    # Clean text
    train_df["clean"] = train_df["text"].apply(lambda x: clean_text(x, stopwords, lowercase=args.lowercase))
    val_df["clean"] = val_df["text"].apply(lambda x: clean_text(x, stopwords, lowercase=args.lowercase))
    test_df["clean"] = test_df["text"].apply(lambda x: clean_text(x, stopwords, lowercase=args.lowercase))

    # Basic diagnostics
    print("Train label distribution:\n", train_df['label'].value_counts(dropna=False))
    print("Val   label distribution:\n", val_df['label'].value_counts(dropna=False))

    # Tokenizer: fit on training clean text (after cleaning)
    tokenizer = Tokenizer(num_words=args.num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean"].tolist())

    def to_seq_from_series(series: pd.Series) -> np.ndarray:
        return pad_sequences(
            tokenizer.texts_to_sequences(series.tolist()),
            maxlen=args.max_len, padding="post", truncating="post",
        )

    # Optional upsampling
    if args.upsample:
        pos = train_df[train_df['label'] == 1]
        neg = train_df[train_df['label'] == 0]
        if len(pos) == 0:
            raise ValueError("No positive (label=1) examples in training set to upsample")
        if len(pos) < len(neg):
            pos_up = resample(pos, replace=True, n_samples=len(neg), random_state=SEED)
            train_df_bal = pd.concat([neg, pos_up]).sample(frac=1, random_state=SEED).reset_index(drop=True)
            print(f"Upsampled positives: {len(pos)} -> {len(pos_up)}; new train size = {len(train_df_bal)}")
        else:
            train_df_bal = train_df.copy()
    else:
        train_df_bal = train_df.copy()

    # Recompute tokenizer? we keep tokenizer fitted on original training texts (fine).
    # Convert to sequences
    X_train = to_seq_from_series(train_df_bal["clean"])    
    X_val = to_seq_from_series(val_df["clean"])    
    X_test = to_seq_from_series(test_df["clean"])

    y_train = train_df_bal["label"].astype(int).values
    y_val = val_df["label"].astype(int).values

    # Compute class weights for imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print("Class weights:", class_weight_dict)

    # Embedding matrix aligned to capped vocab
    vocab_size = min(args.num_words, len(tokenizer.word_index) + 1)
    embedding_matrix = load_fasttext_vec(args.fasttext_vec, tokenizer.word_index, args.embedding_dim, vocab_size)

    # Build & train model
    model = build_model_improved(vocab_size=vocab_size, embedding_dim=args.embedding_dim,
                                 max_len=args.max_len, embedding_matrix=embedding_matrix,
                                 embed_trainable=bool(args.embed_trainable), lr=args.lr)

    # Optionally replace loss with focal
    if args.use_focal:
        print("Using focal loss")
        model.compile(optimizer=Adam(learning_rate=args.lr), loss=focal_loss(), metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

    ckpt_path = os.path.join(outdir, "model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=1, class_weight=class_weight_dict
    )

    # Save history
    with open(os.path.join(outdir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({k: list(map(float, v)) for k, v in history.history.items()}, f, indent=2)

    # ------------ Evaluation (threshold 0.5 + tuned) ------------
    val_probs = model.predict(X_val, batch_size=args.batch_size).ravel()

    # Useful diagnostics
    print("val_probs: min, median, mean, max =", val_probs.min(), np.median(val_probs), val_probs.mean(), val_probs.max())
    print("counts > 0.5, >0.1, >0.05:", (val_probs>0.5).sum(), (val_probs>0.1).sum(), (val_probs>0.05).sum())

    # Default 0.5
    val_pred_05 = (val_probs >= 0.5).astype(int)
    report_05 = classification_report(y_val, val_pred_05, output_dict=True, zero_division=0)
    with open(os.path.join(outdir, "val_classification_report_thr0.50.json"), "w", encoding="utf-8") as f:
        json.dump(report_05, f, indent=2)
    cm_05 = confusion_matrix(y_val, val_pred_05)
    plot_confusion(cm_05, labels=["non-fake (0)", "fake (1)"], path=os.path.join(outdir, "val_confusion_matrix_thr0.50.png"), title="Confusion (thr=0.50)")

    # Threshold sweep to maximize F1 for class 1
    thr_grid = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thr_grid:
        preds = (val_probs >= thr).astype(int)
        f1 = f1_score(y_val, preds, pos_label=1)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    print(f"Best threshold on val = {best_thr:.2f} (F1 for class 1 = {best_f1:.4f})")
    with open(os.path.join(outdir, "best_threshold.txt"), "w") as f:
        f.write(f"{best_thr:.4f}\n")

    # Reports using best threshold
    val_pred_best = (val_probs >= best_thr).astype(int)
    report_best = classification_report(y_val, val_pred_best, output_dict=True, zero_division=0)
    with open(os.path.join(outdir, "val_classification_report_best_thr.json"), "w", encoding="utf-8") as f:
        json.dump({"best_threshold": best_thr, "report": report_best}, f, indent=2)
    cm_best = confusion_matrix(y_val, val_pred_best)
    plot_confusion(cm_best, labels=["non-fake (0)", "fake (1)"], path=os.path.join(outdir, "val_confusion_matrix_best_thr.png"), title=f"Confusion (best thr={best_thr:.2f})")

    # Additional metrics
    try:
        auc = roc_auc_score(y_val, val_probs)
        ap = average_precision_score(y_val, val_probs)
    except Exception:
        auc, ap = None, None
    print("Validation ROC-AUC:", auc, "  PR-AUC (avg precision):", ap)

    print("Validation metrics at thr=0.50:")
    print(json.dumps(report_05, indent=2))
    print(f"\nValidation metrics at best thr={best_thr:.2f}:")
    print(json.dumps(report_best, indent=2))

    # ------------ Save predictions (val + test) ------------
    val_out = val_df.copy()
    val_out["pred_prob"] = val_probs
    val_out["pred_label_0p5"] = val_pred_05
    val_out["pred_label_best"] = val_pred_best
    val_out.to_csv(os.path.join(outdir, "predictions_val.csv"), index=False, encoding="utf-8")

    test_probs = model.predict(X_test, batch_size=args.batch_size).ravel()
    test_pred_05 = (test_probs >= 0.5).astype(int)
    test_pred_best = (test_probs >= best_thr).astype(int)
    test_out = test_df.copy()
    test_out["pred_prob"] = test_probs
    test_out["pred_label_0p5"] = test_pred_05
    test_out["pred_label_best"] = test_pred_best
    test_out.to_csv(os.path.join(outdir, "predictions_test.csv"), index=False, encoding="utf-8")

    # Save tokenizer
    import pickle
    with open(os.path.join(outdir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    # ------------ Export sentence embeddings (LSTM outputs) ------------
    encoder = build_encoder(model)

    def embed_split(X: np.ndarray, ids: pd.Series, split: str):
        vecs = encoder.predict(X, batch_size=args.batch_size)
        df = pd.DataFrame({
            "id": ids.values,
            "embedding": [json.dumps(v.tolist(), ensure_ascii=False) for v in vecs],
        })
        df.to_csv(os.path.join(outdir, f"embeddings_{split}.csv"), index=False, encoding="utf-8")

    embed_split(X_train, train_df_bal["id"], "train")
    embed_split(X_val,   val_df["id"],   "val")
    embed_split(X_test,  test_df["id"],  "test")

    print(f"\nAll artifacts saved under: {outdir}\n")
    print("Key files:")
    for fn in [
        "args.json", "model.keras", "history.json", "best_threshold.txt",
        "val_classification_report_thr0.50.json", "val_classification_report_best_thr.json",
        "val_confusion_matrix_thr0.50.png", "val_confusion_matrix_best_thr.png",
        "predictions_val.csv", "predictions_test.csv", "tokenizer.pkl",
        "embeddings_train.csv", "embeddings_val.csv", "embeddings_test.csv",
    ]:
        print(" -", os.path.join(outdir, fn))


if __name__ == "__main__":
    main()
