"""
======================================================================
LSTM-Based Text Generation — Generative AI Engineer Task
======================================================================
Dataset : Shakespeare's Complete Works (Project Gutenberg)
Framework: TensorFlow / Keras
======================================================================

Quick-start
-----------
1.  pip install tensorflow numpy requests
2.  python lstm_text_generation.py
"""

import os
import re
import random
import urllib.request

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ─── Reproducibility ─────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_URL = "https://www.gutenberg.org/files/100/100-0.txt"
DATA_PATH   = "shakespeare.txt"


# ═══════════════════════════════════════════════════════════════════════
# SECTION A — DATASET LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def download_dataset(url: str, save_path: str) -> None:
    """Download the text corpus if not already cached locally."""
    if os.path.exists(save_path):
        print(f"[INFO] Dataset already exists at '{save_path}'. Skipping download.")
        return
    print(f"[INFO] Downloading dataset from:\n       {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"[INFO] Saved to '{save_path}'.")


def load_and_clean_text(path: str) -> str:
    """
    Load raw text, strip Gutenberg boilerplate, lowercase,
    and remove punctuation.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()

    # Strip Gutenberg header/footer
    start_marker = "*** START OF"
    end_marker   = "*** END OF"
    start_idx = raw.find(start_marker)
    end_idx   = raw.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        raw = raw[start_idx:end_idx]

    text = raw.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace

    print(f"[INFO] Corpus size after cleaning: {len(text):,} characters")
    return text


def build_vocabulary(text: str):
    """Build word-to-index and index-to-word mappings."""
    words        = text.split()
    unique_words = sorted(set(words))
    word2idx     = {w: i for i, w in enumerate(unique_words)}
    idx2word     = {i: w for w, i in word2idx.items()}
    vocab_size   = len(unique_words)
    print(f"[INFO] Total words in corpus : {len(words):,}")
    print(f"[INFO] Vocabulary size       : {vocab_size:,}")
    return words, word2idx, idx2word, vocab_size


def create_sequences(words, word2idx, seq_length: int = 30):
    sequences  = []
    next_words = []
    for i in range(0, len(words) - seq_length, 3):
        sequences.append([word2idx[w] for w in words[i : i + seq_length]])
        next_words.append(word2idx[words[i + seq_length]])
    X = np.array(sequences,  dtype=np.int32)
    y = np.array(next_words, dtype=np.int32)   # integers only — NO to_categorical
    print(f"[INFO] Training samples created: {len(X):,}")
    print(f"[INFO] X shape: {X.shape}  |  y shape: {y.shape}")
    return X, y

# ═══════════════════════════════════════════════════════════════════════
# SECTION B — MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

def build_model(
    vocab_size: int,
    seq_length: int,
    embedding_dim: int   = 128,
    lstm_units: int      = 256,
    dropout_rate: float  = 0.3,
    num_lstm_layers: int = 2,
    use_bidirectional: bool = False,
) -> Sequential:
    """
    Build an LSTM text-generation model.

    Architecture: Embedding -> [LSTM x N] -> Dense(vocab_size, softmax)

    Compiled with sparse_categorical_crossentropy — accepts integer
    labels directly, no one-hot matrix needed.
    """
    model = Sequential(name="LSTM_TextGen")

    # Embedding layer: integer tokens -> dense vectors
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=seq_length,
            name="token_embedding",
        )
    )

    # Stacked LSTM layers
    for idx in range(num_lstm_layers):
        return_seq = (idx < num_lstm_layers - 1)
        lstm_layer = LSTM(
            lstm_units,
            return_sequences=return_seq,
            name=f"lstm_{idx + 1}",
        )
        if use_bidirectional:
            model.add(Bidirectional(lstm_layer, name=f"bilstm_{idx + 1}"))
        else:
            model.add(lstm_layer)
        model.add(Dropout(dropout_rate, name=f"dropout_{idx + 1}"))

    # Output layer: probability over full vocabulary
    model.add(Dense(vocab_size, activation="softmax", name="output"))

    # sparse_categorical_crossentropy = same math as categorical_crossentropy
    # but works on raw integer labels — no 45 GB one-hot allocation.
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ═══════════════════════════════════════════════════════════════════════
# SECTION C — TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_model(
    model: Sequential,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int             = 50,
    batch_size: int         = 128,
    validation_split: float = 0.1,
    checkpoint_path: str    = "best_model.keras",
):
    """
    Train with EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print("\n[INFO] Starting training ...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )
    print("[INFO] Training complete.")
    return history


# ═══════════════════════════════════════════════════════════════════════
# SECTION D — TEXT GENERATION
# ═══════════════════════════════════════════════════════════════════════

def sample_with_temperature(probabilities: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample next token index using temperature scaling.

    temperature < 1  -> more conservative / repetitive
    temperature = 1  -> raw model probabilities
    temperature > 1  -> more creative / noisy
    """
    probabilities = np.asarray(probabilities).astype("float64")
    log_probs     = np.log(probabilities + 1e-10) / temperature
    exp_probs     = np.exp(log_probs - np.max(log_probs))
    exp_probs    /= exp_probs.sum()
    return np.random.choice(len(exp_probs), p=exp_probs)


def generate_text(
    model: Sequential,
    seed_text: str,
    word2idx: dict,
    idx2word: dict,
    seq_length: int,
    num_words: int     = 100,
    temperature: float = 0.8,
) -> str:
    """
    Generate num_words new words from a seed prompt.

    Steps: tokenise seed -> pad/truncate -> predict next token ->
    append -> slide window -> repeat.
    """
    seed_clean = re.sub(r"[^a-z0-9\s]", "", seed_text.lower())
    seed_words = seed_clean.split()

    current_seq = [
        word2idx.get(w, random.randint(0, len(word2idx) - 1))
        for w in seed_words
    ]

    # Pad or truncate to exactly seq_length
    if len(current_seq) >= seq_length:
        current_seq = current_seq[-seq_length:]
    else:
        current_seq = [0] * (seq_length - len(current_seq)) + current_seq

    generated_words = list(seed_words)

    for _ in range(num_words):
        x         = np.array(current_seq).reshape(1, seq_length)
        probs     = model.predict(x, verbose=0)[0]
        next_idx  = sample_with_temperature(probs, temperature)
        next_word = idx2word[next_idx]
        generated_words.append(next_word)
        current_seq = current_seq[1:] + [next_idx]

    return " ".join(generated_words)


# ═══════════════════════════════════════════════════════════════════════
# SECTION E — BONUS: ARCHITECTURE EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENT_CONFIGS = [
    {
        "name": "Baseline (2-layer LSTM, 256 units)",
        "lstm_units": 256, "num_lstm_layers": 2,
        "embedding_dim": 128, "use_bidirectional": False,
    },
    {
        "name": "Deeper (3-layer LSTM, 512 units)",
        "lstm_units": 512, "num_lstm_layers": 3,
        "embedding_dim": 256, "use_bidirectional": False,
    },
    {
        "name": "Bidirectional (2-layer BiLSTM, 256 units)",
        "lstm_units": 256, "num_lstm_layers": 2,
        "embedding_dim": 128, "use_bidirectional": True,
    },
]


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    # ── Hyper-parameters ─────────────────────────────────────────────
    SEQ_LENGTH      = 30
    EMBEDDING_DIM   = 128
    LSTM_UNITS      = 256
    NUM_LSTM_LAYERS = 2
    DROPOUT_RATE    = 0.3
    BATCH_SIZE      = 128
    EPOCHS          = 50
    GENERATE_WORDS  = 100
    TEMPERATURES    = [0.5, 0.8, 1.2]
    MAX_WORDS       = 200_000   # set to None to use full corpus

    # ── A. Load & preprocess ─────────────────────────────────────────
    download_dataset(DATASET_URL, DATA_PATH)
    text = load_and_clean_text(DATA_PATH)
    words, word2idx, idx2word, vocab_size = build_vocabulary(text)

    if MAX_WORDS and len(words) > MAX_WORDS:
        words = words[:MAX_WORDS]
        print(f"[INFO] Corpus truncated to {MAX_WORDS:,} words for speed.")

    # X = integer sequences, y = integer labels (NO one-hot / to_categorical)
    X, y = create_sequences(words, word2idx, seq_length=SEQ_LENGTH)

    # ── B. Build model ───────────────────────────────────────────────
    model = build_model(
        vocab_size=vocab_size,
        seq_length=SEQ_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE,
        num_lstm_layers=NUM_LSTM_LAYERS,
    )

    # ── C. Train ─────────────────────────────────────────────────────
    train_model(
        model=model,
        X=X,
        y=y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_path="best_model.keras",
    )

    # ── D. Generate text ─────────────────────────────────────────────
    seed_inputs = [
        "to be or not to be that is the question",
        "shall i compare thee to a summers day",
        "all the worlds a stage and all the men",
        "good friends romans countrymen lend me your ears",
    ]

    output_lines = ["=" * 70, "GENERATED TEXT SAMPLES", "=" * 70]

    for seed in seed_inputs:
        output_lines.append(f'\nSeed: "{seed}"')
        for temp in TEMPERATURES:
            generated = generate_text(
                model=model,
                seed_text=seed,
                word2idx=word2idx,
                idx2word=idx2word,
                seq_length=SEQ_LENGTH,
                num_words=GENERATE_WORDS,
                temperature=temp,
            )
            output_lines.append(f"\n  [Temperature = {temp}]")
            output_lines.append(f"  {generated}")
        output_lines.append("\n" + "-" * 70)

    full_output = "\n".join(output_lines)
    print("\n" + full_output)

    with open("generated_outputs.txt", "w", encoding="utf-8") as f:
        f.write(full_output)
    print("\n[INFO] Generated text saved to 'generated_outputs.txt'.")

    # ── E. Bonus — experiment summary ────────────────────────────────
    print("\n" + "=" * 70)
    print("BONUS: ARCHITECTURE EXPERIMENT CONFIGURATIONS")
    print("=" * 70)
    for cfg in EXPERIMENT_CONFIGS:
        print(f"\n  {cfg['name']}")
        for k, v in cfg.items():
            if k != "name":
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()