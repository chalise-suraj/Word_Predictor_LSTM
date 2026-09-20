"""Build and train the project's sentence-aware LSTM next-word predictor.

Run from the project root with ``.venv\\Scripts\\python.exe src\\train_lstm.py``.
The PDF is the sole corpus source.  The script deliberately creates each split
at sentence level before vocabulary fitting and sequence generation.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import fitz
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM


ROOT = Path(__file__).resolve().parents[1]
RAW_PDF = ROOT / "data" / "raw" / "Technical-Writing.pdf"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SEED = 42
SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 64
LSTM_UNITS = 64
DROPOUT_RATE = 0.25
BATCH_SIZE = 256
EPOCHS = 12
MIN_FREQUENCY = 4
SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<START>", "<END>")
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def normalise_text(text: str) -> str:
    """Match the train and inference token representation."""
    text = unicodedata.normalize("NFKC", text).lower()
    return text.replace("\u00ad", "")


def tokens_from_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalise_text(text))


def extract_sentences(pdf_path: Path) -> list[list[str]]:
    """Extract PDF text, discard isolated page numbers, then tokenize sentences."""
    pages: list[str] = []
    with fitz.open(pdf_path) as document:
        for page in document:
            lines = []
            for line in page.get_text("text").splitlines():
                line = re.sub(r"\s+", " ", line).strip()
                if line and not re.fullmatch(r"\d+", line):
                    lines.append(line)
            pages.append(" ".join(lines))
    text = "\n".join(pages)
    sentence_texts = re.split(r"(?<=[.!?])\s+", text)
    return [tokens for sentence in sentence_texts if len(tokens := tokens_from_text(sentence)) >= 2]


def build_vocabulary(train_sentences: list[list[str]]) -> tuple[dict[str, int], dict[int, str]]:
    # Train-only vocabulary avoids evaluation leakage. Folding singletons into
    # <UNK> prevents a mostly one-observation output space from dominating this
    # small textbook corpus; common technical words remain individual classes.
    counts = Counter(word for sentence in train_sentences for word in sentence)
    words = sorted(word for word, count in counts.items() if count >= MIN_FREQUENCY)
    word_to_index = {word: index for index, word in enumerate([*SPECIAL_TOKENS, *words])}
    return word_to_index, {index: word for word, index in word_to_index.items()}


def make_sequences(sentences: list[list[str]], word_to_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Create one padded preceding-context -> immediate-next-token example per token."""
    pad, unk, start, end = (word_to_index[token] for token in SPECIAL_TOKENS)
    inputs: list[list[int]] = []
    targets: list[int] = []
    for sentence in sentences:
        token_ids = [start, *(word_to_index.get(word, unk) for word in sentence), end]
        for target_position in range(1, len(token_ids)):
            context = token_ids[max(0, target_position - SEQUENCE_LENGTH):target_position]
            inputs.append([pad] * (SEQUENCE_LENGTH - len(context)) + context)
            targets.append(token_ids[target_position])
    return np.asarray(inputs, dtype=np.int32), np.asarray(targets, dtype=np.int32)


def save_artifacts(corpus: list[list[str]], vocabulary: dict[str, int], inverse_vocabulary: dict[int, str], arrays: dict[str, np.ndarray]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "technical_writing_corpus.txt").write_text(
        "\n".join(" ".join(sentence) for sentence in corpus), encoding="utf-8"
    )
    with (PROCESSED / "vocabulary.json").open("w", encoding="utf-8") as file:
        json.dump({"word_to_index": vocabulary, "index_to_word": {str(k): v for k, v in inverse_vocabulary.items()}}, file, ensure_ascii=False, indent=2)
    for name, array in arrays.items():
        np.save(PROCESSED / f"{name}.npy", array)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    extracted_sentences = extract_sentences(RAW_PDF)
    # Repeated PDF elements can produce identical sentences. Keep the first
    # instance so duplicate contexts cannot straddle train/validation/test.
    seen_sentences: set[tuple[str, ...]] = set()
    sentences = []
    for sentence in extracted_sentences:
        key = tuple(sentence)
        if key not in seen_sentences:
            seen_sentences.add(key)
            sentences.append(sentence)
    sentence_ids = np.arange(len(sentences))
    train_ids, holdout_ids = train_test_split(sentence_ids, test_size=0.20, random_state=SEED)
    validation_ids, test_ids = train_test_split(holdout_ids, test_size=0.50, random_state=SEED)
    train_sentences = [sentences[index] for index in train_ids]
    validation_sentences = [sentences[index] for index in validation_ids]
    test_sentences = [sentences[index] for index in test_ids]
    word_to_index, index_to_word = build_vocabulary(train_sentences)
    X_train, y_train = make_sequences(train_sentences, word_to_index)
    X_validation, y_validation = make_sequences(validation_sentences, word_to_index)
    X_test, y_test = make_sequences(test_sentences, word_to_index)
    arrays = {"X_train": X_train, "y_train": y_train, "X_validation": X_validation, "y_validation": y_validation, "X_test": X_test, "y_test": y_test}
    save_artifacts(sentences, word_to_index, index_to_word, arrays)
    vocab_size = len(word_to_index)
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True, input_shape=(SEQUENCE_LENGTH,)),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT_RATE),
        Dense(vocab_size, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    checkpoint = MODELS / "best_lstm_model.keras"
    history = model.fit(
        X_train, y_train,
        validation_data=(X_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        ],
        verbose=2,
    )
    best_model = tf.keras.models.load_model(checkpoint)
    best_model.save(MODELS / "technical_writing_lstm_best.keras")
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    config = {
        "sequence_length": SEQUENCE_LENGTH, "vocabulary_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM, "lstm_units": LSTM_UNITS, "dropout_rate": DROPOUT_RATE,
        "token_pattern": TOKEN_PATTERN.pattern, "lowercase": True, "unicode_normalization": "NFKC",
        "special_tokens": {"padding": "<PAD>", "unknown": "<UNK>", "start": "<START>", "end": "<END>"},
        "split": {"unit": "unique sentence", "train": 0.80, "validation": 0.10, "test": 0.10, "random_seed": SEED},
        "minimum_train_word_frequency": MIN_FREQUENCY,
        "sequence_method": "left-padded preceding context, one immediate-next-token target per sentence token",
        "test_loss": float(test_loss), "test_accuracy": float(test_accuracy),
        "epochs_ran": len(history.history["loss"]),
    }
    with (PROCESSED / "model_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    print(json.dumps({"sentences": len(sentences), "vocabulary_size": vocab_size, "train": X_train.shape, "validation": X_validation.shape, "test": X_test.shape, "test_loss": test_loss, "test_accuracy": test_accuracy}, indent=2))


if __name__ == "__main__":
    main()
