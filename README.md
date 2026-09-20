# 📚 Technical Writing Next-Word Predictor using LSTM

> **An educational Natural Language Processing project for next-word prediction using a Long Short-Term Memory (LSTM) neural network trained on a Technical Writing textbook.**

---

## 📌 Project Overview

This project implements an **LSTM-based next-word prediction system** using the text extracted from a Technical Writing textbook.

The primary objective is to demonstrate how a sequence model can learn relationships between words and predict the most probable word that comes next based on the preceding context.

The complete workflow follows:

```text
Technical-Writing.pdf
        │
        ▼
PDF Text Extraction
        │
        ▼
Corpus Cleaning & Preprocessing
        │
        ▼
Sentence Segmentation
        │
        ▼
Word Tokenization
        │
        ▼
Vocabulary Creation
        │
        ▼
Training Sequence Generation
        │
        ▼
Train / Validation / Test Split
        │
        ▼
LSTM Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Saved LSTM Model
        │
        ▼
Next-Word Prediction
        │
        ▼
Top-K Word Suggestions
```

The final prediction system is designed to behave similarly to **predictive typing**, where a user enters text and the trained LSTM provides probable next-word suggestions.

---

# 🎯 Project Objectives

The major objectives of this project are:

* Extract textual information from a Technical Writing PDF.
* Build a clean text corpus from the extracted content.
* Analyze and preprocess the corpus.
* Perform sentence segmentation and word tokenization.
* Build a vocabulary and word-index mapping.
* Convert natural language into numerical sequences.
* Generate supervised next-word prediction samples.
* Train an LSTM neural network.
* Evaluate the trained model.
* Save the trained model and required configuration.
* Implement a compact inference notebook.
* Predict multiple possible next words for user-provided text.
* Demonstrate context-dependent next-word prediction.

---

# 🧠 Core Concept

The project treats next-word prediction as a supervised sequence-learning problem.

Given a sequence of previous words:

```text
Technical writing is
```

the model attempts to estimate:

```text
P(next word | previous words)
```

The LSTM produces a probability distribution over the complete vocabulary.

Conceptually:

```text
Previous Context
       │
       ▼
Word Indices
       │
       ▼
Embedding
       │
       ▼
LSTM
       │
       ▼
Dropout
       │
       ▼
Dense + Softmax
       │
       ▼
Probability Distribution
       │
       ▼
Top-K Next-Word Suggestions
```

For example:

```text
Input:
technical

Possible suggestions:

1. writing
2. communication
3. research
```

The exact predictions depend entirely on what the model learns from the training corpus.

---

# 🏗️ Project Architecture

The project uses a traditional **word-level LSTM architecture**.

```text
Input Sequence
      │
      ▼
Embedding Layer
      │
      ▼
LSTM Layer
      │
      ▼
Dropout Layer
      │
      ▼
Dense Layer
      │
      ▼
Softmax Output
      │
      ▼
Next-Word Probabilities
```

### Current Model Configuration

| Parameter               |                           Value |
| ----------------------- | ------------------------------: |
| Architecture            |                            LSTM |
| Embedding Dimension     |                             128 |
| LSTM Units              |                             256 |
| Dropout                 |                            0.20 |
| Sequence Length         |                              20 |
| Batch Size              |                             128 |
| Learning Rate           |                           0.001 |
| Optimizer               |                            Adam |
| Loss                    | Sparse Categorical Crossentropy |
| Output Activation       |                         Softmax |
| Maximum Epochs          |                              30 |
| Early Stopping          |                         Enabled |
| Learning Rate Reduction |                         Enabled |

> **Note:** The model is intentionally implemented using an LSTM. Transformer-based architectures such as GPT or BERT are not used in this project.

---

# 📂 Project Structure

```text
Word_Predictor_LSTM/
│
├── data/
│   │
│   ├── raw/
│   │   └── Technical-Writing.pdf
│   │
│   └── processed/
│       ├── technical_writing_corpus.txt
│       ├── vocabulary.json
│       ├── model_config.json
│       ├── X_train.npy
│       ├── y_train.npy
│       ├── X_validation.npy
│       ├── y_validation.npy
│       ├── X_test.npy
│       └── y_test.npy
│
├── models/
│   └── technical_writing_lstm_best.keras
│
├── notebooks/
│   ├── classification_model.ipynb
│   └── text_prediction.ipynb
│
└── README.md
```

---

# 📁 Directory Description

## `data/raw/`

Contains the original source document used to build the corpus.

```text
Technical-Writing.pdf
```

This document is the primary textual source for the project.

---

## `data/processed/`

Contains all processed data generated during preprocessing and training.

### `technical_writing_corpus.txt`

The cleaned text extracted from the source PDF.

### `vocabulary.json`

Contains the vocabulary and mappings required by the model.

Conceptually:

```text
word → index
index → word
```

It also contains the project's special tokens.

### `model_config.json`

Stores configuration information required to reproduce the model's input/output representation.

### Training arrays

```text
X_train.npy
y_train.npy

X_validation.npy
y_validation.npy

X_test.npy
y_test.npy
```

These files contain the numericalized training, validation, and testing data.

---

# 🤖 `models/`

Contains the trained LSTM model.

```text
technical_writing_lstm_best.keras
```

This is the saved model used by the prediction notebook.

---

# 📓 `notebooks/`

## `classification_model.ipynb`

This is the primary training and data-processing notebook.

It covers:

```text
PDF Extraction
      ↓
Corpus Analysis
      ↓
Corpus Cleaning
      ↓
Sentence Segmentation
      ↓
Vocabulary Construction
      ↓
Tokenization
      ↓
Sequence Generation
      ↓
Dataset Splitting
      ↓
LSTM Training
      ↓
Model Evaluation
      ↓
Model Saving
```

---

## `text_prediction.ipynb`

This notebook is responsible only for inference.

It loads:

```text
Saved LSTM Model
Saved Vocabulary
Saved Model Configuration
```

and provides a simple user-facing prediction interface.

The notebook does **not** retrain the model.

Its workflow is:

```text
User Input
    ↓
Preprocessing
    ↓
Word-to-Index Conversion
    ↓
Fixed-Length Sequence
    ↓
LSTM Prediction
    ↓
Softmax Probabilities
    ↓
Top-K Suggestions
```

---

# 🔄 Complete Project Workflow

## Phase 1 — Source Data

The project begins with:

```text
Technical-Writing.pdf
```

The PDF is processed to obtain its textual content.

Images and irrelevant PDF formatting information are not part of the language-model training objective.

---

## Phase 2 — Corpus Construction

The extracted text is cleaned and converted into:

```text
technical_writing_corpus.txt
```

The corpus becomes the textual foundation of the project.

---

## Phase 3 — Text Analysis

The training notebook analyzes the corpus to understand:

* Number of lines
* Number of sentences
* Number of words
* Vocabulary size
* Word frequencies
* Sentence lengths
* Repeated lines
* Short lines
* Numeric-only content
* Other preprocessing characteristics

This helps verify the quality of the extracted text before model training.

---

# 🧹 Phase 4 — Text Preprocessing

The corpus is prepared for machine learning.

The preprocessing pipeline includes:

```text
Raw Text
   ↓
Cleaning
   ↓
Sentence Segmentation
   ↓
Lowercasing / Normalization
   ↓
Tokenization
   ↓
Vocabulary Construction
   ↓
Numerical Encoding
```

The preprocessing used during inference must remain consistent with the preprocessing used during training.

---

# 📖 Phase 5 — Vocabulary Construction

A vocabulary is created from the processed corpus.

The project uses special tokens including:

```text
<PAD>
<UNK>
<START>
<END>
```

### Special Token Purpose

| Token     | Purpose                                |
| --------- | -------------------------------------- |
| `<PAD>`   | Makes sequences the same length        |
| `<UNK>`   | Represents unknown words               |
| `<START>` | Represents the beginning of a sequence |
| `<END>`   | Represents the end of a sequence       |

These tokens are part of the model's internal vocabulary.

Special tokens that are not appropriate for user-facing suggestions are filtered during prediction rather than being removed from the trained vocabulary.

---

# 🔢 Phase 6 — Sequence Generation

Next-word prediction is formulated as a supervised learning problem.

For example, given:

```text
technical writing is an important skill
```

training examples conceptually become:

```text
technical
        → writing

technical writing
        → is

technical writing is
        → an

technical writing is an
        → important

technical writing is an important
        → skill
```

The model therefore learns:

```text
Previous Context → Immediate Next Word
```

rather than simply learning the most frequent words in the entire corpus.

---

# 📏 Sequence Length

The current model uses:

```text
SEQUENCE_LENGTH = 20
```

This means the LSTM receives a fixed-size sequence representing the available context.

For longer input:

```text
word1 word2 ... word25
```

the appropriate recent context is used according to the project's sequence-generation methodology.

For shorter sequences, padding is applied consistently with the training representation.

---

# ✂️ Phase 7 — Dataset Splitting

The generated data is divided into:

```text
Training Set
Validation Set
Test Set
```

Conceptually:

```text
Complete Dataset
       │
       ├──────────────► Training
       │
       ├──────────────► Validation
       │
       └──────────────► Testing
```

### Training Set

Used to learn model parameters.

### Validation Set

Used during training to monitor generalization and control training through callbacks.

### Test Set

Used to evaluate the final model on previously unseen examples.

---

# 🧠 Phase 8 — LSTM Training

The model learns relationships between preceding words and their likely next words.

The main architecture is:

```text
Input
  │
  ▼
Embedding(128)
  │
  ▼
LSTM(256)
  │
  ▼
Dropout(0.20)
  │
  ▼
Dense(Vocabulary Size)
  │
  ▼
Softmax
```

The model uses:

```text
Optimizer:
Adam

Loss:
Sparse Categorical Crossentropy

Metric:
Accuracy
```

---

# ⚙️ Training Callbacks

The training process uses callbacks to prevent unnecessary training and improve optimization.

### Early Stopping

Training monitors validation loss.

If validation performance stops improving, training can stop early and the best weights can be restored.

### Model Checkpoint

The best-performing model is saved.

### Reduce Learning Rate

The learning rate can be reduced when validation performance stops improving.

This allows the optimizer to make smaller updates during later stages of training.

---

# 📊 Phase 9 — Model Evaluation

Model performance is evaluated using the held-out validation and test data.

Important considerations include:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy
* Test performance
* Prediction behavior
* Context sensitivity

Accuracy alone is not sufficient for understanding a language model.

The project also checks whether different contexts produce different next-word probability distributions.

---

# 🧪 Contextual Prediction

The most important behavioral test is whether the model responds to context.

For example:

```text
Input:
technical

Prediction:
writing
```

while another context may produce:

```text
Input:
is the

Prediction:
audience
```

and another:

```text
Input:
the best way

Prediction:
you
```

The objective is not to force a particular answer.

Instead, the model should learn relationships present in the Technical Writing corpus.

---

# ⌨️ Prediction Interface

The final notebook provides a simple interactive interface:

```text
Type text and press Enter for suggestions.
Type 'exit' and press Enter to stop.
```

Example:

```text
Input: technical

1. writing              27.72%
2. communication         3.46%
3. research              2.91%
```

The user can continue entering different contexts.

For example:

```text
Input: writing

1. the
2. a
3. ...
```

The model returns the highest-probability normal words while excluding internal special tokens from the user interface.

---

# 📈 Prediction Method

For every input:

```text
User Text
    ↓
Tokenization
    ↓
Word-to-Index Mapping
    ↓
Sequence Preparation
    ↓
LSTM
    ↓
Softmax
    ↓
Probability Ranking
    ↓
Top-K Suggestions
```

The model does not generate an entire paragraph automatically.

The primary objective is **next-word suggestion**, similar to predictive typing.

---

# 🚫 What This Project Does Not Use

This project intentionally does not use:

* ❌ Transformer architecture
* ❌ GPT
* ❌ BERT
* ❌ Large Language Models
* ❌ External general-purpose corpus
* ❌ Hard-coded predictions
* ❌ Artificial removal of common words
* ❌ Random predictions to make output appear diverse

The project focuses specifically on understanding and implementing an **LSTM-based next-word prediction system**.

---

# 🔍 Why Common Words May Still Appear

Words such as:

```text
the
and
of
to
a
is
```

are common in natural language.

Therefore, the model may legitimately predict them.

The objective is **not** to remove common words.

Instead, the model should learn:

```text
Context
   ↓
Probability distribution
   ↓
Contextually appropriate next words
```

A common word should remain a valid prediction when it is appropriate for the context.

---

# 🧩 Technologies Used

| Technology           | Purpose                          |
| -------------------- | -------------------------------- |
| Python               | Core programming language        |
| Jupyter Notebook     | Development and experimentation  |
| TensorFlow           | Deep learning framework          |
| Keras                | LSTM model implementation        |
| NumPy                | Numerical data processing        |
| JSON                 | Vocabulary/configuration storage |
| Regular Expressions  | Text preprocessing               |
| Pathlib              | Project path management          |
| PDF Processing Tools | Text extraction                  |

---

# 🛠️ Installation

## 1. Clone the Project

```bash
git clone <repository-url>
cd Word_Predictor_LSTM
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install tensorflow numpy jupyter matplotlib
```

If additional PDF-processing libraries are used by the training notebook, install those packages as required by the notebook.

---

# ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/classification_model.ipynb
```

Run the training notebook from top to bottom.

After training and saving the model, open:

```text
notebooks/text_prediction.ipynb
```

Run the prediction notebook.

---

# 🏃 Recommended Execution Order

Always follow this order:

```text
1. Technical-Writing.pdf
        ↓
2. classification_model.ipynb
        ↓
3. Processed Dataset
        ↓
4. Trained LSTM
        ↓
5. text_prediction.ipynb
        ↓
6. Next-Word Suggestions
```

Do not run the prediction notebook expecting it to train the model.

The prediction notebook loads the already-trained model.

---

# 🔗 Relationship Between Project Files

```text
Technical-Writing.pdf
        │
        ▼
classification_model.ipynb
        │
        ├──► technical_writing_corpus.txt
        │
        ├──► vocabulary.json
        │
        ├──► model_config.json
        │
        ├──► X_train.npy
        ├──► y_train.npy
        │
        ├──► X_validation.npy
        ├──► y_validation.npy
        │
        ├──► X_test.npy
        ├──► y_test.npy
        │
        └──► technical_writing_lstm_best.keras
                         │
                         ▼
                 text_prediction.ipynb
                         │
                         ▼
                Next-Word Suggestions
```

---

# 🧪 Example Workflow

Suppose the user enters:

```text
technical
```

The text is converted into numerical form:

```text
technical
    ↓
word index
    ↓
fixed-length sequence
```

The sequence is passed to the LSTM:

```text
Sequence
   ↓
Embedding
   ↓
LSTM
   ↓
Dense
   ↓
Softmax
```

The output might be:

```text
writing          27.72%
communication     3.46%
research          2.91%
```

The top three predictions are then displayed to the user.

---

# 📚 Educational Concepts Demonstrated

This project demonstrates several important concepts in Natural Language Processing and Deep Learning:

### Natural Language Processing

* Corpus construction
* Text cleaning
* Sentence segmentation
* Tokenization
* Vocabulary creation
* Numerical encoding

### Deep Learning

* Embedding layers
* Recurrent Neural Networks
* LSTM networks
* Dropout
* Softmax classification
* Cross-entropy loss
* Optimization

### Machine Learning

* Training data
* Validation data
* Test data
* Overfitting
* Early stopping
* Model checkpointing
* Learning-rate scheduling

### NLP Prediction

* Sequence modeling
* Next-word prediction
* Context-dependent probability
* Top-K prediction

---

# ⚠️ Limitations

This project is an educational LSTM implementation and therefore has several limitations.

### 1. Limited Corpus

The model is trained primarily on a single Technical Writing textbook.

Therefore, its language knowledge is restricted to the vocabulary and writing style found in that source.

### 2. Small-Scale Language Modeling

This is not intended to compete with modern large language models.

It demonstrates the underlying concept of sequential language prediction.

### 3. Word-Level Vocabulary

The current implementation primarily uses word-level tokenization.

Unknown or rare words may therefore be represented using:

```text
<UNK>
```

### 4. Context Window

The model uses a fixed sequence length.

Therefore, it cannot directly maintain unlimited conversational context.

### 5. Prediction Quality

Some predictions may not be semantically perfect.

This is expected from a relatively small educational corpus and LSTM architecture.

---

# 🔮 Future Improvements

Possible future extensions include:

* Larger and more diverse training corpora
* Improved text preprocessing
* Subword tokenization
* Pre-trained word embeddings
* Hyperparameter optimization
* Larger LSTM architectures
* Multiple LSTM layers
* Better evaluation metrics
* Top-K accuracy evaluation
* Perplexity measurement
* Web-based prediction interface
* Mobile-friendly predictive typing interface
* Real-time next-word suggestions
* User-specific vocabulary adaptation

These improvements can be explored independently without changing the fundamental educational objective of the project.

---

# 🧭 Recommended Development Workflow

Future development should follow:

```text
             ┌─────────────────────┐
             │  Source PDF / Data  │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Corpus Processing  │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Vocabulary & Token  │
             │      Mapping        │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Sequence Generation │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │    LSTM Training    │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Evaluation & Tests  │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │    Save Model       │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │   Inference/Test    │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Next-Word Suggestions│
             └─────────────────────┘
```

---

# 🔐 Reproducibility

For reproducible results, the following files must remain synchronized:

```text
vocabulary.json
model_config.json
X_train.npy
y_train.npy
X_validation.npy
y_validation.npy
X_test.npy
y_test.npy
technical_writing_lstm_best.keras
```

The saved model must always correspond to the vocabulary and configuration used during its training.

Changing the vocabulary without retraining the model can cause incorrect word-index mappings and invalid predictions.

---

# 📋 Project Checklist

## Data

* [x] PDF source collected
* [x] Text extracted
* [x] Corpus cleaned
* [x] Corpus saved
* [x] Corpus analyzed

## Preprocessing

* [x] Sentence segmentation
* [x] Tokenization
* [x] Vocabulary creation
* [x] Special tokens
* [x] Numerical encoding
* [x] Sequence generation

## Dataset

* [x] Training dataset
* [x] Validation dataset
* [x] Test dataset
* [x] NumPy arrays saved

## Model

* [x] Embedding layer
* [x] LSTM layer
* [x] Dropout
* [x] Dense output
* [x] Softmax
* [x] Adam optimizer
* [x] Early stopping
* [x] Learning-rate scheduling
* [x] Model checkpointing

## Prediction

* [x] Saved model loading
* [x] Vocabulary loading
* [x] Configuration loading
* [x] Input preprocessing
* [x] Next-word prediction
* [x] Top-K suggestions
* [x] Special-token filtering
* [x] Interactive user input

---

# 🎓 Academic Purpose

This project is designed primarily as an **educational implementation of Natural Language Processing using Recurrent Neural Networks**.

It demonstrates the complete lifecycle of a machine-learning project:

```text
Raw Data
   ↓
Data Processing
   ↓
Feature / Sequence Construction
   ↓
Model Development
   ↓
Training
   ↓
Evaluation
   ↓
Inference
```

The project provides a practical demonstration of how an LSTM can learn sequential relationships from natural language.

---

# 👨‍💻 Author

**Suraj Chalise**

BSc CSIT — Educational Machine Learning / NLP Project

---

# 📄 License

This project is intended for **educational and academic purposes**.

The source textbook and its contents remain subject to their respective copyright and licensing terms. The project should not be used to redistribute copyrighted source material without appropriate permission.

---

# ⭐ Final Summary

The **Technical Writing Next-Word Predictor** demonstrates how an LSTM can learn next-word relationships from a textual corpus.

The complete system can be summarized as:

```text
┌─────────────────────────────┐
│    Technical Writing PDF    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Text Extraction         │
│     & Corpus Cleaning       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Tokenization & Vocabulary   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Sequence Generation       │
│ Context → Next Word         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       LSTM Network          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Softmax Probability       │
│       Distribution          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Top-K Word Suggestions    │
└─────────────────────────────┘
```

### Core Principle

> **The purpose of the project is not simply to predict frequent words. It is to demonstrate how an LSTM learns the relationship between preceding context and the next word from a Technical Writing corpus.**

---
