# 🧠 Word Predictor LSTM

<p align="center">

**Educational LSTM-Based Next-Word Text Prediction System**

A complete Natural Language Processing project that uses a **Long Short-Term Memory (LSTM)** neural network to predict the most probable next words from a **Technical Writing** corpus.

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-LSTM-red?logo=keras)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Gradio](https://img.shields.io/badge/Gradio-Web%20Interface-ff7c00)
![Git](https://img.shields.io/badge/Git-Version%20Control-black?logo=git)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)
![Git%20LFS](https://img.shields.io/badge/Git%20LFS-Large%20Files-4e9a06)

</p>

---

## 📌 Table of Contents

* [📖 Project Overview](#-project-overview)
* [🎯 Objectives](#-objectives)
* [💡 Core Concept](#-core-concept)
* [🏗️ Model Architecture](#️-model-architecture)
* [📁 Project Structure](#-project-structure)
* [📂 Directory Description](#-directory-description)
* [📓 Notebooks](#-notebooks)
* [🔄 Complete Project Workflow](#-complete-project-workflow)
* [🧹 Data Preprocessing](#-data-preprocessing)
* [📊 Corpus Analysis](#-corpus-analysis)
* [📚 Vocabulary Construction](#-vocabulary-construction)
* [🔤 Special Tokens](#-special-tokens)
* [🔢 Sequence Generation](#-sequence-generation)
* [✂️ Dataset Splitting](#️-dataset-splitting)
* [🧠 LSTM Training](#-lstm-training)
* [💾 Saved Model](#-saved-model)
* [⚙️ Model Configuration](#️-model-configuration)
* [🔮 Next-Word Prediction](#-next-word-prediction)
* [🌐 Web Prediction Interface](#-web-prediction-interface)
* [📝 Prediction Method](#-prediction-method)
* [🧪 Example Predictions](#-example-predictions)
* [🚫 What the Project Does Not Use](#-what-the-project-does-not-use)
* [📈 Common Words and Frequency Bias](#-common-words-and-frequency-bias)
* [🛠️ Technologies Used](#️-technologies-used)
* [⚙️ Installation](#️-installation)
* [▶️ Running the Project](#️-running-the-project)
* [📌 Recommended Execution Order](#-recommended-execution-order)
* [🔗 Relationship Between Project Files](#-relationship-between-project-files)
* [🧪 Example Workflow](#-example-workflow)
* [🎓 Educational Concepts Demonstrated](#-educational-concepts-demonstrated)
* [⚠️ Limitations](#️-limitations)
* [🚀 Future Improvements](#-future-improvements)
* [📏 Evaluation Considerations](#-evaluation-considerations)
* [♻️ Reproducibility](#️-reproducibility)
* [✅ Project Checklist](#-project-checklist)
* [🎓 Academic Purpose](#-academic-purpose)
* [👨‍💻 Author](#-author)
* [📄 License](#-license)
* [🏁 Final Summary](#-final-summary)

---

# 📖 Project Overview

**Word Predictor LSTM** is an educational Natural Language Processing project that implements a **next-word text prediction system using an LSTM neural network**.

The model is trained on text extracted from a **Technical Writing PDF**.

The system learns statistical relationships between words and uses the learned patterns to predict the most probable next words for a given input.

### Example

```text
Input:
technical

Prediction:
1. writing          27.72%
2. communication     3.46%
3. research          2.91%
```

Another example:

```text
Input:
is the

Prediction:
1. audience          2.27%
2. same              1.79%
3. following         1.64%
```

The project is designed primarily for **educational purposes**, demonstrating how an LSTM can be used for sequential text prediction.

---

# 🎯 Objectives

The main objectives of this project are:

* 📄 Extract text from a Technical Writing PDF.
* 🧹 Clean and preprocess the extracted text.
* 📊 Analyze the resulting corpus.
* 📚 Build a word-level vocabulary.
* 🔢 Convert words into numerical representations.
* 🔗 Generate fixed-length word sequences.
* ✂️ Split the dataset into training, validation, and testing sets.
* 🧠 Train an LSTM-based neural network.
* 💾 Save the trained model and configuration.
* 🔮 Predict the most probable next words.
* 📊 Display prediction probabilities.
* 🌐 Provide an interactive web-based prediction interface.
* 🎓 Demonstrate a complete NLP and deep-learning workflow.

---

# 💡 Core Concept

The main task of this project is **next-word prediction**.

Given a sequence of words:

```text
technical writing is
```

the model attempts to predict the word that is most likely to appear next.

The overall concept can be represented as:

```text
                 📄 Technical Writing PDF
                           │
                           ▼
                  📝 Text Extraction
                           │
                           ▼
                    🧹 Text Cleaning
                           │
                           ▼
                    📚 Vocabulary
                           │
                           ▼
                    🔢 Tokenization
                           │
                           ▼
                  🔗 Sequence Generation
                           │
                           ▼
                ✂️ Dataset Splitting
                           │
                           ▼
                     🧠 LSTM Model
                           │
                           ▼
                    🎯 Model Training
                           │
                           ▼
                     💾 Saved Model
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       📝 Text Prediction          🌐 Web Interface
```

---

# 🏗️ Model Architecture

The project uses a simple **word-level LSTM architecture**.

```text
┌─────────────────────────┐
│     Input Sequence      │
│      20 Word Tokens     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     Embedding Layer     │
│      Dimension: 128     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│       LSTM Layer        │
│       Units: 256        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│      Dropout Layer      │
│       Rate: 0.20        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│      Dense Layer        │
│      Softmax Output     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Vocabulary Probabilities│
└─────────────────────────┘
```

### ⚙️ Model Configuration

| Parameter              |                           Value |
| ---------------------- | ------------------------------: |
| 🧠 Architecture        |                            LSTM |
| 🔤 Tokenization        |                      Word-Level |
| 📏 Sequence Length     |                              20 |
| 📐 Embedding Dimension |                             128 |
| 🧠 LSTM Units          |                             256 |
| 🎲 Dropout Rate        |                            0.20 |
| 📦 Batch Size          |                             128 |
| 🔁 Maximum Epochs      |                              30 |
| 📉 Learning Rate       |                           0.001 |
| ⚙️ Optimizer           |                            Adam |
| 📊 Loss Function       | Sparse Categorical Crossentropy |
| 🎯 Output Activation   |                         Softmax |
| 📈 Metric              |                        Accuracy |

---

# 📁 Project Structure

```text
Word_Predictor_LSTM/
│
├── .venv/
│
├── configs/
│
├── data/
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
│   ├── best_lstm_model.keras
│   └── technical_writing_lstm_best.keras
│
├── notebooks/
│   ├── classification_model.ipynb
│   ├── text_prediction.ipynb
│   └── web_word_view.ipynb
│
├── report/
│
├── src/
│
├── tests/
│
├── .gitignore
├── README.md
└── LSTM_Limitations.md
```

---

# 📂 Directory Description

### 📄 `data/raw/`

Contains the original input data.

```text
Technical-Writing.pdf
```

This document is used as the primary source for creating the training corpus.

---

### 🧹 `data/processed/`

Contains processed datasets and configuration files.

Important files include:

| File                           | Purpose                    |
| ------------------------------ | -------------------------- |
| `technical_writing_corpus.txt` | Cleaned text corpus        |
| `vocabulary.json`              | Word/index mappings        |
| `model_config.json`            | Model configuration        |
| `X_train.npy`                  | Training input sequences   |
| `y_train.npy`                  | Training target words      |
| `X_validation.npy`             | Validation input sequences |
| `y_validation.npy`             | Validation targets         |
| `X_test.npy`                   | Testing input sequences    |
| `y_test.npy`                   | Testing targets            |

---

### 🧠 `models/`

Contains trained LSTM models.

```text
best_lstm_model.keras
technical_writing_lstm_best.keras
```

The `technical_writing_lstm_best.keras` model is used by the prediction notebooks.

---

### 📓 `notebooks/`

Contains the main Jupyter notebooks:

```text
classification_model.ipynb
text_prediction.ipynb
web_word_view.ipynb
```

---

### 📑 `report/`

Reserved for project reports, documentation, diagrams, results, and academic material.

---

### 🧩 `src/`

Reserved for reusable Python source code as the project is further modularized.

---

### 🧪 `tests/`

Reserved for project testing and validation code.

---

# 📓 Notebooks

## 1️⃣ `classification_model.ipynb`

This is the **main training and model-development notebook**.

It contains:

* PDF extraction
* Corpus analysis
* Corpus statistics
* Sentence-length analysis
* Word-frequency analysis
* Vocabulary statistics
* Repeated-line analysis
* Numeric-line analysis
* Short-line analysis
* Corpus cleaning
* Sentence segmentation
* Vocabulary creation
* Special-token creation
* Tokenization
* Sequence generation
* Sequence-length analysis
* Dataset splitting
* LSTM construction
* Model training
* Validation
* Early stopping
* Model checkpointing
* Learning-rate reduction
* Best-model loading
* Model configuration saving

The final trained model is saved for later prediction.

---

## 2️⃣ `text_prediction.ipynb`

This notebook is used for **direct interactive testing of the trained model**.

It:

* Loads the trained LSTM.
* Loads the vocabulary.
* Loads the model configuration.
* Preprocesses user text.
* Converts words into indices.
* Creates the required input sequence.
* Performs model inference.
* Ranks the output probabilities.
* Displays the top three predicted words.

Example:

```text
Input:
technical

Output:
1. writing          27.72%
2. communication     3.46%
3. research          2.91%
```

This notebook is useful for testing the model without launching a web interface.

---

## 3️⃣ `web_word_view.ipynb`

This notebook provides a **web-based interface for the trained LSTM model**.

It uses **Gradio**.

The interface allows the user to:

```text
Enter Text
    ↓
Submit
    ↓
LSTM Prediction
    ↓
Top-3 Next Words
    ↓
Prediction Probabilities
```

The notebook does **not retrain the model**.

It directly loads:

```text
models/technical_writing_lstm_best.keras
data/processed/vocabulary.json
data/processed/model_config.json
```

---

# 🔄 Complete Project Workflow

```text
                 📄 Raw PDF
                    │
                    ▼
            📝 Text Extraction
                    │
                    ▼
             📊 Corpus Analysis
                    │
                    ▼
              🧹 Corpus Cleaning
                    │
                    ▼
             📚 Vocabulary
                    │
                    ▼
             🔢 Tokenization
                    │
                    ▼
           🔗 Sequence Generation
                    │
                    ▼
          ✂️ Dataset Splitting
                    │
                    ▼
              🧠 LSTM Training
                    │
                    ▼
              💾 Saved Model
                    │
          ┌─────────┴──────────┐
          │                    │
          ▼                    ▼
  📝 Text Prediction     🌐 Web Prediction
  text_prediction.ipynb  web_word_view.ipynb
```

---

# 🧹 Data Preprocessing

The original PDF is processed to obtain usable textual data.

The preprocessing workflow includes:

1. 📄 Extract text from the PDF.
2. 🧹 Remove unnecessary extraction artifacts.
3. 📊 Analyze the extracted corpus.
4. 🧹 Remove unsuitable lines and noise.
5. 🔗 Segment text into sentences.
6. 📚 Build the vocabulary.
7. 🔢 Convert words into numerical indices.
8. 🔗 Generate training sequences.

The project focuses on **textual information** from the document.

Images and page numbers are not used as prediction inputs.

---

# 📊 Corpus Analysis

Before model training, the corpus is analyzed to understand its characteristics.

The notebook analyzes:

* 📏 Total lines
* 🔤 Total words
* 📚 Vocabulary size
* 📊 Word frequencies
* 📐 Sentence lengths
* 🔁 Repeated lines
* 🔢 Numeric-only lines
* 📄 Short lines
* 📈 Sequence-length distribution

This helps identify potential problems in the extracted PDF text before training.

---

# 📚 Vocabulary Construction

The project uses a **word-level vocabulary**.

Each unique word is assigned an integer index.

For example:

```text
technical  → 125
writing    → 86
research   → 314
```

Two mappings are stored:

```text
word_to_index
```

and:

```text
index_to_word
```

These mappings allow the system to convert:

```text
Word → Number
```

and:

```text
Number → Word
```

---

# 🔤 Special Tokens

The project uses four special tokens:

| Token     | Purpose                         |
| --------- | ------------------------------- |
| `<PAD>`   | Padding shorter sequences       |
| `<UNK>`   | Unknown/out-of-vocabulary words |
| `<START>` | Sentence beginning              |
| `<END>`   | Sentence ending                 |

During prediction, these special tokens are filtered from the final suggestions so that the user receives actual words.

---

# 🔢 Sequence Generation

The model uses:

```text
SEQUENCE_LENGTH = 20
```

Training data is converted into:

```text
Input Sequence → Target Word
```

For example:

```text
Word 1
Word 2
Word 3
...
Word 20
       ↓
   Target Word
```

This teaches the LSTM to estimate:

> Given the previous words, what word is most likely to come next?

---

# ✂️ Dataset Splitting

The dataset is separated into:

```text
┌───────────────────┐
│   Training Set    │
└───────────────────┘

┌───────────────────┐
│ Validation Set    │
└───────────────────┘

┌───────────────────┐
│     Test Set      │
└───────────────────┘
```

The training set is used to learn model parameters.

The validation set is used to monitor performance during training.

The test set is reserved for evaluating the trained model.

The project performs the split at the **sentence level**.

---

# 🧠 LSTM Training

The model is trained using:

### Optimizer

```text
Adam
```

### Loss Function

```text
Sparse Categorical Crossentropy
```

### Metric

```text
Accuracy
```

### Callbacks

The training process uses:

* ⏹️ Early Stopping
* 💾 Model Checkpoint
* 📉 Reduce Learning Rate on Plateau

### Early Stopping

```text
patience = 5
restore_best_weights = True
```

### Reduce Learning Rate

```text
factor = 0.5
patience = 2
min_lr = 1e-6
```

These techniques help control training and reduce unnecessary epochs.

---

# 💾 Saved Model

The best trained model is saved as:

```text
models/technical_writing_lstm_best.keras
```

A checkpoint is also maintained as:

```text
models/best_lstm_model.keras
```

The prediction notebooks load the saved model instead of retraining it.

---

# ⚙️ Model Configuration

The important model configuration is stored in:

```text
data/processed/model_config.json
```

This allows the prediction notebooks to use the same configuration used during training.

For example:

```text
sequence_length
vocabulary_size
```

are stored so that the inference process remains consistent with the trained model.

---

# 🔮 Next-Word Prediction

The prediction system accepts text from the user.

For example:

```text
technical
```

The system performs the following:

```text
User Text
   ↓
Lowercase Conversion
   ↓
Tokenization
   ↓
Word → Index
   ↓
Sequence Length Adjustment
   ↓
Padding / Truncation
   ↓
LSTM Model
   ↓
Probability Distribution
   ↓
Probability Ranking
   ↓
Top-3 Suggestions
```

---

# 🌐 Web Prediction Interface

The project includes a simple web interface built using **Gradio**.

The interface is available through:

```text
notebooks/web_word_view.ipynb
```

The user enters text into a textbox and receives the three most probable next words.

### Interface Workflow

```text
┌────────────────────────────┐
│       Enter Your Text      │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│     Text Preprocessing     │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│       LSTM Inference       │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│   Probability Distribution │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│   Top-3 Word Suggestions   │
└────────────────────────────┘
```

The interface is intended for demonstration and educational use.

---

# 📝 Prediction Method

The model predicts a probability for every word in its vocabulary.

The predicted probabilities are sorted in descending order.

For example:

```text
writing          0.2772
communication    0.0346
research         0.0291
```

The system then displays:

```text
1. writing          27.72%
2. communication     3.46%
3. research          2.91%
```

Special tokens such as `<PAD>` and `<UNK>` are excluded from the displayed suggestions.

---

# 🧪 Example Predictions

### Example 1 — `technical`

```text
Input:
technical
```

```text
1. writing          27.72%
2. communication     3.46%
3. research          2.91%
```

---

### Example 2 — `writing`

```text
Input:
writing
```

```text
1. the               8.87%
2. a                 6.13%
3. p                 3.95%
```

---

### Example 3 — `is the`

```text
Input:
is the
```

```text
1. audience          2.27%
2. same              1.79%
3. following         1.64%
```

---

### Example 4 — `the best way`

```text
Input:
the best way
```

```text
1. you               12.42%
2. the               10.73%
3. to                10.00%
```

---

### Example 5 — `you are capable of`

```text
Input:
you are capable of
```

```text
1. the               27.91%
2. a                 10.79%
3. your               4.20%
```

> **Note:** The exact predictions and probabilities depend on the trained model and input context.

---

# 🚫 What the Project Does Not Use

This project intentionally uses a relatively simple LSTM-based architecture.

It does **not** use:

* ❌ Transformers
* ❌ GPT
* ❌ BERT
* ❌ Large Language Models
* ❌ Attention mechanisms
* ❌ Pretrained language models
* ❌ Subword tokenization
* ❌ External word embeddings
* ❌ Internet-based knowledge
* ❌ Retrieval-Augmented Generation

The purpose is to demonstrate the fundamentals of **LSTM-based sequential text prediction**.

---

# 📈 Common Words and Frequency Bias

Because the model is trained on a limited corpus, frequently occurring words may receive higher probabilities.

Examples include:

```text
the
a
to
of
is
and
```

This is expected because the model learns statistical patterns from the training data.

The model does not simply choose random words. It produces a probability distribution based on the patterns learned during training.

---

# 🛠️ Technologies Used

| Technology          | Purpose                   |
| ------------------- | ------------------------- |
| 🐍 Python           | Main programming language |
| 🧠 TensorFlow       | Deep learning framework   |
| 🔥 Keras            | LSTM model implementation |
| 🔢 NumPy            | Numerical processing      |
| 📊 Matplotlib       | Data visualization        |
| 📓 Jupyter Notebook | Development environment   |
| 📄 PDF Processing   | Text extraction           |
| 🌐 Gradio           | Web prediction interface  |
| 🔧 Git              | Version control           |
| 🐙 GitHub           | Repository hosting        |
| 📦 Git LFS          | Large file management     |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/chalise-suraj/Word_Predictor_LSTM.git
```

## 2. Enter the Project Directory

```bash
cd Word_Predictor_LSTM
```

## 3. Create Virtual Environment

```bash
python -m venv .venv
```

## 4. Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 5. Install Required Packages

```bash
pip install tensorflow numpy matplotlib jupyter gradio
```

Additional packages required by the PDF extraction implementation may also need to be installed depending on the notebook environment.

---

# ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/
```

The project contains three main notebooks:

```text
classification_model.ipynb
text_prediction.ipynb
web_word_view.ipynb
```

---

# 📌 Recommended Execution Order

## 1️⃣ Train the Model

Open:

```text
classification_model.ipynb
```

Run the notebook to:

* Extract the PDF text.
* Analyze the corpus.
* Clean the corpus.
* Build the vocabulary.
* Generate sequences.
* Split the dataset.
* Train the LSTM.
* Save the best model.
* Save the model configuration.

---

## 2️⃣ Test Predictions

Open:

```text
text_prediction.ipynb
```

Use it to directly test different input sentences and observe the model's predictions.

---

## 3️⃣ Launch the Web Interface

Open:

```text
web_word_view.ipynb
```

Run the notebook and launch the Gradio interface.

Then enter text into the web interface to receive the top-three next-word predictions.

---

# 🔗 Relationship Between Project Files

The main project components are connected as follows:

```text
                    📄 Technical-Writing.pdf
                              │
                              ▼
                  classification_model.ipynb
                              │
                              ▼
                     🧹 Processed Corpus
                              │
                              ▼
                     📚 Vocabulary
                              │
                              ▼
                     🔢 Training Sequences
                              │
                              ▼
                         🧠 LSTM
                              │
                              ▼
                     💾 Saved Model
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
       text_prediction.ipynb       web_word_view.ipynb
                │                           │
                ▼                           ▼
       📝 Direct Prediction          🌐 Web Prediction
```

---

# 🧪 Example Workflow

A complete user workflow looks like this:

```text
1. 📄 Place Technical-Writing.pdf
       ↓
2. 🧠 Run classification_model.ipynb
       ↓
3. 🧹 Process the corpus
       ↓
4. 📚 Build vocabulary
       ↓
5. 🔢 Generate sequences
       ↓
6. ✂️ Split datasets
       ↓
7. 🧠 Train LSTM
       ↓
8. 💾 Save trained model
       ↓
9. 📝 Run text_prediction.ipynb
       ↓
10. 🔮 Test next-word prediction
       ↓
11. 🌐 Run web_word_view.ipynb
       ↓
12. 🖥️ Open Gradio interface
       ↓
13. ✍️ Enter text
       ↓
14. 🎯 View Top-3 predictions
```

---

# 🎓 Educational Concepts Demonstrated

## Natural Language Processing

* 📄 Text extraction
* 🧹 Text cleaning
* 🔤 Tokenization
* 📚 Vocabulary construction
* 🔢 Numerical representation
* 🔗 Sequence generation

## Deep Learning

* 🧠 Embedding
* 🔄 Recurrent Neural Networks
* 🧠 LSTM
* 🎲 Dropout
* 🔗 Dense layers
* 📊 Softmax classification

## Model Training

* ✂️ Dataset splitting
* 📉 Loss functions
* ⚙️ Optimizers
* ⏹️ Early stopping
* 💾 Model checkpointing
* 📉 Learning-rate scheduling

## Model Inference

* 🔤 Input preprocessing
* 🔢 Sequence preparation
* 🧠 Model prediction
* 📊 Probability ranking
* 🎯 Top-k prediction

## Application Development

* 💾 Saved model loading
* 🔄 Reusable inference
* 📝 Interactive prediction
* 🌐 Gradio web interface

---

# ⚠️ Limitations

This project is primarily an **educational implementation** and therefore has several limitations.

### 📚 1. Small Training Corpus

The model is trained on a relatively limited amount of text.

A larger and more diverse corpus would provide more language patterns.

### 📖 2. Narrow Domain

The training data comes from Technical Writing material.

Therefore, the model is not designed to be a general-purpose language model.

### 🔤 3. Word-Level Tokenization

The model uses word-level tokenization.

Words outside the vocabulary are represented using:

```text
<UNK>
```

This creates an out-of-vocabulary limitation.

### 📏 4. Fixed Context Length

The model uses:

```text
SEQUENCE_LENGTH = 20
```

Therefore, only the most recent 20 tokens are considered when the input is longer than the allowed sequence length.

### 🔄 5. LSTM Sequential Limitation

LSTM can model sequential dependencies better than a basic RNN, but it still has limitations when dealing with very long-range dependencies.

### 🎯 6. Ambiguous Predictions

A sentence can have many grammatically or semantically valid next words.

Therefore, a low-probability prediction is not necessarily an incorrect word in natural language.

### 📊 7. Frequency Bias

Common words may receive higher probabilities because of their frequency in the training corpus.

### 🧠 8. Limited Model Capacity

The current project uses a relatively simple single-layer LSTM architecture.

### 🌎 9. Limited Generalization

The model is not expected to perform like modern general-purpose language models.

### 🌐 10. No External Knowledge

The model has no connection to external knowledge sources or internet search.

### 👤 11. No Personalization

The model does not learn an individual user's writing style.

### 🔄 12. No Continuous Learning

The current system does not automatically update its model based on newly entered text.

A detailed discussion of these limitations is available in:

```text
LSTM_Limitations.md
```

---

# 🚀 Future Improvements

Possible future improvements include:

* 📚 Increase the size of the training corpus.
* 📖 Use multiple Technical Writing documents.
* 🧹 Improve PDF text extraction.
* 🔤 Introduce subword tokenization.
* 🧠 Experiment with pretrained embeddings.
* 🧠 Experiment with stacked LSTM layers.
* ⚙️ Perform hyperparameter tuning.
* 📏 Experiment with different sequence lengths.
* 📊 Add Top-1, Top-3, and Top-5 evaluation.
* 📉 Calculate perplexity.
* 📈 Calculate Mean Reciprocal Rank.
* 🧪 Improve test-set evaluation.
* 🌐 Improve the web interface.
* 📝 Add prediction history.
* ⚙️ Add configurable prediction settings.
* 🔬 Compare multiple architectures.
* 🧠 Explore attention-based architectures.
* 🤖 Compare the LSTM approach with Transformer-based models.

> The web-based prediction interface is already implemented through `web_word_view.ipynb`; future work can focus on improving its functionality and user experience.

---

# 📏 Evaluation Considerations

Next-word prediction can be evaluated using several metrics.

Possible metrics include:

| Metric            | Purpose                                   |
| ----------------- | ----------------------------------------- |
| 🎯 Top-1 Accuracy | Whether the correct word is ranked first  |
| 🎯 Top-3 Accuracy | Whether the correct word appears in top 3 |
| 🎯 Top-5 Accuracy | Whether the correct word appears in top 5 |
| 📉 Cross-Entropy  | Measures prediction loss                  |
| 📊 Perplexity     | Measures language-model uncertainty       |
| 🔎 MRR            | Measures ranking quality                  |

These metrics can provide a more complete evaluation than accuracy alone.

---

# ♻️ Reproducibility

The project stores important artifacts required for inference.

### Vocabulary

```text
data/processed/vocabulary.json
```

### Configuration

```text
data/processed/model_config.json
```

### Trained Model

```text
models/technical_writing_lstm_best.keras
```

### Processed Dataset

```text
data/processed/
```

The saved files allow the prediction notebooks to load the trained model without repeating the entire training process.

---

# ✅ Project Checklist

## 📄 Data Processing

* [x] PDF text extraction
* [x] Corpus analysis
* [x] Corpus cleaning
* [x] Sentence segmentation
* [x] Vocabulary construction
* [x] Tokenization
* [x] Sequence generation
* [x] Sequence-length analysis

## 🧠 Model Development

* [x] Embedding layer
* [x] LSTM architecture
* [x] Dropout
* [x] Dense softmax output
* [x] Training
* [x] Validation
* [x] Early stopping
* [x] Model checkpointing
* [x] Learning-rate reduction
* [x] Best model saving
* [x] Model configuration saving

## 🔮 Prediction

* [x] Saved model loading
* [x] Vocabulary loading
* [x] Configuration loading
* [x] User text preprocessing
* [x] Sequence preparation
* [x] Next-word prediction
* [x] Top-3 suggestions
* [x] Prediction probabilities

## 🌐 Web Prediction

* [x] Web prediction notebook
* [x] Gradio interface
* [x] User text input
* [x] LSTM inference
* [x] Top-3 predictions
* [x] Prediction probabilities

## 📚 Documentation

* [x] README
* [x] LSTM limitations document
* [x] Notebook documentation
* [x] Project structure
* [x] Workflow documentation

---

# 🎓 Academic Purpose

This project is designed primarily for **educational and academic purposes**.

It demonstrates how a traditional recurrent neural network architecture such as **LSTM** can be applied to a practical NLP problem.

The project provides an end-to-end understanding of:

```text
Raw Data
   ↓
Data Processing
   ↓
Text Representation
   ↓
Sequence Generation
   ↓
Deep Learning
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Inference
   ↓
Application Interface
```

It is particularly useful for understanding the relationship between:

* Natural Language Processing
* Machine Learning
* Deep Learning
* Recurrent Neural Networks
* LSTM
* Word Embeddings
* Sequence Modeling
* Next-Word Prediction
* Model Inference
* Web-Based AI Applications

---

# 👨‍💻 Author

**Suraj Chalise**

---

# 📄 License

This project is intended for **educational and academic use**.

Please refer to the repository for the applicable license and usage information.

---

# 🏁 Final Summary

**Word Predictor LSTM** is an end-to-end educational NLP project that implements a **word-level next-word prediction system using an LSTM neural network**.

The project starts with a Technical Writing PDF and follows a complete machine-learning workflow:

```text
📄 Technical Writing PDF
        ↓
📝 Text Extraction
        ↓
🧹 Corpus Cleaning
        ↓
📊 Corpus Analysis
        ↓
📚 Vocabulary Construction
        ↓
🔢 Tokenization
        ↓
🔗 Sequence Generation
        ↓
✂️ Dataset Splitting
        ↓
🧠 LSTM Training
        ↓
💾 Model Saving
        ↓
🔮 Next-Word Prediction
        ↓
🌐 Web-Based Interface
```

The trained model can be used in two ways:

### 📝 Direct Prediction

```text
text_prediction.ipynb
```

for experimenting directly with text input.

### 🌐 Web Prediction

```text
web_word_view.ipynb
```

for interacting with the trained model through a Gradio-based interface.

The project intentionally focuses on **LSTM rather than Transformer-based architectures**, making it suitable for studying the fundamentals of sequential neural networks and NLP.

Despite its limitations, the project provides a complete practical example of how raw textual data can be transformed into a trained neural network and finally deployed as an interactive prediction application.

---

<p align="center">

### 🧠 Learn → Build → Train → Predict → Deploy

**Word Predictor LSTM**

</p>
