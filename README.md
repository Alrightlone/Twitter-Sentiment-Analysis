

# 💬 From Statistical Pooling to Large-Scale Transformers
### A Comprehensive Study on Twitter Sentiment Analysis

<a href="https://www.westlake.edu.cn/">
    <img src="https://img.shields.io/badge/Institution-Westlake%20University-005BAC?style=for-the-badge&logo=academy&logoColor=white" alt="Westlake University">
</a>
<a href="https://alrightlone.github.io/">
    <img src="https://img.shields.io/badge/Author-Junhan%20Zhu-black?style=for-the-badge&logo=github&logoColor=white" alt="Junhan Zhu">
</a>
<br>

<img src="https://img.shields.io/badge/Student%20ID-20233121004-gray?style=flat-square&logo=id-card">
<a href="mailto:zhujunhan@westlake.edu.cn">
    <img src="https://img.shields.io/badge/Email-zhujunhan%40westlake.edu.cn-D14836?style=flat-square&logo=gmail&logoColor=white">
</a>
<img src="https://img.shields.io/badge/School-School%20of%20Engineering-blue?style=flat-square">

<br>
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

</div>

---


## Overview

This project presents a systematic study of sentiment classification methodologies on a large-scale Twitter dataset (~2.5 million tweets). The study evolves from traditional machine learning baselines to state-of-the-art Deep Learning architectures. 

We investigate the limitations of static word embeddings (GloVe) and propose a **Feature Enhancement Strategy** (Multi-View Pooling) that significantly boosts the performance of shallow classifiers. Furthermore, we implement a **TextCNN** to capture local N-gram features and finally leverage **Transfer Learning** by fine-tuning a **DeBERTa-v3-Large** model on an NVIDIA RTX 4090.

**Key Achievements:**
* Proposed a **Mean+Max+Min pooling strategy** that improved traditional models by ~8%.
* Achieved **State-of-the-Art (SOTA)** performance with DeBERTa-v3-Large (F1-Score: 0.896).
* Optimized training on consumer hardware using **BF16 mixed-precision** and **gradient accumulation**.

## Project Structure

```text
project_text_classification/
├── twitter-datasets/              # Dataset directory
│   ├── train_pos_full.txt         # Full positive training data (1.25M)
│   ├── train_neg_full.txt         # Full negative training data (1.25M)
│   ├── test_data.txt              # Unlabeled test data (10k)
│   └── sample_submission.csv      # Submission format
│
├── Word Embedding Generation
│   ├── build_vocab.sh             # Vocabulary construction
│   ├── cut_vocab.sh               # Frequency filtering (min_count=5)
│   ├── pickle_vocab.py            # Vocabulary serialization
│   ├── cooc.py                    # Co-occurrence matrix construction
│   ├── glove_template.py          # Custom GloVe training (SGD)
│   └── embeddings.npy             # Trained word embeddings (d=20 or 200)
│
├── Machine Learning Models (CPUs/M4)
│   ├── logistic.py                # Baseline Logistic Regression
│   ├── logistic_enhance.py        # Enhanced LR (Multi-View Pooling)
│   ├── random_forest.py           # Baseline Random Forest
│   └── random_forest_enhanced.py  # Enhanced RF (Multi-View Pooling)
│
├── Deep Learning Models (GPUs)
│   ├── text_cnn.py                # TextCNN (Parallel Convolutions)
│   └── deberta_finetuning.py      # DeBERTa-v3-Large Fine-tuning
│
└── Submission Files
    ├── submission_deberta.csv     # [FINAL SUBMISSION] Best Performance
    └── ... (other intermediate submissions)
```
## Environment Setup
This project utilizes a heterogeneous hardware environment: Apple M4 for statistical models and embedding generation, and NVIDIA RTX 4090 for Transformer fine-tuning.
```python
# Core dependencies
pip install numpy scipy scikit-learn tqdm pandas matplotlib seaborn

# Deep Learning dependencies
pip install torch torchvision              # For TextCNN (MPS/CUDA)
pip install transformers modelscope        # For DeBERTa (Hugging Face)
pip install accelerate                     # For mixed-precision training
```

## Quick Usage

### Step 1: Generate GloVe Word Embeddings
```bash
# Build vocabulary from training data (min frequency = 5)
bash build_vocab.sh
bash cut_vocab.sh
python3 pickle_vocab.py

# Construct co-occurrence matrix
python3 cooc.py

# Train GloVe embeddings (SGD-based matrix factorization)
python3 glove_template.py
```

### Step 2: Train and Evaluate Models

**Machine Learning Baselines (CPU/M4):**
```bash
# Baseline Logistic Regression
python3 logistic.py

# Enhanced Logistic Regression (Multi-View Pooling)
python3 logistic_enhance.py

# Baseline Random Forest
python3 random_forest.py

# Enhanced Random Forest (Multi-View Pooling)
python3 random_forest_enhanced.py
```

**Deep Learning Models (GPU Required):**
```bash
# TextCNN (Apple MPS or CUDA)
python3 text_cnn.py

# DeBERTa-v3-Large Fine-tuning (CUDA recommended)
python3 deberta_finetuning.py
```

### Step 3: Submit Predictions
```bash
# All models generate submission_*.csv files automatically
# Select the best performing model (e.g., submission_deberta.csv)
```

---

## Methodology

We adopted a progressive approach to sentiment classification, evolving from statistical feature engineering to deep representation learning.

### 1. Statistical Baselines & Feature Enhancement
We started with **Logistic Regression** and **Random Forest** using pre-trained **GloVe word embeddings**.
* **Problem:** Standard "Mean Pooling" of word vectors tends to dilute critical sentiment signals (e.g., a single strong negative word like "terrible" in a long neutral sentence).
* **Solution (Feature Enhancement):** We implemented a **Multi-View Pooling Strategy**. Instead of just averaging word vectors, we concatenated three statistical views:
    * `Mean Pooling`
    * `Max Pooling`
    * `Min Pooling`
    * **Result:** The input dimension increased from $d$ to $3d$.

### 2. Deep Learning: TextCNN
To capture local dependencies and phrase-level patterns (N-grams), we implemented a **TextCNN** architecture:
* **Embedding Layer:** Initialized with pre-trained GloVe vectors (fine-tunable).
* **Convolution:** Parallel filters of window sizes **3, 4, and 5** to simulate tri-gram to five-gram extraction.
* **Pooling:** Max-over-time pooling to isolate the most salient features regardless of their position in the tweet.

### 3. Transfer Learning: DeBERTa-v3-Large
We employed the **DeBERTa-v3-Large** model as our state-of-the-art candidate. Unlike standard BERT, DeBERTa uses a disentangled attention mechanism, making it highly effective for the complex context of tweets.
* **Training Strategy:** We fine-tuned the model on a balanced subset of **200,000 tweets** (100k positive / 100k negative).
* **Optimization:** To train this large model on a consumer GPU (RTX 4090), we utilized **BF16 mixed-precision training** and **Gradient Accumulation** (steps=32) to simulate a large batch size while minimizing memory footprint.

---

## 📊 Results and Performance Comparison

We evaluated all models on the held-out test set (10,000 samples). The table below summarizes the performance evolution. The introduction of the "Enhanced" pooling strategy yielded significant gains for traditional models, while the Transformer architecture achieved the best overall results.

| Model Architecture | Feature Strategy | Accuracy | F1-Score | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Logistic Regression** | Baseline (Mean Pooling) | 0.532 | 0.560 | |
| **Logistic Regression** | **Enhanced (Mean+Max+Min)** | **0.612** | **0.640** | *+8.0% gain* |
| **Random Forest** | Baseline (Mean Pooling) | 0.638 | 0.633 | |
| **Random Forest** | **Enhanced (Mean+Max+Min)** | **0.723** | **0.739** | *+8.5% gain* |
| **TextCNN** | Learnable N-grams | 0.860 | 0.861 | Strong Baseline |
| **DeBERTa-v3-Large** | **Fine-Tuned Transformer** | **0.895** | **0.896** | **🏆 SOTA** |

> **Key Insight:** The 8%+ improvement in LR and RF demonstrates that simple feature engineering (Max/Min pooling) can significantly bridge the gap when compute resources are limited. However, for maximum accuracy, DeBERTa's deep contextual understanding is superior.

---

## 🚀 Final Submission

Based on the comparative analysis above, we selected the predictions from the fine-tuned Transformer model for the final submission.

* **Selected File:** `submission_deberta.csv`
* **Performance:** **Accuracy 0.895 / F1 0.896**
* **Justification:** This model demonstrates the strongest generalization capability.