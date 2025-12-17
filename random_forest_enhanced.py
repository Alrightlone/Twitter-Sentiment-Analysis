#!/usr/bin/env python3
import numpy as np
import pickle
import csv
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# === config ===
VOCAB_FILE = "vocab.pkl"
EMBEDDINGS_FILE = "embeddings.npy"
POS_TRAIN_FILE = "twitter-datasets/train_pos_full.txt"
NEG_TRAIN_FILE = "twitter-datasets/train_neg_full.txt"
TEST_DATA_FILE = "twitter-datasets/test_data.txt"
SUBMISSION_FILE = "submission_rf_enhanced.csv"

def get_enhanced_vector(text, vocab, embeddings):
    """
    we will concatenate Mean(mean value), Max(max value), Min(min value).
    if your word vector is 20 dimensional, now the feature is 60 dimensional.
    this can make the model capture the "extreme emotional words" in the sentence.
    """
    words = text.strip().split()
    vectors = [embeddings[vocab[w]] for w in words if w in vocab]
    
    if not vectors: 
        # return 3 times the length of 0 vector
        return np.zeros(embeddings.shape[1] * 3)
    
    vectors = np.array(vectors)
    
    # 1. mean value (overall semantic)
    v_mean = np.mean(vectors, axis=0)
    
    # 2. max value (most significant positive feature)
    v_max = np.max(vectors, axis=0)
    
    # 3. min value (most significant negative feature)
    v_min = np.min(vectors, axis=0)
    
    # concatenate! 20 + 20 + 20 = 60 dimensional
    return np.concatenate([v_mean, v_max, v_min])

def main():
    print("1. loading word vectors...")
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_FILE)

    print("2. preparing training data (feature concatenation Mean+Max+Min)...")
    X, y = [], []
    
    # load positive
    with open(POS_TRAIN_FILE, 'r', errors='ignore') as f:
        # for demonstration, if memory is not enough, you can add [:100000], but it's better to run the full amount!
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading Positive"):
            X.append(get_enhanced_vector(line, vocab, embeddings))
            y.append(1)
            
    # load negative
    with open(NEG_TRAIN_FILE, 'r', errors='ignore') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading Negative"):
            X.append(get_enhanced_vector(line, vocab, embeddings))
            y.append(-1) # since your RF is -1, here keep -1

    X = np.array(X)
    y = np.array(y)
    print(f"   final feature shape: {X.shape}") 
    # expected shape: (2500000, 60)

    print("3. train Enhanced Random Forest...")
    # n_estimators=100: more trees, better performance
    # min_samples_leaf=2: slightly prune to prevent overfitting, also reduce memory usage
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, verbose=1)
    clf.fit(X, y)

    print("4. predict Test set...")
    ids, X_test = [], []
    with open(TEST_DATA_FILE, 'r', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) < 2: continue
            ids.append(parts[0])
            X_test.append(get_enhanced_vector(parts[1], vocab, embeddings))
            
    y_pred = clf.predict(X_test)

    print("5. save results...")
    with open(SUBMISSION_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"])
        for i, pred in zip(ids, y_pred):
            writer.writerow([i, int(pred)])
            

if __name__ == "__main__":
    main()