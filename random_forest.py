#!/usr/bin/env python3
import numpy as np
import pickle
import csv
from sklearn.ensemble import RandomForestClassifier

# === config ===
VOCAB_FILE = "vocab.pkl"
EMBEDDINGS_FILE = "embeddings.npy"
POS_TRAIN_FILE = "twitter-datasets/train_pos_full.txt"
NEG_TRAIN_FILE = "twitter-datasets/train_neg_full.txt"
TEST_DATA_FILE = "twitter-datasets/test_data.txt"
SUBMISSION_FILE = "submission_rf.csv"

def load_resources():
    print("Loading vocab and embeddings...")
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_FILE)
    return vocab, embeddings

def get_sentence_vector(text, vocab, embeddings):
    # still use the simplest average, let's see if the model is effective
    words = text.strip().split()
    vectors = [embeddings[vocab[w]] for w in words if w in vocab]
    if not vectors: return np.zeros(embeddings.shape[1])
    return np.mean(vectors, axis=0)

def main():
    vocab, embeddings = load_resources()
    
    # 1. load data
    print("Loading Training Data...")
    X, y = [], []
    
    # load positive
    with open(POS_TRAIN_FILE, 'r', errors='ignore') as f:
        # for demonstration, if memory is not enough, you can add [:50000], but it's better to run the full amount!
        for line in f: 
            X.append(get_sentence_vector(line, vocab, embeddings))
            y.append(1)
            
    # load negative
    with open(NEG_TRAIN_FILE, 'r', errors='ignore') as f:
        for line in f:
            X.append(get_sentence_vector(line, vocab, embeddings))
            y.append(-1) # since your RF is -1, here keep -1

    X = np.array(X)
    y = np.array(y)
    
    # 2. train random forest
    print("Training Random Forest (this uses all CPU cores)...")
    # n_estimators=100: more trees, better performance
    # n_jobs=-1: use all M4's cores
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1)
    clf.fit(X, y)
    
    # 3. predict test set
    print("Predicting Test Data...")
    ids, X_test = [], []
    with open(TEST_DATA_FILE, 'r', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) < 2: continue
            ids.append(parts[0])
            X_test.append(get_sentence_vector(parts[1], vocab, embeddings))
            
    y_pred = clf.predict(X_test)
    
    # 4. save results
    with open(SUBMISSION_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"])
        for i, pred in zip(ids, y_pred):
            writer.writerow([i, int(pred)])
            

if __name__ == "__main__":
    main()