#!/usr/bin/env python3
import numpy as np
import pickle
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# === config ===
VOCAB_FILE = "vocab.pkl"
EMBEDDINGS_FILE = "embeddings.npy"
POS_TRAIN_FILE = "twitter-datasets/train_pos_full.txt"
NEG_TRAIN_FILE = "twitter-datasets/train_neg_full.txt"
TEST_DATA_FILE = "twitter-datasets/test_data.txt"
SUBMISSION_FILE = "submission_lr_enhanced.csv"

def get_enhanced_vector(text, vocab, embeddings):
    """
    🔥 enhanced feature core: Mean + Max + Min Pooling
    input dimension: D (e.g. 20)
    output dimension: 3 * D (e.g. 60)
    """
    words = text.strip().split()
    vectors = [embeddings[vocab[w]] for w in words if w in vocab]
    
    if not vectors:
        # if the sentence is empty or all words are rare, return 3 times the length of 0 vector
        return np.zeros(embeddings.shape[1] * 3)
    
    vectors = np.array(vectors)
    
    # 1. mean pooling (capture overall semantic)
    v_mean = np.mean(vectors, axis=0)
    
    # 2. max pooling (capture most significant positive feature)
    v_max = np.max(vectors, axis=0)
    
    # 3. min pooling (capture most significant negative feature)
    v_min = np.min(vectors, axis=0)
    
    # concatenate
    return np.concatenate([v_mean, v_max, v_min])

def load_resources():
    print("1. Loading vocab and embeddings...")
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_FILE)
    return vocab, embeddings

def load_training_data(vocab, embeddings):
    print("2. Loading training data (Full) with Enhanced Features...")
    X = []
    y = []
    
    # load positive data (Label = 1)
    with open(POS_TRAIN_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Processing Positive"):
            vec = get_enhanced_vector(line, vocab, embeddings)
            X.append(vec)
            y.append(1)
            
    # load negative data (Label = -1)
    with open(NEG_TRAIN_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Processing Negative"):
            vec = get_enhanced_vector(line, vocab, embeddings)
            X.append(vec)
            y.append(-1) 

    return np.array(X), np.array(y)

def main():
    # A. load resources
    vocab, embeddings = load_resources()
    
    # B. prepare training data
    X_train, y_train = load_training_data(vocab, embeddings)
    print(f"   Feature Shape: {X_train.shape}") 
    # expected: (2500000, 60) if the original vector is 20 dimensional
    
    # C. data standardization (StandardScaler)
    # logistic regression is very sensitive to feature scaling, this step usually improves the score
    print("3. Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # D. train logistic regression
    print("4. Training Logistic Regression (Enhanced)...")
    # C=1.0 is the regularization strength, max_iter=1000 ensures convergence
    clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1, solver='lbfgs') 
    clf.fit(X_train, y_train)
    
    # E. predict test set
    print("5. Predicting on test set...")
    ids = []
    X_test = []
    
    with open(TEST_DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) < 2: continue
            
            tweet_id = parts[0]
            tweet_text = parts[1]
            
            vec = get_enhanced_vector(tweet_text, vocab, embeddings)
            
            ids.append(tweet_id)
            X_test.append(vec)
            
    X_test = np.array(X_test)
    
    # don't forget to standardize the test set!
    X_test = scaler.transform(X_test)
    
    y_pred = clf.predict(X_test)
    
    # F. save submission
    print(f"6. Saving submission to {SUBMISSION_FILE}...")
    with open(SUBMISSION_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"])
        
        for i, pred in zip(ids, y_pred):
            writer.writerow([i, int(pred)])
            

if __name__ == "__main__":
    main()