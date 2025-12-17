#!/usr/bin/env python3
import numpy as np
import pickle
import csv
from sklearn.linear_model import LogisticRegression

# === config ===
VOCAB_FILE = "vocab.pkl"
EMBEDDINGS_FILE = "embeddings.npy"
POS_TRAIN_FILE = "twitter-datasets/train_pos_full.txt"
NEG_TRAIN_FILE = "twitter-datasets/train_neg_full.txt"
TEST_DATA_FILE = "twitter-datasets/test_data.txt"
SUBMISSION_FILE = "submission_lr.csv"

def load_resources():
    print("1. Loading vocab and embeddings...")
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_FILE)
    return vocab, embeddings

def get_sentence_vector(text, vocab, embeddings):
    """
    simplest linear method:
    find the vector of each word in the sentence, then take the average (Mean Pooling)
    """
    words = text.strip().split()
    vectors = []
    
    for w in words:
        if w in vocab:
            idx = vocab[w]
            vectors.append(embeddings[idx])
            
    if not vectors:
        # if all words are rare, return all 0 vector
        return np.zeros(embeddings.shape[1])
    
    # convert to numpy array and take the average
    return np.mean(vectors, axis=0)

def load_training_data(vocab, embeddings):
    print("2. Loading training data...")
    X = []
    y = []
    
    # load positive data (Label = 1)
    with open(POS_TRAIN_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            vec = get_sentence_vector(line, vocab, embeddings)
            X.append(vec)
            y.append(1)
            
    # load negative data (Label = -1)
    # note: usually these competitions require negative labels to be -1, if it's 0, please modify it yourself
    with open(NEG_TRAIN_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            vec = get_sentence_vector(line, vocab, embeddings)
            X.append(vec)
            y.append(-1) 

    return np.array(X), np.array(y)

def create_submission(clf, vocab, embeddings):
    print("4. Predicting on test set...")
    
    ids = []
    X_test = []
    
    with open(TEST_DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # test_data.txt format usually is: "id,text"
            # we need to split by the first comma, and get the ID and text
            parts = line.strip().split(',', 1)
            if len(parts) < 2: continue # skip bad data
            
            tweet_id = parts[0]
            tweet_text = parts[1]
            
            vec = get_sentence_vector(tweet_text, vocab, embeddings)
            
            ids.append(tweet_id)
            X_test.append(vec)
            
    X_test = np.array(X_test)
    
    # predict
    y_pred = clf.predict(X_test)
    
    # write to CSV
    print(f"5. Saving submission to {SUBMISSION_FILE}...")
    with open(SUBMISSION_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"]) # header
        
        for i, pred in zip(ids, y_pred):
            writer.writerow([i, int(pred)])
            

def main():
    # A. load resources
    vocab, embeddings = load_resources()
    
    # B. prepare training data
    X_train, y_train = load_training_data(vocab, embeddings)
    print(f"   Training data shape: {X_train.shape}")
    
    # C. train logistic regression (simplest linear model)
    print("3. Training Logistic Regression...")
    # max_iter=1000 to prevent warnings, n_jobs=-1 to use M4's multiple cores
    clf = LogisticRegression(max_iter=1000, n_jobs=-1) 
    clf.fit(X_train, y_train)
    
    # D. generate prediction file
    create_submission(clf, vocab, embeddings)

if __name__ == "__main__":
    main()