#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

# === config ===
VOCAB_FILE = "vocab.pkl"
EMBEDDINGS_FILE = "embeddings.npy"
POS_FILE = "twitter-datasets/train_pos_full.txt"
NEG_FILE = "twitter-datasets/train_neg_full.txt"
TEST_FILE = "twitter-datasets/test_data.txt"
SUBMISSION_FILE = "submission_cnn.csv"

# === hyperparameters ===
MAX_LEN = 60        # sentence length
BATCH_SIZE = 128    # batch size
EPOCHS = 5          # epochs
LR = 0.001          # learning rate

# === 1. define TextCNN model ===
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, pretrained_embeddings, num_classes=2):
        super(TextCNN, self).__init__()
        
        # 1. embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # load pretrained GloVe embeddings
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        # allow fine-tuning
        self.embedding.weight.requires_grad = True 

        # 2. convolution layers (extract 3-gram, 4-gram, 5-gram features)
        # input channel=1, output channel=100
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, embed_dim)) for k in [3, 4, 5]
        ])
        
        # 3. dropout (prevent overfitting)
        self.dropout = nn.Dropout(0.5)
        
        # 4. fully connected layer
        self.fc = nn.Linear(100 * 3, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x) # [batch, seq_len, dim]
        x = x.unsqueeze(1)    # [batch, 1, seq_len, dim]
        
        # convolution + ReLU + max pooling
        # conv(x) -> [batch, 100, seq_len-k+1, 1]
        # squeeze(3) -> [batch, 100, seq_len-k+1]
        # max_pool -> [batch, 100, 1]
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        # concatenate the results of three convolution kernels
        x = torch.cat(x, 1) # [batch, 300]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# === 2. dataset processing ===
class TwitterDataset(Dataset):
    def __init__(self, X, y=None, max_len=60):
        self.X = X
        self.y = y
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # padding or truncation
        ids = self.X[idx]
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids)) # Padding 0
        else:
            ids = ids[:self.max_len]
            
        x_tensor = torch.tensor(ids, dtype=torch.long)
        
        if self.y is not None:
            # PyTorch needs 0 and 1 labels, even though the competition requires -1
            y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
            return x_tensor, y_tensor
        else:
            return x_tensor

def text_to_indices(text, vocab):
    return [vocab.get(w, 0) for w in text.strip().split()]

def main():
    # === A. hardware acceleration detection ===
    # M4 must see: use MPS (Metal Performance Shaders)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 successfully activated M4 GPU acceleration (MPS mode)")
    else:
        device = torch.device("cpu")
        print("⚠️ no MPS detected, using CPU (will be much slower)")

    # === B. load data ===
    print("1. load vocabulary and embeddings...")
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_FILE)
    vocab_size, embed_dim = embeddings.shape
    print(f"Vocab size: {vocab_size}, Embed dim: {embed_dim}")
    
    print("2. preprocess training data...")
    X_all, y_all = [], []
    
    # read Positive (Label=1)
    with open(POS_FILE, 'r', errors='ignore') as f:
        for line in f:
            X_all.append(text_to_indices(line, vocab))
            y_all.append(1) # PyTorch uses 1 internally
            
    # read Negative (Label=0) -> note: although the competition requires -1, PyTorch must start from 0 during training
    with open(NEG_FILE, 'r', errors='ignore') as f:
        for line in f:
            X_all.append(text_to_indices(line, vocab))
            y_all.append(0) # PyTorch uses 0 internally

    # split validation set (90% training, 10% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=42)
    
    train_ds = TwitterDataset(X_train, y_train, MAX_LEN)
    val_ds = TwitterDataset(X_val, y_val, MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # === C. initialize model ===
    model = TextCNN(vocab_size, embed_dim, embeddings).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # === D. training loop ===
    print(f"3. start training (total {EPOCHS} epochs)...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # training step
        for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"   Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {acc:.2f}%")

    # === E. predict Test set ===
    print("4. predict Test Data...")
    ids, X_test = [], []
    with open(TEST_FILE, 'r', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) < 2: continue
            ids.append(parts[0])
            X_test.append(text_to_indices(parts[1], vocab))
            
    test_ds = TwitterDataset(X_test, max_len=MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    all_preds = []
    model.eval()
    with torch.no_grad():
        for texts in tqdm(test_loader, desc="Predicting"):
            texts = texts.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    # === F. generate submission file ===
    print(f"5. save to {SUBMISSION_FILE}...")
    with open(SUBMISSION_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"])
        
        for i, pred in zip(ids, all_preds):
            # key: convert PyTorch's 0 back to the competition's -1
            final_label = 1 if pred == 1 else -1
            writer.writerow([i, final_label])
            
    print("🎉 done! use this TextCNN result to submit!")

if __name__ == "__main__":
    main()