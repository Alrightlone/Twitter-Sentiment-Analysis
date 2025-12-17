import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from modelscope import snapshot_download

# ================= config =================
# 1. model path
print("locating local model path...")
MODEL_DIR = snapshot_download('microsoft/deberta-v3-large')
print(f"✅ model path: {MODEL_DIR}")

 
# 2. hyperparameters
MAX_LEN = 128
BATCH_SIZE = 32       
GRAD_ACCUMULATION = 32  
EPOCHS = 3
LR = 8e-6             

# 3. file path
POS_FILE = "/text_classification/project_text_classification/twitter-datasets/train_pos.txt"
NEG_FILE = "/text_classification/project_text_classification/twitter-datasets/train_neg.txt"
TEST_FILE = "/text_classification/project_text_classification/twitter-datasets/test_data.txt"
SUBMISSION_FILE = "/text_classification/project_text_classification/submission_deberta.csv"
# ===========================================

class TwitterDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def load_data():
    print("loading raw data...")
    texts, labels = [], []
    
    # read positive data (Label = 1)
    with open(POS_FILE, 'r', errors='ignore') as f:
        for line in f:
            texts.append(line.strip())
            labels.append(1)
            
    # read negative data (Label = 0) -> note: Transformer training uses 0 temporarily, and then converted back to -1 at the end
    with open(NEG_FILE, 'r', errors='ignore') as f:
        for line in f:
            texts.append(line.strip())
            labels.append(0)
            
    return texts, labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def main():
    # check GPU
    if torch.cuda.is_available():
        print(f"GPU ready: {torch.cuda.get_device_name(0)}")
    else:
        print("no GPU detected, code will run very slowly!")

    # 1. prepare data
    texts, labels = load_data()
    # split 5% as validation set, monitor if the model is overfitting
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.05, random_state=42)

    # 2. load Tokenizer (from local path)
    print("loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LEN)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LEN)

    train_ds = TwitterDataset(train_encodings, train_labels)
    val_ds = TwitterDataset(val_encodings, val_labels)

    # 3. load model (from local path)
    print("loading DeBERTa-v3-Large model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)

    # 4. set training parameters
    args = TrainingArguments(
        output_dir='./results_deberta_4090',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        
        fp16=False,    # disable FP16
        bf16=True,     # enable BF16 (Ampere/Ada architecture, more stable and faster)
        
        dataloader_num_workers=8, 
        save_total_limit=1,       # only save the best model, save disk space
        report_to="none"          # don't upload wandb, keep clean
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # 5. start training
    print("🔥 engine started: start fine-tuning on 4090...")
    trainer.train()

    # 6. predict test set
    print("predicting Test Set...")
    test_ids, test_texts = [], []
    with open(TEST_FILE, 'r', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) < 2: continue
            test_ids.append(parts[0])
            test_texts.append(parts[1])

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LEN)
    test_ds = TwitterDataset(test_encodings)
    
    # batch prediction
    predictions = trainer.predict(test_ds)
    preds = predictions.predictions.argmax(-1)

    # 7. save results (key: map 0 back to -1)
    print(f"writing results to {SUBMISSION_FILE}...")
    with open(SUBMISSION_FILE, 'w') as f:
        f.write("Id,Prediction\n")
        for i, p in zip(test_ids, preds):
            # if the model predicts 1 (Positive), fill 1
            # if the model predicts 0 (Negative), fill -1
            label = 1 if p == 1 else -1
            f.write(f"{i},{label}\n")

    print("task completed! please submit the CSV file directly.")

if __name__ == "__main__":
    main()