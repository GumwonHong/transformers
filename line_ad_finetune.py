"""
LINE Plus contextual ad BERT fine-tuning script using HuggingFace Transformers.
Fine-tunes multilingual BERT for ad category classification.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import List


@dataclass
class AdCategory:
    label: str
    id: int


AD_CATEGORIES = [
    AdCategory("sports", 0), AdCategory("finance", 1),
    AdCategory("technology", 2), AdCategory("lifestyle", 3),
    AdCategory("entertainment", 4), AdCategory("health", 5),
]


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )
    return train_texts, val_texts, train_labels, val_labels


def tokenize_data(tokenizer, texts: List[str], labels: List[int]):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
    dataset = Dataset.from_dict({**encodings, "labels": labels})
    return dataset


def train(model_name: str = "bert-base-multilingual-cased", csv_path: str = "ad_data.csv"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(AD_CATEGORIES)
    )
    train_texts, val_texts, train_labels, val_labels = load_data(csv_path)
    train_dataset = tokenize_data(tokenizer, train_texts, train_labels)
    val_dataset = tokenize_data(tokenizer, val_texts, val_labels)
    args = TrainingArguments(
        output_dir="./line_ad_bert",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir="./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
    )
    trainer.train()
    model.save_pretrained("./line_ad_bert_final")
    tokenizer.save_pretrained("./line_ad_bert_final")


if __name__ == "__main__":
    train()
