"""
TurkishLegalBench Classification Script
Supported Tasks: TurkVerdict, TurkVenue, TurkCanon, TurkCoherence, TurkAudit
"""

import os
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from utils import compute_metrics, get_label_mapping


MODELS_TO_BENCHMARK = {
    "BERTurk": "dbmdz/bert-base-turkish-cased",
    "DistilBERT-TR": "dbmdz/distilbert-base-turkish-cased",
    "XLM-RoBERTa": "xlm-roberta-base",
    "Legal-BERT (Eng)": "nlpaueb/legal-bert-base-uncased",
    "Longformer": "allenai/longformer-base-4096",
    "BERT-Tr-128": "dbmdz/bert-base-turkish-128k-uncased"
}

# Standard labels for TurkVerdict task
labels = ['ONAMA', 'BOZMA', 'GÖNDERME', 'RED', 'DÜZELTME', 'DÜŞME']
label2id, id2label = get_label_mapping(labels)

def train_and_evaluate(model_name, model_key):
    print(f"\n>>> Training Model: {model_key} ({model_name})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    data_files = {
        "train": "../data/TurkVerdict/train.jsonl", 
        "test": "../data/TurkVerdict/test.jsonl", 
        "val": "../data/TurkVerdict/dev.jsonl"
    }
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.map(lambda x: {"label": label2id[x["outcome"]]})
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=f"results/{model_key}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro", #for imbalanced legal datasets
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer.evaluate(tokenized_datasets["test"])

if __name__ == "__main__":
    benchmark_results = []
    for key, name in MODELS_TO_BENCHMARK.items():
        try:
            res = train_and_evaluate(name, key)
            benchmark_results.append({
                "Model": key,
                "Accuracy": res["eval_accuracy"],
                "F1-Weighted": res["eval_f1_weighted"],
                "F1-Macro": res["eval_f1_macro"],
                "F1-Micro": res["eval_f1_micro"],
                "Loss": res["eval_loss"]
            })
        except Exception as e:
            print(f"FAILED: {key}. Reason: {e}")

    if benchmark_results:
        results_df = pd.DataFrame(benchmark_results)
        results_df.to_csv("benchmark_summary.csv", index=False)
        print(results_df.sort_values(by="F1-Macro", ascending=False).to_string(index=False))