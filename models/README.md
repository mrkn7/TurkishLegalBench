# 🤖 TurkishLegalBench Model Suite

This directory contains the pipeline for training and evaluating Transformer-based models on the TurkLexBench dataset.

## 🚀 Overview
The `train_classification.py` script benchmarks multiple architectures (BERTurk, XLM-RoBERTa, Longformer, etc.) on legal text classification tasks. We prioritize **Macro-F1** as the primary metric to account for the class imbalance inherent in judicial decisions.

## 🛠️ Components
* **`train_classification.py`**: The main training loop utilizing the Hugging Face `Trainer` API.
* **`utils.py`**: Helper functions for dynamic label mapping and comprehensive legal NLP metrics.

## 📈 Hyperparameters
* **Epochs**: 3
* **Learning Rate**: 2e-5
* **Batch Size**: 16
* **FP16**: Enabled (when GPU is available)
* **Optimization Metric**: Macro-F1

## 📖 Accessing Data
Before running the scripts, ensure the dataset is downloaded from **Zenodo** and placed in the `data/` directory at the repository root.
