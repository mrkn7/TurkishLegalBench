# 📊 TurkLexBench Analysis Notebooks

This directory contains Jupyter notebooks used for the exploratory data analysis (EDA), result visualization, and model interpretability experiments presented in the **TurkLexBench** paper.

## 📂 Notebooks Overview

### 1. `01_Dataset_Statistics_and_Analysis.ipynb`
This notebook provides a deep dive into the dataset characteristics:
* **Class Distribution:** Visualizes the imbalance in tasks like *TurkVerdict* and *TurkVenue*.
* **Token Length Analysis:** Examines the distribution of document lengths using **BERTurk** sub-word units, highlighting the challenge of long legal documents.
* **Vocabulary Stats:** Analyzes the most frequent legal terms and n-grams across different courts.

### 2. `02_Model_Evaluation_and_Comparison.ipynb`
This notebook visualizes the benchmark results:
* **Performance Charts:** Compare Macro-F1, Weighted-F1, and Accuracy scores across all 7 tasks for models like BERTurk, XLM-RoBERTa, and Longformer.
* **Confusion Matrices:** Detailed error analysis for classification tasks to identify common misclassifications (e.g., confusing *Onama* with *Düzelterek Onama*).

### 3. `03_LIME_Interpretability_Analysis.ipynb`
This notebook focuses on the "Reasoning" capability of the models:
* **LIME Visualizations:** Uses Local Interpretable Model-agnostic Explanations (LIME) to highlight which words (e.g., specific law article numbers, dates) influenced the model's decision.
* **Case Studies:** specifically analyzes "Anachronistic" decisions in the *TurkAudit* task.

---

## 🛠️ Utility Script (`analysis_utils.py`)
We provide a helper script to streamline the notebooks and ensure reproducibility. It includes functions for:
* **Robust Data Loading:** Handles various encoding issues (UTF-8, CP1254) common in Turkish texts.
* **Tokenization:** Calculates real sub-word token counts using the `dbmdz/bert-base-turkish-cased` tokenizer.
* **Plotting:** Generates publication-ready figures using `matplotlib` and `seaborn`.

## 🚀 How to Run
1. Install the requirements:
   ```bash
   pip install -r ../requirements.txt
   pip install lime matplotlib seaborn