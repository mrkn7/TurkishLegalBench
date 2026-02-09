# ⚖️ TurkLexBench: A Comprehensive Multi-Task Benchmark Suite for Turkish Legal NLP

> [!WARNING]
> **Under Review:** This repository contains the official dataset and codebase for the paper **"TurkLexBench: A Comprehensive Multi-Task Benchmark Suite for Turkish Legal NLP"**, currently under review for **KDD 2026 (Datasets & Benchmarks Track)**. 
> While the data is open for reproducibility, please cite the work if you use it.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## 📖 Overview

**TurkLexBench** is the first large-scale, open-source legal NLP benchmark for the Turkish language. It comprises **38,009 authentic legal documents** sourced from high courts (Yargıtay, Danıştay) and the Official Gazette (Resmi Gazete).

The benchmark covers **7 distinct tasks** organized into three cognitive pillars, designed to evaluate models on structural understanding, information extraction, and logical reasoning.

---

## 🏛️ The Tasks (The 3 Pillars)

We organize the benchmark into three pillars representing different levels of legal cognition:

### I. The Gavel (High-Level Classification)
| Task | Description | Metric | Size (Train/Dev/Test) |
| :--- | :--- | :--- | :--- |
| **TurkVerdict** | Predict the judgment outcome (e.g., Affirmation, Reversal) from the case rationale. | Macro-F1 | 18k / 2.2k / 2.2k |
| **TurkVenue** | Identify the competent court chamber (Daire) based on case facts (36 classes). | Macro-F1 | 19.2k / 2.7k / 5.5k |
| **TurkCanon** | Classify legislative documents into types (Law, Regulation, Decree, etc.). | Macro-F1 | 6.3k / 0.9k / 1.8k |

### II. The Quill (Information Extraction)
| Task | Description | Metric | Size (Train/Dev/Test) |
| :--- | :--- | :--- | :--- |
| **TurkChronos** | Identify the decision year of a case amidst distractor dates. | Accuracy | 19.4k / 2.7k / 5.5k |
| **TurkCite** | Extract citations (Law No. & Article No.) from unstructured text (NER). | Entity F1 | 10.9k / 1.5k / 3.1k |

### III. The Scale (Legal Reasoning)
| Task | Description | Metric | Size (Train/Dev/Test) |
| :--- | :--- | :--- | :--- |
| **TurkCoherence** | Natural Language Inference (NLI) to check if the reasoning supports the verdict. | Macro-F1 | 4.9k / 0.7k / 1.4k |
| **TurkAudit** | Detect "legal hallucinations" and anachronistic citations (e.g., citing a 2016 law in 2010). | Weighted-F1 | 7k / 1k / 2k |

---

## 📂 Repository Structure

The data is provided in **JSONL** format, pre-split to ensure reproducibility.

```text
TurkLexBench/
├── data/
│   ├── TurkVerdict/       # JSONL files for verdict prediction
│   ├── TurkVenue/         # JSONL files for chamber classification
│   ├── TurkCanon/         # Legislative classification
│   ├── TurkChronos/       # Year prediction
│   ├── TurkCite/          # NER data (BIO format)
│   ├── TurkCoherence/     # NLI pairs
│   └── TurkAudit/         # Hallucination detection
├── models/                # Fine-tuning scripts
├── notebooks/             # Analysis notebooks
├── requirements.txt       # Dependencies
└── LICENSE                # CC BY-NC-SA 4.0
```


## Installation

```bash
git clone [https://github.com/mrkn7/TurkishLegalBench.git](https://github.com/mrkn7/TurkishLegalBench.git)
cd TurkishLegalBench
pip install -r requirements.txt
```


2. Loading Data (Example)
You can easily load the data using the datasets library or standard JSON lines:

```bash
import json
# Load Train Set for TurkVerdict
with open('data/TurkVerdict/train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        print(data['text'], data['label'])
```

## 📊 Benchmark Results

We evaluated baseline and domain-adapted models across all 7 tasks using the **Test Set**. The table below reports the primary metric for each task (Macro-F1 for classification, Accuracy for Chronos, and Entity-F1 for NER).

| Model | Verdict <br> *(m-F1)* | Venue <br> *(m-F1)* | Canon <br> *(m-F1)* | Chronos <br> *(Acc)* | Cite <br> *(F1)* | Coherence <br> *(m-F1)* | Audit <br> *(W-F1)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TFIDF-SVM** | 68.4 | 41.7 | 35.8 | 56.0 | 76.4 | - | 87.6 |
| **BERTurk** | 81.9 | 79.4 | 92.6 | **94.1** 🏆 | 95.5 | **56.8** 🏆 | 96.9 |
| **Legal-BERT (Eng)** | 68.1 | 63.1 | 91.1 | 92.3 | 95.5 | 50.1 | 95.1 |
| **XLM-RoBERTa** | 79.2 | 64.4 | 70.4 | 70.2 | 95.4 | 33.7 | 96.6 |
| **Longformer** | 69.3 | 59.9 | 92.6 | 93.7 | 94.5 | 43.9 | **98.5** 🏆 |
| **BERT-TR-128k** | **82.6** 🏆 | **82.2** 🏆 | **93.8** 🏆 | 94.0 | **96.0** 🏆 | 50.4 | 97.6 |

> **🏆 Key Takeaways:**
> * **Vocabulary Matters:** The **BERT-TR-128k** model (with expanded vocabulary) achieves State-of-the-Art (SOTA) in 4 out of 7 tasks, significantly outperforming standard BERTurk in extraction and rare-class classification tasks.
> * **Context is King for Auditing:** **Longformer** dominates the *TurkAudit* task (%98.5), proving that large context windows are essential for "needle-in-a-haystack" retrieval tasks where the anomaly appears late in the document.
> * **Reasoning Gap:** All models struggle with the *TurkCoherence* (NLI) task, indicating that current LLMs are better at surface-level pattern matching than deep legal logical entailment.


## 📜 License

This dataset and benchmark suite are distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.

[Image of Creative Commons License logo]

### Under this license, you are free to:
* **Share** — copy and redistribute the material in any medium or format.
* **Adapt** — remix, transform, and build upon the material.

### Under the following terms:
* **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
* **NonCommercial** — You may not use the material for commercial purposes.
* **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

> **Note:** The underlying raw texts (court decisions and laws) are public records. This license applies to the **curated benchmark, annotations, and structured dataset** created by the authors.

---


## 📧 Contact
For questions, feedback, or collaboration opportunities, please contact:

Mehmet Ali Erkan - maerkan@metu.edu.tr

Prof. Dr. Ceylan Yozgatlıgil - ceylan@metu.edu.tr
