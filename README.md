# ⚖️ TurkishLegalBench: A Comprehensive Multi-Task Benchmark Suite for Turkish Legal NLP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18555735.svg)](https://doi.org/10.5281/zenodo.18555735)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## 📖 Overview

**TurkishLegalBench** is the first large-scale, open-source legal NLP benchmark for the Turkish language. It comprises **38,009 authentic legal documents** sourced from high courts (Yargıtay, Danıştay) and the Official Gazette (Resmi Gazete).

The benchmark covers **7 distinct tasks** organized into three cognitive pillars, designed to evaluate models on structural understanding, information extraction, and logical reasoning.

---

## 🏛️ The Tasks (The 3 Pillars)

We organize the benchmark into three pillars representing different levels of legal cognition:

### I. The Gavel (High-Level Classification)
| Task | Description | Metric | Size (Train/Dev/Test) |
| :--- | :--- | :--- | :--- |
| **TurkVerdict** | Predict the judgment outcome (e.g., Affirmation, Reversal) from the case rationale. | Macro-F1 | 18k / 2.2k / 2.2k |
| **TurkVenue** | Identify the competent court chamber (Daire) based on case facts (36 classes). | Macro-F1 | 17.7k / 2.5k / 5.0k |
| **TurkCanon** | Classify legislative documents into types (Law, Regulation, Decree, etc.). | Macro-F1 | 6.4k / 0.9k / 1.8k |

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


## 📂 Dataset Access (Zenodo)

Due to GitHub's file size limitations, the full dataset is hosted on **Zenodo** for long-term archival and reproducibility. 

**Download Link:** [https://doi.org/10.5281/zenodo.18555735](https://doi.org/10.5281/zenodo.18555735)

To replicate the environment:
1. Download the task-specific `.zip` files from Zenodo.
2. Extract them into the `data/` directory of this repository following the structure below.

```text
TurkLexBench/
├── data/
│   ├── TurkVerdict/      
│   ├── TurkVenue/         
│   ├── TurkCanon/        
│   ├── TurkChronos/       
│   ├── TurkCite/          
│   ├── TurkCoherence/     
│   └── TurkAudit/        
├── models/                
├── notebooks/            
├── requirements.txt       
└── LICENSE                
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
with open('data/TurkVerdict/train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        print(f"Text: {data['text'][:100]}... | Outcome: {data['outcome']}")
```

## 📊 Benchmark Results

We evaluated baseline and domain-adapted models across all 7 tasks using the **Test Set**. The table below reports the primary metric for each task: Macro-F1 for classification tasks, Accuracy for Chronos, Entity-F1 for Cite, and Weighted-F1 for Audit.

| Model                | Verdict <br> *(m-F1)* | Venue <br> *(m-F1)* | Canon <br> *(m-F1)* | Chronos <br> *(Acc)* | Cite <br> *(F1)* | Coherence <br> *(m-F1)* | Audit <br> *(W-F1)* |
| :------------------- | :-------------------: | :-----------------: | :-----------------: | :------------------: | :--------------: | :---------------------: | :-----------------: |
| **BERTurk**          |          75.3         |     **82.0** 🏆     |     **92.9** 🏆     |         88.8         |       45.8       |           53.1          |         97.4        |
| **Legal-BERT (Eng)** |          68.8         |         62.9        |         92.2        |         90.0         |       37.5       |           45.9          |         95.8        |
| **XLM-RoBERTa**      |          72.0         |         64.0        |         52.5        |         84.9         |       41.9       |           33.0          |         96.8        |
| **DistilBERT-TR**    |          70.6         |         62.5        |         87.1        |         82.6         |       41.6       |           33.2          |         97.4        |
| **Longformer**       |          70.1         |         62.1        |         91.6        |         83.7         |    **58.4** 🏆   |           33.0          |         97.0        |
| **BERT-TR-128k**     |      **79.4** 🏆      |         81.1        |         92.6        |      **92.0** 🏆     |       49.3       |       **64.3** 🏆       |     **98.3** 🏆     |

> **🏆 Key Takeaways:**
>
> * **Expanded Turkish Vocabulary Improves Overall Performance:** The **BERT-TR-128k** model achieves the best score in **4 out of 7 tasks**: Verdict, Chronos, Coherence, and Audit. This shows that expanding the Turkish vocabulary provides consistent gains, especially in decision classification, temporal reasoning, logical coherence, and audit-style detection.
> * **BERTurk Remains Strong for Core Legal Classification:** Standard **BERTurk** achieves the best performance on **Venue** and **Canon**, indicating that a strong Turkish encoder is still highly competitive for structured legal classification tasks.
> * **Long Context Helps Citation Extraction:** **Longformer** achieves the highest score on the **Cite** task, suggesting that longer context windows are particularly useful when citation-related evidence may appear in different parts of the document.
> * **Coherence Remains Challenging:** Although **BERT-TR-128k** substantially improves TurkCoherence performance, the task remains more difficult than surface-level classification tasks. This suggests that legal entailment and logical consistency still require deeper reasoning beyond lexical or structural pattern matching.


## 📜 License

This dataset and benchmark suite are distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.

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
