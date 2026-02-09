import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

sns.set_theme(style="whitegrid", palette="muted")

def read_jsonl_robust(file_path):
    """
    Reads a JSONL file with multiple encoding attempts.
    Crucial for Turkish texts that might be saved in different environments (Windows/Mac).
    
    Args:
        file_path (str): Path to the .jsonl file.
        
    Returns:
        pd.DataFrame: Loaded data or None if failed.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found -> {file_path}")
        return None

    data = []
    encodings = ['utf-8', 'utf-8-sig', 'windows-1254', 'latin-1']
    
    success = False
    for enc in encodings:
        try:
            temp_data = []
            with open(file_path, 'r', encoding=enc) as f:
                for line in f:
                    if line.strip():
                        temp_data.append(json.loads(line))
            data = temp_data
            success = True

            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
            
    if not success:
        print(f"Failed to read {file_path} with any supported encoding.")
        return None

    return pd.DataFrame(data)


#STATISTICS & TOKENIZATION

def get_dataset_stats(df, label_col='outcome'):
    """
    Calculates basic statistics for the dataset including class distribution.
    """
    if label_col not in df.columns:
        # Fallback if 'label' is used instead of 'outcome'
        if 'label' in df.columns:
            label_col = 'label'
        else:
            return {"error": "Label column not found"}

    stats = {
        "total_documents": len(df),
        "num_classes": df[label_col].nunique(),
        "class_distribution": df[label_col].value_counts().to_dict()
    }
    return stats

def calculate_token_counts(df, tokenizer_name="dbmdz/bert-base-turkish-cased"):
    """
    Calculates REAL token counts using a specific tokenizer (default: BERTurk).
    Adds a 'token_count' column to the DataFrame.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Tokenizer load failed: {e}")
        return df

    # Helper to count tokens (efficiently)
    def count_tokens(text):
        if not isinstance(text, str): return 0
        # encode(add_special_tokens=True) ensures we count CLS and SEP too
        return len(tokenizer.encode(text, add_special_tokens=True))

    df['token_count'] = df['text'].apply(count_tokens)
    return df

# VISUALIZATION

def plot_class_distribution(stats, title="Class Distribution"):
    """
    Plots a bar chart for class distribution.
    """
    if 'class_distribution' not in stats or not stats['class_distribution']:
        print("No class distribution data to plot.")
        return

    data = stats['class_distribution']
    
    plt.figure(figsize=(12, 6))
    # Create barplot
    ax = sns.barplot(x=list(data.keys()), y=list(data.values()), palette="viridis")
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Number of Documents", fontsize=12)
    
    # Rotate x-labels if there are many classes (like in TurkVenue)
    if len(data) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Add numbers on top of bars
    for i, v in enumerate(data.values()):
        ax.text(i, v + (max(data.values())*0.01), str(v), ha='center', fontsize=10)
        
    plt.tight_layout()
    plt.show()

def plot_token_histogram(df, title="Token Length Distribution"):
    """
    Plots a histogram of token counts to visualize document lengths.
    Adds a red line at 512 to show BERT limitations.
    """
    if 'token_count' not in df.columns:
        print("'token_count' column missing. Run calculate_token_counts() first.")
        return

    plt.figure(figsize=(12, 6))
    sns.histplot(df['token_count'], bins=40, kde=True, color="#1f77b4", edgecolor="black")
    
    # Critical Visual: The BERT Limit Line
    plt.axvline(x=512, color='red', linestyle='--', linewidth=2, label="BERT Limit (512 Tokens)")
    
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("Number of Tokens (Sub-words)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


#LIME / MODEL HELPERS

def get_lime_predictor(model, tokenizer, device):
    """
    Returns a predictor function compatible with LIME.
    LIME requires a function that takes raw text list and returns numpy probabilities.
    """
    import torch
    import torch.nn.functional as F

    def predictor(texts):
        # Tokenize
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Move inputs to GPU
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            # Convert logits to probabilities
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
            
        return probs
    
    return predictor