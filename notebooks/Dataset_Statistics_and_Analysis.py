# This notebook explores the distributions and characteristics of the TurkishLegalBench.
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.getcwd())
import analysis_utils as utils

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

tasks = {
    "TurkVerdict": "../data/TurkVerdict/train.jsonl",
    "TurkVenue": "../data/TurkVenue/train.jsonl",
    "TurkAudit": "../data/TurkAudit/train.jsonl"
}

dfs = {}
for task_name, path in tasks.items():
    print(f"Loading {task_name}...")
    df = utils.read_jsonl_robust(path)
    if df is not None:
        dfs[task_name] = df

if "TurkVerdict" in dfs:
    stats = utils.get_dataset_stats(dfs["TurkVerdict"], label_col="outcome")
    utils.plot_class_distribution(stats, title="TurkVerdict - Class Imbalance")


if "TurkAudit" in dfs:
    df_audit = utils.calculate_token_counts(dfs["TurkAudit"], tokenizer_name="dbmdz/bert-base-turkish-cased")
    
    utils.plot_token_histogram(df_audit, title="TurkAudit - Document Length Distribution")
    # Print Stats
    print(f"Mean Token Count: {df_audit['token_count'].mean():.2f}")
    print(f"Max Token Count: {df_audit['token_count'].max()}")
    print(f"Documents > 512 tokens: {len(df_audit[df_audit['token_count'] > 512])} / {len(df_audit)}")