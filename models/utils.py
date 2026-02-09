import numpy as np
import evaluate
import re


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
  
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    f1_macro = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    f1_micro = f1_metric.compute(predictions=predictions, references=labels, average="micro")

    return {
        "accuracy": acc["accuracy"],
        "f1_weighted": f1_weighted["f1"],
        "f1_macro": f1_macro["f1"],
        "f1_micro": f1_micro["f1"]
    }

def get_label_mapping(labels):
   
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return label2id, id2label