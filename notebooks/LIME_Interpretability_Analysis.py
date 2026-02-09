# # LIME Interpretability Analysis
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
from datasets import load_dataset


# UPDATE THIS PATH to your trained model directory
MODEL_PATH = "../results/BERTurk/checkpoint-final" 
TEST_FILE = "../data/TurkVerdict/test.jsonl"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model path '{MODEL_PATH}' not found. Please train a model first using `models/train_classification.py`.")
    # Fallback to a default HF model just to show the code runs (Optional)
    # MODEL_PATH = "dbmdz/bert-base-turkish-cased" 

# ## Load Model & Tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")


# ## Define LIME Predictor Function
# LIME requires a function that takes raw text and returns probability arrays.

def predictor(texts):
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    
    # Move to GPU
    for k, v in inputs.items():
        inputs[k] = v.to(device)
        
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
        
    return probs


# ##Select a Case for Analysis
# We select a random sample from the test set to explain.

dataset = load_dataset('json', data_files={'test': TEST_FILE})
test_data = dataset['test']

# Pick a sample (e.g., Index 5)
idx = 5
text_instance = test_data[idx]['text']
true_label = test_data[idx]['label']

print(f"Document Text (First 300 chars):\n{text_instance[:300]}...")
print(f"True Label: {true_label}")


class_names = ['ONAMA', 'BOZMA', 'GÖNDERME', 'RED', 'DÜZELTME', 'DÜŞME'] # Update based on task

explainer = LimeTextExplainer(class_names=class_names)

exp = explainer.explain_instance(
    text_instance, 
    predictor, 
    num_features=10, 
    num_samples=100  # Increase to 1000 for high precision
)

exp.show_in_notebook(text=True)

for word, weight in exp.as_list():
    effect = "Supports" if weight > 0 else "Opposes"
    print(f"{word:20} : {weight:+.4f} ({effect})")