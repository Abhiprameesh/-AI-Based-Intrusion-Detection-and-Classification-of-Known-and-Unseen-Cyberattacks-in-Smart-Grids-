import os
import random
import time
import json
import joblib
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_model(model, filepath):
    ensure_dir(os.path.dirname(filepath))
    if filepath.endswith('.pt') or filepath.endswith('.pth'):
        torch.save(model.state_dict(), filepath)
    else:
        joblib.dump(model, filepath)

def load_model(filepath, model_class=None):
    if filepath.endswith('.pt') or filepath.endswith('.pth'):
        if model_class is None:
            raise ValueError("model_class must be provided to load PyTorch state dict")
        model = model_class()
        model.load_state_dict(torch.load(filepath, weights_only=True))
        model.eval()
        return model
    else:
        return joblib.load(filepath)

def save_json(data, filepath):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
