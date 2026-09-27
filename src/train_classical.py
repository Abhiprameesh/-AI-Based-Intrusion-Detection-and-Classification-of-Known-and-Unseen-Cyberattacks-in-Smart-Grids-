import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.evaluate import evaluate_binary_classifier, evaluate_multiclass_classifier
from src.preprocessing import get_preprocessor
from src.utils import save_model, save_json, ensure_dir

def train_and_eval_binary_models(data_dict, results_dir, random_state=42):
    """
    Trains and evaluates the classical ML model suite on a binary detection task:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. XGBoost
    5. SVM (LinearSVC with Probability Calibration)
    """
    ensure_dir(results_dir)
    models_dir = ensure_dir(os.path.join(results_dir, "models"))
    metrics_dir = ensure_dir(os.path.join(results_dir, "metrics"))

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    dataset_name = data_dict['dataset_name']
    feature_names = data_dict['feature_names']

    # Preprocessors fitted strictly on training data
    prep_scaled = get_preprocessor(scaling=True)
    prep_unscaled = get_preprocessor(scaling=False)

    print(f"[{dataset_name}] Fitting preprocessors on training data...")
    X_train_scaled = prep_scaled.fit_transform(X_train)
    X_val_scaled = prep_scaled.transform(X_val)
    X_test_scaled = prep_scaled.transform(X_test)

    X_train_raw = prep_unscaled.fit_transform(X_train)
    X_val_raw = prep_unscaled.transform(X_val)
    X_test_raw = prep_unscaled.transform(X_test)

    # Save fitted preprocessor
    save_model(prep_scaled, os.path.join(models_dir, "preprocessor_scaled.joblib"))
    save_model(prep_unscaled, os.path.join(models_dir, "preprocessor_unscaled.joblib"))

    models_config = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'),
            "scaled": True
        },
        "Decision_Tree": {
            "model": DecisionTreeClassifier(max_depth=15, random_state=random_state, class_weight='balanced'),
            "scaled": False
        },
        "Random_Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=random_state, class_weight='balanced_subsample'),
            "scaled": False
        },
        "XGBoost": {
            "model": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=random_state, eval_metric='logloss'),
            "scaled": False
        },
        "SVM": {
            "model": SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3, random_state=random_state, class_weight='balanced'),
            "scaled": True
        }
    }

    all_metrics = []

    for name, cfg in models_config.items():
        print(f"[{dataset_name}] Training {name}...")
        model = cfg["model"]
        scaled = cfg["scaled"]

        X_tr = X_train_scaled if scaled else X_train_raw
        X_te = X_test_scaled if scaled else X_test_raw

        t_start = time.time()
        model.fit(X_tr, y_train)
        train_time = time.time() - t_start

        model_path = os.path.join(models_dir, f"{name}.joblib")
        save_model(model, model_path)

        metrics = evaluate_binary_classifier(
            model=model,
            X_test=X_te,
            y_test=y_test,
            training_time=train_time,
            feature_names=feature_names,
            model_filepath=model_path
        )
        metrics['model'] = name
        metrics['dataset_name'] = dataset_name
        metrics['task'] = 'binary'
        metrics['feature_count'] = len(feature_names)
        metrics['train_samples'] = len(y_train)
        metrics['test_samples'] = len(y_test)

        save_json(metrics, os.path.join(metrics_dir, f"{name}_metrics.json"))
        all_metrics.append(metrics)

        print(f"  -> Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Attack Rec: {metrics['recall']:.4f} | FPR: {metrics['fpr']:.4f} | Latency: {metrics['latency_per_sample_ms']:.3f} ms")

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = os.path.join(results_dir, "binary_model_comparison.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[{dataset_name}] Saved summary to {summary_csv}")

    return summary_df

def train_and_eval_multiclass_models(data_dict, results_dir, random_state=42):
    """
    Trains and evaluates the classical ML model suite on a multiclass attack classification task.
    """
    ensure_dir(results_dir)
    models_dir = ensure_dir(os.path.join(results_dir, "models"))
    metrics_dir = ensure_dir(os.path.join(results_dir, "metrics"))

    X_train = data_dict['X_train']
    y_train = data_dict['y_train_multi']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val_multi']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test_multi']
    dataset_name = data_dict['dataset_name']
    class_names = data_dict['class_names_multi']
    feature_names = data_dict['feature_names']

    prep_scaled = get_preprocessor(scaling=True)
    prep_unscaled = get_preprocessor(scaling=False)

    print(f"[{dataset_name} MULTICLASS] Fitting preprocessors...")
    X_train_scaled = prep_scaled.fit_transform(X_train)
    X_val_scaled = prep_scaled.transform(X_val)
    X_test_scaled = prep_scaled.transform(X_test)

    X_train_raw = prep_unscaled.fit_transform(X_train)
    X_val_raw = prep_unscaled.transform(X_val)
    X_test_raw = prep_unscaled.transform(X_test)

    save_model(prep_scaled, os.path.join(models_dir, "preprocessor_scaled.joblib"))
    save_model(prep_unscaled, os.path.join(models_dir, "preprocessor_unscaled.joblib"))

    models_config = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'),
            "scaled": True
        },
        "Decision_Tree": {
            "model": DecisionTreeClassifier(max_depth=18, random_state=random_state, class_weight='balanced'),
            "scaled": False
        },
        "Random_Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=22, n_jobs=-1, random_state=random_state, class_weight='balanced_subsample'),
            "scaled": False
        },
        "XGBoost": {
            "model": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=random_state, eval_metric='mlogloss'),
            "scaled": False
        },
        "SVM": {
            "model": SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3, random_state=random_state, class_weight='balanced'),
            "scaled": True
        }
    }

    all_metrics = []

    for name, cfg in models_config.items():
        print(f"[{dataset_name} MULTICLASS] Training {name}...")
        model = cfg["model"]
        scaled = cfg["scaled"]

        X_tr = X_train_scaled if scaled else X_train_raw
        X_te = X_test_scaled if scaled else X_test_raw

        t_start = time.time()
        model.fit(X_tr, y_train)
        train_time = time.time() - t_start

        model_path = os.path.join(models_dir, f"{name}_multiclass.joblib")
        save_model(model, model_path)

        metrics = evaluate_multiclass_classifier(
            model=model,
            X_test=X_te,
            y_test=y_test,
            class_names=class_names,
            training_time=train_time,
            model_filepath=model_path
        )
        metrics['model'] = name
        metrics['dataset_name'] = dataset_name
        metrics['task'] = 'multiclass'
        metrics['feature_count'] = len(feature_names)
        metrics['train_samples'] = len(y_train)
        metrics['test_samples'] = len(y_test)

        save_json(metrics, os.path.join(metrics_dir, f"{name}_multiclass_metrics.json"))
        all_metrics.append(metrics)

        print(f"  -> Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f} | Latency: {metrics['latency_per_sample_ms']:.3f} ms")

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = os.path.join(results_dir, "multiclass_model_comparison.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[{dataset_name} MULTICLASS] Saved summary to {summary_csv}")

    return summary_df
