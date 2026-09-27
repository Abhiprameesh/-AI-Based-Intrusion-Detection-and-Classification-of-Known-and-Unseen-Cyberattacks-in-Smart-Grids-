import os
import pandas as pd
from src.utils import ensure_dir

WORKSPACE = r"c:\ALLMLPROJ\MAINMLPROJS\MAJOR PROJECT"
REPORTS_DIR = os.path.join(WORKSPACE, "reports")

MASTER_COMPARISON_CSV = os.path.join(REPORTS_DIR, "final_model_comparison.csv")
DATASET_COMPARISON_CSV = os.path.join(REPORTS_DIR, "dataset_comparison.csv")

MODEL_COMPARISON_COLUMNS = [
    "Dataset_Name", "Task", "Model", "Accuracy", "Precision", "Recall",
    "Macro_F1", "Weighted_F1", "FPR", "FNR", "ROC_AUC", "PR_AUC",
    "Training_Time", "Inference_Time", "Feature_Count", "Training_Samples",
    "Testing_Samples", "Number_of_Classes"
]

def log_model_result(
    dataset_name, task, model_name, accuracy, precision, recall,
    macro_f1, weighted_f1, fpr, fnr, roc_auc, pr_auc,
    training_time, inference_time, feature_count, training_samples,
    testing_samples, number_of_classes
):
    ensure_dir(REPORTS_DIR)
    row = {
        "Dataset_Name": dataset_name,
        "Task": task,
        "Model": model_name,
        "Accuracy": round(float(accuracy), 4) if accuracy is not None else None,
        "Precision": round(float(precision), 4) if precision is not None else None,
        "Recall": round(float(recall), 4) if recall is not None else None,
        "Macro_F1": round(float(macro_f1), 4) if macro_f1 is not None else None,
        "Weighted_F1": round(float(weighted_f1), 4) if weighted_f1 is not None else None,
        "FPR": round(float(fpr), 4) if fpr is not None else None,
        "FNR": round(float(fnr), 4) if fnr is not None else None,
        "ROC_AUC": round(float(roc_auc), 4) if roc_auc is not None else None,
        "PR_AUC": round(float(pr_auc), 4) if pr_auc is not None else None,
        "Training_Time": round(float(training_time), 3) if training_time is not None else None,
        "Inference_Time": round(float(inference_time), 4) if inference_time is not None else None,
        "Feature_Count": int(feature_count),
        "Training_Samples": int(training_samples),
        "Testing_Samples": int(testing_samples),
        "Number_of_Classes": int(number_of_classes)
    }

    if os.path.exists(MASTER_COMPARISON_CSV):
        df = pd.read_csv(MASTER_COMPARISON_CSV)
        # Update if exists or append
        mask = (df['Dataset_Name'] == dataset_name) & (df['Task'] == task) & (df['Model'] == model_name)
        if mask.any():
            for k, v in row.items():
                df.loc[mask, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=MODEL_COMPARISON_COLUMNS)

    df.to_csv(MASTER_COMPARISON_CSV, index=False)
    print(f"[Tracker] Logged {dataset_name} | {task} | {model_name} -> Macro F1: {row['Macro_F1']}")
