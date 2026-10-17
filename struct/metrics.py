import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def aggregate_series_to_study_predictions(
    study_uids: List[str],
    probs: np.ndarray,
    targets: np.ndarray,
    labels: List[str],
    method: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates series-level probability predictions into study-level predictions
    grouped by StudyInstanceUID.
    Supported methods: 'mean', 'max', 'noisy_or'.
    """
    pred_df = pd.DataFrame(probs, columns=labels)
    pred_df["StudyInstanceUID"] = study_uids

    true_df = pd.DataFrame(targets, columns=labels)
    true_df["StudyInstanceUID"] = study_uids

    if method == "max":
        study_preds = pred_df.groupby("StudyInstanceUID", sort=True)[labels].max()
    elif method == "noisy_or":
        # 1 - prod(1 - p_i)
        clipped = pred_df.copy()
        clipped[labels] = np.clip(1.0 - clipped[labels].values, 1e-6, 1.0)
        study_preds = 1.0 - clipped.groupby("StudyInstanceUID", sort=True)[labels].prod()
    else:
        study_preds = pred_df.groupby("StudyInstanceUID", sort=True)[labels].mean()

    study_trues = true_df.groupby("StudyInstanceUID", sort=True)[labels].max()
    return study_preds, study_trues


def compute_study_level_metrics(
    study_uids: List[str],
    probs: np.ndarray,
    targets: np.ndarray,
    labels: List[str],
    aggregation: str = "mean",
) -> Tuple[float, Dict[str, float], int]:
    """
    Computes per-class ROC AUC and Macro ROC AUC strictly at the StudyInstanceUID level.
    Safely handles missing or single-class columns without crashing.
    """
    study_preds_df, study_trues_df = aggregate_series_to_study_predictions(
        study_uids, probs, targets, labels, method=aggregation
    )
    y_pred = study_preds_df[labels].values
    y_true = study_trues_df[labels].values

    per_class_auc: Dict[str, float] = {}
    valid_aucs: List[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx, label in enumerate(labels):
            col_true = y_true[:, idx]
            col_pred = y_pred[:, idx]
            valid_mask = ~np.isnan(col_true) & ~np.isnan(col_pred)
            col_true = col_true[valid_mask]
            col_pred = col_pred[valid_mask]
            if len(np.unique(col_true)) > 1:
                score = float(roc_auc_score(col_true, col_pred))
                per_class_auc[label] = score
                valid_aucs.append(score)
            else:
                per_class_auc[label] = float("nan")

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return macro_auc, per_class_auc, len(study_preds_df)
