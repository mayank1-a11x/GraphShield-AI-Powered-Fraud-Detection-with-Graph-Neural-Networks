"""Non-graph baseline: XGBoost on raw tabular features.

This is the control your GNN results need to beat. If the GNN doesn't clear
this bar, that's a real finding too — it usually means the entity-sharing
signal isn't strong enough in your graph construction, not that GNNs "don't
work" for the problem.
"""
import logging

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, classification_report

from src import config
from src.data_preprocessing import run_preprocessing, get_feature_matrix
from src.graph_construction import make_train_val_test_masks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Of the top-k highest-scored transactions, what fraction of all fraud
    did we catch? Mirrors a "review queue" production setting."""
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    caught = y_true[top_k_idx].sum()
    total_fraud = y_true.sum()
    return caught / total_fraud if total_fraud > 0 else 0.0


def train_baseline():
    df = run_preprocessing()
    X, feature_cols = get_feature_matrix(df)
    y = df["isFraud"].values

    train_mask, val_mask, test_mask = make_train_val_test_masks(len(df))
    train_idx, val_idx, test_idx = (
        train_mask.numpy(), val_mask.numpy(), test_mask.numpy(),
    )

    scale_pos_weight = (y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=config.SEED,
    )

    logger.info("Training XGBoost on %d rows, %d features", train_idx.sum(), X.shape[1])
    model.fit(
        X[train_idx], y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        verbose=False,
    )

    test_probs = model.predict_proba(X[test_idx])[:, 1]
    test_labels = y[test_idx]

    ap = average_precision_score(test_labels, test_probs)
    f1 = f1_score(test_labels, test_probs > 0.5)

    logger.info("=== XGBoost baseline (test set) ===")
    logger.info("AUC-PR: %.4f", ap)
    logger.info("F1 (fraud class): %.4f", f1)
    for k in config.RECALL_AT_K:
        r = recall_at_k(test_labels, test_probs, k)
        logger.info("Recall@%d: %.4f", k, r)

    print(classification_report(test_labels, test_probs > 0.5, target_names=["legit", "fraud"]))

    model.save_model(f"{config.CHECKPOINT_DIR}/xgboost_baseline.json")
    return {"ap": ap, "f1": f1, "model": "xgboost"}


if __name__ == "__main__":
    train_baseline()
