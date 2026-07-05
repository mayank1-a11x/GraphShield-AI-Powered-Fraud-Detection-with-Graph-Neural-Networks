"""Create a plain-English fraud prediction report.

Outputs:
- outputs/prediction_results.csv: one row per transaction
- outputs/simple_summary.json: caught/missed counts and sentence
- outputs/fraud_caught_missed.svg: simple bar chart
- outputs/review_queue.json: analyst-friendly dashboard data
"""
import csv
import json
import os

import numpy as np
import xgboost as xgb

from src import config
from src.data_preprocessing import run_preprocessing, get_feature_matrix
from src.graph_construction import make_train_val_test_masks


PREDICTIONS_PATH = os.path.join(config.OUTPUT_DIR, "prediction_results.csv")
SUMMARY_PATH = os.path.join(config.OUTPUT_DIR, "simple_summary.json")
CHART_PATH = os.path.join(config.OUTPUT_DIR, "fraud_caught_missed.svg")
REVIEW_QUEUE_PATH = os.path.join(config.OUTPUT_DIR, "review_queue.json")


def _train_model(X: np.ndarray, y: np.ndarray):
    train_mask, _, _ = make_train_val_test_masks(len(y))
    train_idx = train_mask.numpy()
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
    model.fit(X[train_idx], y[train_idx], verbose=False)
    return model


def _write_predictions(df, probabilities: np.ndarray, predicted: np.ndarray):
    decision_confidence = np.where(predicted == 1, probabilities, 1 - probabilities)
    with open(PREDICTIONS_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "TransactionID",
                "Actual Fraud",
                "Model Thought Fraud",
                "Confidence",
            ],
        )
        writer.writeheader()
        for _, row in df.iterrows():
            idx = row.name
            writer.writerow(
                {
                    "TransactionID": row["TransactionID"],
                    "Actual Fraud": "Yes" if int(row["isFraud"]) == 1 else "No",
                    "Model Thought Fraud": "Yes" if int(predicted[idx]) == 1 else "No",
                    "Confidence": f"{decision_confidence[idx] * 100:.1f}%",
                }
            )


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.90:
        return "High"
    if confidence >= 0.70:
        return "Medium"
    return "Low"


def _reason_flagged(row, amount_p95: float, risky_device_values: set, risky_email_values: set) -> str:
    reasons = []
    if "TransactionAmt" in row and float(row["TransactionAmt"]) >= amount_p95:
        reasons.append("Unusual transaction amount")

    device_value = f"{row.get('DeviceType', 'unknown')}_{row.get('DeviceInfo', 'unknown')}"
    if device_value in risky_device_values:
        reasons.append("High-risk device pattern")

    email_value = f"{row.get('P_emaildomain', 'unknown')}_{row.get('R_emaildomain', 'unknown')}"
    if email_value in risky_email_values:
        reasons.append("Shared email pattern")

    if not reasons:
        reasons.append("Similar to known fraud cases")
    return "; ".join(reasons[:2])


def _risky_entity_values(df, columns: list[str], min_group_size: int = 3) -> set:
    present = [col for col in columns if col in df.columns]
    if not present:
        return set()

    composite = df[present].astype(str).agg("_".join, axis=1)
    grouped = df.assign(_entity_value=composite).groupby("_entity_value")["isFraud"].agg(["sum", "count"])
    risky = grouped[(grouped["count"] >= min_group_size) & (grouped["sum"] > 0)]
    return set(risky.index)


def _build_review_queue(df, probabilities: np.ndarray, predicted: np.ndarray, summary: dict) -> dict:
    decision_confidence = np.where(predicted == 1, probabilities, 1 - probabilities)
    amount_p95 = float(df["TransactionAmt"].quantile(0.95)) if "TransactionAmt" in df.columns else 0.0
    risky_device_values = _risky_entity_values(df, config.ENTITY_COLUMNS.get("device", []))
    risky_email_values = _risky_entity_values(df, config.ENTITY_COLUMNS.get("email", []))

    review_rows = []
    ranked_indexes = np.argsort(probabilities)[::-1][:20]
    for idx in ranked_indexes:
        row = df.iloc[int(idx)]
        confidence = float(decision_confidence[idx])
        review_rows.append(
            {
                "transaction_id": str(row["TransactionID"]),
                "actual_fraud": "Yes" if int(row["isFraud"]) == 1 else "No",
                "model_prediction": "Fraud" if int(predicted[idx]) == 1 else "Not Fraud",
                "confidence": round(confidence * 100, 1),
                "amount": round(float(row.get("TransactionAmt", 0.0)), 2),
                "reason": _reason_flagged(row, amount_p95, risky_device_values, risky_email_values),
            }
        )

    high = int((decision_confidence >= 0.90).sum())
    medium = int(((decision_confidence >= 0.70) & (decision_confidence < 0.90)).sum())
    low = int((decision_confidence < 0.70).sum())

    return {
        "summary": summary,
        "cards": {
            "fraud_caught": summary["fraud_caught"],
            "fraud_missed": summary["fraud_missed"],
            "transactions_reviewed": int(len(df)),
        },
        "confidence": {
            "high": high,
            "medium": medium,
            "low": low,
        },
        "top_suspicious": review_rows,
    }


def _render_chart(caught: int, missed: int) -> str:
    max_value = max(caught, missed, 1)
    chart_height = 250
    caught_height = round(chart_height * caught / max_value)
    missed_height = round(chart_height * missed / max_value)
    base_y = 320

    return f"""<svg viewBox="0 0 720 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Fraud caught versus fraud missed">
  <rect width="720" height="420" fill="#ffffff"/>
  <line x1="90" y1="{base_y}" x2="630" y2="{base_y}" stroke="#94a3b8" stroke-width="2"/>
  <rect x="165" y="{base_y - caught_height}" width="135" height="{caught_height}" rx="7" fill="#16a34a"/>
  <rect x="420" y="{base_y - missed_height}" width="135" height="{missed_height}" rx="7" fill="#dc2626"/>
  <text x="232.5" y="{base_y - caught_height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#14532d">{caught}</text>
  <text x="487.5" y="{base_y - missed_height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#7f1d1d">{missed}</text>
  <text x="232.5" y="370" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#111827">Fraud Caught</text>
  <text x="487.5" y="370" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#111827">Fraud Missed</text>
</svg>
"""


def build_simple_report():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df = run_preprocessing()
    X, _ = get_feature_matrix(df)
    y = df["isFraud"].values.astype(int)

    model = _train_model(X, y)
    probabilities = model.predict_proba(X)[:, 1]
    predicted = (probabilities >= 0.5).astype(int)

    fraud_mask = y == 1
    caught = int(((predicted == 1) & fraud_mask).sum())
    total_fraud = int(fraud_mask.sum())
    missed = total_fraud - caught
    sentence = (
        f"Out of {total_fraud} fraudulent transactions in the provided dataset, "
        f"the model correctly caught {caught} of them."
    )
    summary = {
        "total_fraud": total_fraud,
        "fraud_caught": caught,
        "fraud_missed": missed,
        "sentence": sentence,
    }

    _write_predictions(df, probabilities, predicted)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(CHART_PATH, "w", encoding="utf-8") as handle:
        handle.write(_render_chart(caught, missed))
    with open(REVIEW_QUEUE_PATH, "w", encoding="utf-8") as handle:
        json.dump(_build_review_queue(df, probabilities, predicted, summary), handle, indent=2)

    print(sentence)
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {CHART_PATH}")
    print(f"Wrote {REVIEW_QUEUE_PATH}")


if __name__ == "__main__":
    build_simple_report()
