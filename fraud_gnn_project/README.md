# GraphShield: Fraud Review Queue

GraphShield is a fraud-detection project that turns transaction data into an
easy-to-read analyst dashboard. The current dashboard focuses on practical
review decisions instead of model jargon.

## Project Structure

```text
fraud_gnn_project/
├── data/
│   └── fraud_synthetic_dataset.xlsx
├── outputs/
│   ├── fraud_caught_missed.svg
│   └── review_queue.json
├── src/
│   ├── baseline_xgboost.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── graph_construction.py
│   ├── models.py
│   ├── simple_report.py
│   └── train.py
├── tests/
│   └── test_data_preprocessing.py
├── index.html
├── main.py
├── README.md
└── requirements.txt
```

## Dashboard

Open the dashboard from a local static server:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

Then visit:

```text
http://127.0.0.1:8765/index.html
```

The dashboard shows:

- Fraud caught, fraud missed, and transactions reviewed
- A plain-English summary sentence
- A simple fraud caught vs missed chart
- Confidence bands
- Top 20 suspicious transactions with a reason flagged

## Regenerate Dashboard Data

From the project folder:

```bash
python -m src.simple_report
```

This recreates:

- `outputs/review_queue.json`
- `outputs/fraud_caught_missed.svg`
- `outputs/prediction_results.csv`
- `outputs/simple_summary.json`

The CSV and simple summary are treated as generated exports and are ignored by
Git.

## Model Pipeline

The full GNN pipeline is still available:

```bash
python -m src.baseline_xgboost
python main.py --model gat
python -m src.evaluate --compare
```
