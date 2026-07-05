# Fraud Detection with Graph Neural Networks

A GNN-based fraud detection pipeline built on the IEEE-CIS Fraud Detection dataset.
Transactions and the entities they touch (cards, devices, email domains, addresses)
are modeled as a heterogeneous graph, so fraud signal can propagate through shared
entities (e.g., five transactions on the same stolen card) instead of treating each
transaction as independent.

## Project structure

```
fraud_gnn_project/
├── data/                      # put IEEE-CIS CSVs here (not included)
├── src/
│   ├── config.py              # central config: paths, hyperparams
│   ├── data_preprocessing.py  # load + clean raw CSVs, feature engineering
│   ├── graph_construction.py  # build a PyG HeteroData graph
│   ├── models.py               # GAT baseline + RGCN + HGT model classes
│   ├── train.py                # training loop (mini-batch, neighbor sampling)
│   ├── evaluate.py             # AUC-PR, F1, recall@k, plots
│   └── baseline_xgboost.py     # non-graph baseline for comparison
├── main.py                     # run the full pipeline end-to-end
├── requirements.txt
└── checkpoints/                # saved models land here
```

## 1. Get the data

Download from Kaggle: https://www.kaggle.com/c/ieee-fraud-detection/data

You need:
- `train_transaction.csv`
- `train_identity.csv`

Place both in `data/`.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch Geometric install can be finicky — if `pip install torch_geometric` fails,
follow the official instructions for your CUDA/CPU version:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## 3. Run the pipeline

```bash
# Step 1: baseline (no graph) — establishes the control to beat
python -m src.baseline_xgboost

# Step 2: train the GNN (choose model in config.py: "gat", "rgcn", or "hgt")
python main.py --model gat
python main.py --model rgcn
python main.py --model hgt

# Step 3: compare
python -m src.evaluate --compare
```

## Recommended build order (also matches the code's design)

1. **XGBoost baseline** on raw tabular features — your "no graph" control.
2. **GAT on a homogeneous projected graph** — collapse entity nodes into
   transaction–transaction edges (shared card/device/email → edge), use
   attention to weigh which shared connections matter most. Fast, interpretable,
   good first GNN result.
3. **RGCN / HGT on the full heterogeneous graph** — keep transaction, card,
   device, email, and address as distinct node types with distinct relations.
   This is where the "advanced" contribution lives, and gives you a clean
   ablation story: tabular → homogeneous GNN → heterogeneous GNN.

## Handling class imbalance

Fraud is ~3.5% of transactions. The code uses:
- Class-weighted / focal loss (see `src/models.py::FocalLoss`)
- Neighbor sampling via PyG `NeighborLoader` / `HGTLoader` (full graph doesn't
  fit in memory as dense batches anyway)
- Evaluation on **AUC-PR, F1 (fraud class), recall@k** — not accuracy, which is
  meaningless at this imbalance ratio

## Notes on scale

IEEE-CIS has ~590K transactions. Full-batch training on the heterogeneous graph
will not fit in GPU memory for most setups — the training loop uses mini-batch
neighbor sampling by default. If you're on CPU only, drop `hidden_dim` in
`config.py` and reduce `batch_size`.
