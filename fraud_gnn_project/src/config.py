"""Central configuration for the fraud GNN pipeline."""
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TRANSACTION_CSV = os.path.join(DATA_DIR, "train_transaction.csv")
IDENTITY_CSV = os.path.join(DATA_DIR, "train_identity.csv")
WORKBOOK_PATH = os.path.join(DATA_DIR, "fraud_synthetic_dataset.xlsx")

# ---- Reproducibility ----
SEED = 42

# ---- Graph construction ----
# Columns treated as "entity" node types. Transactions sharing a value in one
# of these columns get linked (directly, in the homogeneous projection) or
# both connect to a shared entity node (in the heterogeneous graph).
ENTITY_COLUMNS = {
    "card": ["card1", "card2", "card3", "card5"],
    "device": ["DeviceType", "DeviceInfo"],
    "email": ["P_emaildomain", "R_emaildomain"],
    "address": ["addr1", "addr2"],
}

# Max number of transactions a single entity value can connect before it's
# dropped (extremely common values, like a generic email domain, create huge
# hub nodes that add noise rather than signal).
MAX_ENTITY_DEGREE = 500

# ---- Feature engineering ----
NUMERIC_TRANSACTION_COLS = [
    "TransactionAmt", "card1", "card2", "card3", "card5",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15",
]

# ---- Model / training hyperparameters ----
MODEL_TYPE = "gat"          # "gat" | "rgcn" | "hgt"
HIDDEN_DIM = 128
NUM_LAYERS = 3
NUM_HEADS = 4               # for GAT / HGT attention heads
DROPOUT = 0.3

BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10, 5]  # per-layer neighbor sampling fanout
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4

# Focal loss params (helps with ~3.5% fraud class imbalance)
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# ---- Train/val/test split ----
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---- Evaluation ----
RECALL_AT_K = [100, 500, 1000]  # top-k transactions reviewed, by predicted risk
