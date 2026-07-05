"""Build a graph from IEEE-CIS tabular data.

Two graph views are supported:

1. Heterogeneous (`build_hetero_graph`): transaction nodes plus one node type
   per entity group (card, device, email, address). Edges connect a
   transaction to the entities it touches. Use this with RGCN / HGT.

2. Homogeneous projection (`build_homogeneous_graph`): transactions become
   the only node type; two transactions get a direct edge if they share any
   entity value. Simpler, faster, good first baseline with GAT.

Entity values that appear on more than MAX_ENTITY_DEGREE transactions are
dropped before edge construction — otherwise a generic value like a common
email domain creates a hub node connecting a huge fraction of the graph,
which adds noise rather than fraud signal.
"""
import logging
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, Data

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _filter_high_degree_entities(df: pd.DataFrame, col: str, max_degree: int) -> pd.Series:
    """Return a copy of df[col] with over-common values replaced by NaN."""
    counts = df[col].value_counts()
    common_values = counts[counts > max_degree].index
    series = df[col].copy()
    series[series.isin(common_values)] = np.nan
    return series


def build_hetero_graph(df: pd.DataFrame, tx_features: np.ndarray) -> HeteroData:
    """Build a heterogeneous graph: transaction node type + one node type
    per entity group, with (transaction, links_to, entity) edges.
    """
    data = HeteroData()
    n_tx = len(df)

    data["transaction"].x = torch.tensor(tx_features, dtype=torch.float)
    data["transaction"].y = torch.tensor(df["isFraud"].values, dtype=torch.long)

    for entity_type, cols in config.ENTITY_COLUMNS.items():
        cols_present = [c for c in cols if c in df.columns]
        if not cols_present:
            continue

        # Build a composite entity key from all columns in this group, e.g.
        # card1+card2+card3+card5 -> a single "card identity" string.
        composite = df[cols_present].astype(str).agg("_".join, axis=1)
        composite = _filter_high_degree_entities(
            pd.DataFrame({entity_type: composite}), entity_type, config.MAX_ENTITY_DEGREE
        )

        valid_mask = composite.notna()
        entity_values = composite[valid_mask].unique()
        entity_id_map = {v: i for i, v in enumerate(entity_values)}

        # Entity node features: just a learned embedding, so give a dummy
        # 1-dim feature (id) — the model's embedding layer will handle it.
        data[entity_type].num_nodes = len(entity_values)

        tx_idx = np.where(valid_mask.values)[0]
        entity_idx = composite[valid_mask].map(entity_id_map).values

        edge_index = torch.tensor(np.vstack([tx_idx, entity_idx]), dtype=torch.long)
        data["transaction", "links_to", entity_type].edge_index = edge_index
        data[entity_type, "linked_by", "transaction"].edge_index = edge_index.flip(0)

        logger.info(
            "Entity type '%s': %d nodes, %d edges",
            entity_type, len(entity_values), edge_index.shape[1],
        )

    logger.info("Heterogeneous graph built: %d transaction nodes", n_tx)
    return data


def build_homogeneous_graph(df: pd.DataFrame, tx_features: np.ndarray,
                             max_edges_per_entity: int = 200) -> Data:
    """Project entity-sharing relationships directly onto transaction nodes.

    Two transactions get an edge if they share a value in any entity column.
    To keep edge count tractable, entity groups with more than
    `max_edges_per_entity` members are sampled down (all-pairs would be
    combinatorial for large groups).
    """
    n_tx = len(df)
    edge_src, edge_dst = [], []

    for entity_type, cols in config.ENTITY_COLUMNS.items():
        cols_present = [c for c in cols if c in df.columns]
        if not cols_present:
            continue

        composite = df[cols_present].astype(str).agg("_".join, axis=1)
        composite = _filter_high_degree_entities(
            pd.DataFrame({entity_type: composite}), entity_type, config.MAX_ENTITY_DEGREE
        )

        groups = composite.dropna().groupby(composite.dropna()).groups
        for _, idx in groups.items():
            idx = list(idx)
            if len(idx) < 2:
                continue
            if len(idx) > max_edges_per_entity:
                # cap: connect each member to a random subset rather than
                # full all-pairs, to avoid an O(n^2) blowup on hub entities
                rng = np.random.default_rng(config.SEED)
                idx = list(rng.choice(idx, size=max_edges_per_entity, replace=False))
            for a, b in combinations(idx, 2):
                edge_src.append(a)
                edge_dst.append(b)

    edge_index = torch.tensor([edge_src + edge_dst, edge_dst + edge_src], dtype=torch.long)

    data = Data(
        x=torch.tensor(tx_features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(df["isFraud"].values, dtype=torch.long),
    )
    logger.info(
        "Homogeneous graph built: %d nodes, %d edges (%.1f avg degree)",
        n_tx, edge_index.shape[1], edge_index.shape[1] / n_tx,
    )
    return data


def make_train_val_test_masks(n: int, seed: int = config.SEED):
    """Random split respecting config ratios. For a real project consider a
    time-based split instead (train on earlier transactions, test on later
    ones) since fraud patterns drift over time."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[idx[:n_train]] = True
    val_mask[idx[n_train:n_train + n_val]] = True
    test_mask[idx[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask
