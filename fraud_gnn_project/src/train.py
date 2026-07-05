"""Train a fraud-detection GNN.

Supports the homogeneous GAT path out of the box. RGCN/HGT paths need the
heterogeneous graph's edges converted to a single relation-typed edge_index
(for RGCN) or kept as dicts (for HGT, via PyG's HGTLoader) — see the
`# TODO(heterogeneous)` markers below for where to plug that in once you've
built the hetero graph for your run.
"""
import argparse
import logging

import torch
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import average_precision_score, f1_score

from src import config
from src.data_preprocessing import run_preprocessing, get_feature_matrix
from src.graph_construction import build_homogeneous_graph, make_train_val_test_masks
from src.models import build_model, FocalLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_gat(device: torch.device):
    df = run_preprocessing()
    X, feature_cols = get_feature_matrix(df)
    graph = build_homogeneous_graph(df, X)

    train_mask, val_mask, test_mask = make_train_val_test_masks(graph.num_nodes)
    graph.train_mask, graph.val_mask, graph.test_mask = train_mask, val_mask, test_mask

    train_loader = NeighborLoader(
        graph, num_neighbors=config.NUM_NEIGHBORS, batch_size=config.BATCH_SIZE,
        input_nodes=graph.train_mask, shuffle=True,
    )
    val_loader = NeighborLoader(
        graph, num_neighbors=config.NUM_NEIGHBORS, batch_size=config.BATCH_SIZE,
        input_nodes=graph.val_mask, shuffle=False,
    )

    model = build_model(
        "gat", in_dim=X.shape[1], hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS, num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

    best_val_ap = 0.0
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            # only compute loss on the "seed" nodes of this batch, not the
            # sampled neighbors pulled in for message passing
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.batch_size

        val_ap, val_f1 = evaluate_loader(model, val_loader, device)
        logger.info(
            "Epoch %02d | train_loss=%.4f | val_AP=%.4f | val_F1=%.4f",
            epoch, total_loss / int(train_mask.sum()), val_ap, val_f1,
        )

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            torch.save(model.state_dict(), f"{config.CHECKPOINT_DIR}/gat_best.pt")
            logger.info("  -> new best model saved (val_AP=%.4f)", val_ap)

    return model, graph, feature_cols


@torch.no_grad()
def evaluate_loader(model, loader, device) -> tuple[float, float]:
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        probs = torch.softmax(out[:batch.batch_size], dim=1)[:, 1]
        all_probs.append(probs.cpu())
        all_labels.append(batch.y[:batch.batch_size].cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    ap = average_precision_score(labels, probs)
    f1 = f1_score(labels, probs > 0.5)
    return ap, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=config.MODEL_TYPE,
                         choices=["gat", "rgcn", "hgt"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if args.model == "gat":
        train_gat(device)
    else:
        # TODO(heterogeneous): build_hetero_graph(...) then either:
        #   - RGCN: convert edges to (edge_index, edge_type) via
        #     HeteroData.to_homogeneous(), pass num_relations to build_model
        #   - HGT: use torch_geometric.loader.HGTLoader over the HeteroData
        #     directly, mirroring the loop above but with x_dict/edge_index_dict
        raise NotImplementedError(
            f"'{args.model}' training loop needs the heterogeneous graph "
            "wiring described in the TODO above — the GAT path is fully "
            "runnable now; RGCN/HGT reuse build_hetero_graph() plus this "
            "same train/eval structure."
        )


if __name__ == "__main__":
    main()
