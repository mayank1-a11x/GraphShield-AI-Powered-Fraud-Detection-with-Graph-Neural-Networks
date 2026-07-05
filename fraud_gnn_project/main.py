"""End-to-end entrypoint: train a chosen GNN model and save its metrics
alongside the XGBoost baseline for comparison.

Usage:
    python main.py --model gat
"""
import argparse
import logging

import torch

from src import config
from src.train import train_gat, evaluate_loader
from src.evaluate import save_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=config.MODEL_TYPE,
                         choices=["gat", "rgcn", "hgt"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if args.model != "gat":
        raise NotImplementedError(
            "RGCN/HGT wiring is scaffolded in src/models.py and "
            "src/graph_construction.py (build_hetero_graph) — plug them into "
            "a training loop mirroring train_gat() once you're ready to "
            "extend past the GAT baseline."
        )

    model, graph, feature_cols = train_gat(device)

    from torch_geometric.loader import NeighborLoader
    test_loader = NeighborLoader(
        graph, num_neighbors=config.NUM_NEIGHBORS, batch_size=config.BATCH_SIZE,
        input_nodes=graph.test_mask, shuffle=False,
    )
    test_ap, test_f1 = evaluate_loader(model, test_loader, device)
    logger.info("Final test set — AUC-PR: %.4f | F1: %.4f", test_ap, test_f1)

    save_result(args.model, {"ap": test_ap, "f1": test_f1})


if __name__ == "__main__":
    main()
