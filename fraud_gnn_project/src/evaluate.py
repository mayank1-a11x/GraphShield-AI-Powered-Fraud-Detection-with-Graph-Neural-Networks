"""Evaluation utilities: metrics used consistently across baseline and GNN
models, plus a comparison plot.

Run with --compare after you've trained both the baseline and at least one
GNN model, to produce a side-by-side chart in outputs/.
"""
import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join(config.OUTPUT_DIR, "results.json")


def save_result(name: str, metrics: dict):
    """Append a model's metrics to the shared results file so --compare can
    plot everything that's been run so far."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    results[name] = metrics
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results for '%s' -> %s", name, RESULTS_PATH)


def plot_comparison():
    if not os.path.exists(RESULTS_PATH):
        logger.warning("No results file found at %s — train models first.", RESULTS_PATH)
        return

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    names = list(results.keys())
    aps = [results[n].get("ap", 0) for n in names]
    f1s = [results[n].get("f1", 0) for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, aps, width, label="AUC-PR")
    ax.bar(x + width / 2, f1s, width, label="F1 (fraud class)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Fraud detection: tabular vs. graph models")
    ax.legend()
    ax.set_ylim(0, 1)

    out_path = os.path.join(config.OUTPUT_DIR, "model_comparison.svg")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    logger.info("Saved comparison plot -> %s", out_path)

    print("\n=== Model comparison ===")
    for n in names:
        print(f"{n:12s}  AUC-PR={results[n].get('ap', 0):.4f}  F1={results[n].get('f1', 0):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true",
                         help="Plot comparison across all saved results")
    args = parser.parse_args()

    if args.compare:
        plot_comparison()


if __name__ == "__main__":
    main()
