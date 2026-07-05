"""Generate a lightweight SVG summary of the transaction graph."""
import json
import math
import os
from collections import defaultdict
from itertools import combinations

import pandas as pd

from src import config
from src.data_preprocessing import run_preprocessing


SUMMARY_PATH = os.path.join(config.OUTPUT_DIR, "graph_summary.json")
SVG_PATH = os.path.join(config.OUTPUT_DIR, "transaction_graph.svg")


def _entity_composite(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [col for col in cols if col in df.columns]
    if not present:
        return pd.Series(index=df.index, dtype=object)
    return df[present].astype(str).agg("_".join, axis=1)


def summarize_graph(df: pd.DataFrame) -> dict:
    entity_stats = {}
    edge_pairs = set()
    sample_links = defaultdict(set)

    for entity_type, cols in config.ENTITY_COLUMNS.items():
        composite = _entity_composite(df, cols)
        composite = composite[composite.notna()]
        if composite.empty:
            continue

        counts = composite.value_counts()
        usable = counts[counts <= config.MAX_ENTITY_DEGREE]
        groups = composite[composite.isin(usable.index)].groupby(composite).groups

        relation_edges = 0
        for value, indexes in groups.items():
            members = list(indexes)
            if len(members) < 2:
                continue
            relation_edges += len(members) * (len(members) - 1)
            for a, b in combinations(members[:25], 2):
                edge_pairs.add(tuple(sorted((int(a), int(b)))))
            if len(sample_links[entity_type]) < 6:
                for member in members[:3]:
                    sample_links[entity_type].add(int(member))

        entity_stats[entity_type] = {
            "columns": [col for col in cols if col in df.columns],
            "unique_entities": int(counts.size),
            "usable_entities": int(usable.size),
            "dropped_hub_entities": int(counts.size - usable.size),
            "transaction_entity_edges": int(len(composite[composite.isin(usable.index)])),
            "projected_directed_edges": int(relation_edges),
        }

    summary = {
        "transactions": int(len(df)),
        "fraud_transactions": int(df["isFraud"].sum()),
        "fraud_rate": float(df["isFraud"].mean()),
        "entity_types": entity_stats,
        "sample_transaction_edges": len(edge_pairs),
        "sample_links": {key: sorted(value) for key, value in sample_links.items()},
    }
    return summary


def _svg_text(x, y, text, size=14, weight="400", fill="#111827"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'font-family="Arial, sans-serif" fill="{fill}">{text}</text>'
    )


def render_svg(summary: dict) -> str:
    width, height = 920, 520
    cx, cy = 450, 260
    radius = 165
    colors = {
        "card": "#2563eb",
        "device": "#16a34a",
        "email": "#dc2626",
        "address": "#9333ea",
    }

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Transaction graph generated from dataset">',
        '<rect width="920" height="520" fill="#f8fafc"/>',
        _svg_text(32, 42, "Transaction Graph From Provided Dataset", 24, "700"),
        _svg_text(
            32,
            70,
            f'{summary["transactions"]:,} transactions | '
            f'{summary["fraud_transactions"]:,} fraud '
            f'({summary["fraud_rate"] * 100:.2f}%)',
            14,
            "400",
            "#475569",
        ),
        '<g opacity="0.95">',
    ]

    entity_items = list(summary["entity_types"].items())
    for idx, (entity_type, stats) in enumerate(entity_items):
        angle = (2 * math.pi * idx / max(len(entity_items), 1)) - math.pi / 2
        ex = cx + radius * math.cos(angle)
        ey = cy + radius * math.sin(angle)
        color = colors.get(entity_type, "#64748b")
        parts.extend(
            [
                f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{color}" stroke-width="3" stroke-opacity="0.45"/>',
                f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="48" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="3"/>',
                _svg_text(ex - 32, ey - 4, entity_type.title(), 15, "700", color),
                _svg_text(ex - 34, ey + 18, f'{stats["usable_entities"]:,} usable', 12, "400", "#475569"),
            ]
        )

    parts.extend(
        [
            f'<circle cx="{cx}" cy="{cy}" r="74" fill="#fff7ed" stroke="#f97316" stroke-width="4"/>',
            _svg_text(cx - 51, cy - 8, "Transactions", 17, "700", "#9a3412"),
            _svg_text(cx - 43, cy + 18, f'{summary["transactions"]:,} nodes', 13, "400", "#9a3412"),
        ]
    )

    legend_x, legend_y = 650, 118
    parts.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="230" height="286" rx="8" fill="#ffffff" stroke="#cbd5e1"/>',
            _svg_text(legend_x + 18, legend_y + 32, "Entity Edges", 17, "700"),
        ]
    )
    y = legend_y + 64
    for entity_type, stats in entity_items:
        color = colors.get(entity_type, "#64748b")
        parts.extend(
            [
                f'<circle cx="{legend_x + 24}" cy="{y - 4}" r="7" fill="{color}"/>',
                _svg_text(legend_x + 42, y, entity_type.title(), 14, "700"),
                _svg_text(legend_x + 42, y + 20, f'{stats["transaction_entity_edges"]:,} tx-entity edges', 12, "400", "#475569"),
                _svg_text(legend_x + 42, y + 38, f'{stats["dropped_hub_entities"]:,} high-degree hubs dropped', 12, "400", "#64748b"),
            ]
        )
        y += 58

    parts.extend(
        [
            "</g>",
            _svg_text(32, 486, "Edges connect transactions that share card, device, email, or address values after high-degree hub filtering.", 13, "400", "#475569"),
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df = run_preprocessing()
    summary = summarize_graph(df)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(SVG_PATH, "w", encoding="utf-8") as handle:
        handle.write(render_svg(summary))
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
