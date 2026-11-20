
#!/usr/bin/env python3
"""Compare original and property-matched NP-likeness results."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAT_NAME = "np_likeness_full_ensemble_main_statistics.tsv"

MODEL_NAMES = {
    "chemberta_100m_mlm": "ChemBERTa-100M",
    "chemberta_77m_mlm": "ChemBERTa-77M",
    "molformer_c3_1_1b": "MoLFormer",
    "nafm_graph_mean": "NaFM",
    "molclr_gcn": "MolCLR-GCN",
    "molclr_gin": "MolCLR-GIN",
    "graphmvp_gin": "GraphMVP-GIN",
    "molebert_gin": "Mole-BERT",
    "morgan_r2_2048": "Morgan (r=2)",
    "morgan_r3_2048": "Morgan (r=3)",
    "mw_logp_rings": "MW + logP + rings",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--original",
        default=f"results/np_likeness/full/{STAT_NAME}",
    )
    p.add_argument(
        "--original-descriptors",
        default=f"results/np_likeness/original_descriptors/{STAT_NAME}",
    )
    p.add_argument(
        "--matched",
        default=f"results/np_likeness/matched/full/{STAT_NAME}",
    )
    p.add_argument(
        "--output",
        default="results/np_likeness/matched/original_vs_matched_metrics.tsv",
    )
    p.add_argument(
        "--figure",
        default="results/figures/supplementary/np_likeness_original_vs_matched.png",
    )
    return p.parse_args()


def load(path):
    df = pd.read_csv(path, sep="\t")
    df = df[df["feature_type"].ne("probability_ensemble")].copy()
    return df[
        ["feature", "split_mode", "auroc_mean", "auprc_mean"]
    ].drop_duplicates(["feature", "split_mode"])


def main():
    args = parse_args()

    original = load(args.original)

    if Path(args.original_descriptors).exists():
        desc = load(args.original_descriptors)
        desc = desc[desc["feature"].eq("mw_logp_rings")]
        original = pd.concat(
            [original, desc], ignore_index=True
        ).drop_duplicates(
            ["feature", "split_mode"], keep="last"
        )

    matched = load(args.matched)

    comparison = original.merge(
        matched,
        on=["feature", "split_mode"],
        how="inner",
        suffixes=("_original", "_matched"),
        validate="one_to_one",
    )

    if comparison.empty:
        raise RuntimeError("No features in common.")

    for metric in ("auroc", "auprc"):
        comparison[f"{metric}_difference"] = (
            comparison[f"{metric}_mean_matched"]
            - comparison[f"{metric}_mean_original"]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    comparison.sort_values(
        ["split_mode", "feature"]
    ).to_csv(output, sep="\t", index=False)

    print("Wrote", output)

    print(
        comparison[
            [
                "feature",
                "split_mode",
                "auroc_mean_original",
                "auroc_mean_matched",
            ]
        ].to_string(index=False)
    )

    # AUROC-only figure, two vertically stacked panels.
    n_models = comparison["feature"].nunique()

    fig, axs = plt.subplots(
        2, 1,
        figsize=(9.5, max(8, n_models * 0.85 + 3)),
        sharex=True,
        constrained_layout=True,
    )

    for ax, mode in zip(axs, ("random", "scaffold")):
        sub = comparison[
            comparison["split_mode"].eq(mode)
        ].sort_values(
            "auroc_mean_matched", ascending=False
        )

        y = np.arange(len(sub))
        labels = [
            MODEL_NAMES.get(f, f)
            for f in sub["feature"]
        ]

        ax.scatter(
            sub["auroc_mean_original"],
            y - 0.13,
            s=48,
            marker="o",
            color="tab:blue",
            label="Original",
            zorder=3,
        )

        ax.scatter(
            sub["auroc_mean_matched"],
            y + 0.13,
            s=55,
            marker="x",
            linewidths=1.8,
            color="tab:orange",
            label="Property-matched",
            zorder=3,
        )

        ax.axvline(
            0.5,
            linestyle="--",
            linewidth=1,
            color="gray",
            alpha=0.7,
        )

        ax.set_yticks(y, labels=labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("AUROC")
        ax.set_title(f"{mode.title()} split", fontsize=13)

        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    axs[0].tick_params(axis="x", labelbottom=True)
    axs[0].legend(loc="upper left", frameon=False)

    fig_path = Path(args.figure)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Wrote", fig_path)




if __name__ == "__main__":
    main()
