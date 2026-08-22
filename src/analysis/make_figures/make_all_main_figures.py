#!/usr/bin/env python3
"""Generate all currently supported main task figures."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import parse_formats
from make_np_likeness_figure import make_figure as make_np_likeness
from make_np_classification_figure import make_figure as make_np_classification
from make_pmn_secondary_pathway_figure import make_figure as make_pmn
from make_kegg_pathway_figure import make_figure as make_kegg
from make_plantcyc_reaction_neighbor_figure import make_figure as make_plantcyc
from make_pathway_ensemble_figure import make_figure as make_pathway_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--curve-uncertainty",
        choices=["sd", "se", "ci90", "ci95", "none"],
        default="ci95",
        help="Uncertainty ribbon for all label-efficiency curves",
    )
    parser.add_argument(
        "--include-pathway-ensemble-figure",
        action="store_true",
        help="Also generate the PMN/KEGG ensemble showcase figure",
    )
    args = parser.parse_args()

    formats = parse_formats(args.formats)
    outputs = []
    outputs.extend(
        make_np_likeness(
            args.project_root,
            args.output_dir,
            formats,
            args.dpi,
            args.curve_uncertainty,
        )
    )
    outputs.extend(
        make_np_classification(
            args.project_root,
            args.output_dir,
            formats,
            args.dpi,
            args.curve_uncertainty,
        )
    )
    outputs.extend(
        make_pmn(
            args.project_root,
            args.output_dir,
            formats,
            args.dpi,
            args.curve_uncertainty,
        )
    )

    kegg_full = args.project_root / "results/kegg_pathway/full/kegg_pathway_full_ensemble_repeated_runs.tsv"
    kegg_eff = args.project_root / "results/kegg_pathway/data_efficiency/kegg_pathway_data_efficiency_repeated_runs.tsv"
    if kegg_full.exists() and kegg_eff.exists():
        outputs.extend(
            make_kegg(
                args.project_root,
                args.output_dir,
                formats,
                args.dpi,
                False,
                args.curve_uncertainty,
            )
        )
        if args.include_pathway_ensemble_figure:
            outputs.extend(
                make_pathway_ensemble(
                    args.project_root,
                    args.output_dir,
                    formats,
                    args.dpi,
                )
            )
    else:
        print("SKIP KEGG pathway figure: current results/kegg_pathway full and data-efficiency tables are not both present")
        if args.include_pathway_ensemble_figure:
            print("SKIP pathway ensemble figure: current KEGG results are required")

    outputs.extend(make_plantcyc(args.project_root, args.output_dir, formats, args.dpi))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
