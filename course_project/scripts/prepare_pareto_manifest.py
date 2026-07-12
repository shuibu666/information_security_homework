#!/usr/bin/env python
"""Add the newly generated 500-prompt CA-KL epsilon runs to a Pareto manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_manifest", required=True)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.20, 0.30, 0.35, 0.40, 0.50])
    args = parser.parse_args()

    manifest = json.loads(Path(args.template_manifest).read_text(encoding="utf-8"))
    # Replace the older epsilon=0.50 source with fresh runs that also contain
    # exact generation token ids and per-sequence delta distribution metrics.
    manifest["sources"] = [source for source in manifest["sources"] if source.get("family") != "CA-KL"]
    raw_dir = Path(args.raw_dir)
    for epsilon in args.epsilons:
        tag = f"{epsilon:.2f}".replace(".", "")
        prefix = raw_dir / f"cakl_opt27b_c4_500_eps{tag}"
        manifest["sources"].append(
            {
                "source_id": f"cakl_eps{tag}",
                "name": f"CA-KL epsilon {epsilon:.2f}",
                "family": "CA-KL",
                "parameter": f"epsilon={epsilon:.2f}",
                "epsilon": epsilon,
                "method": "CA-KL",
                "results_csv": str(prefix) + "_results.csv",
                "ppl_csv": str(prefix) + "_ppl_results.csv",
            }
        )
    Path(args.output_manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Pareto manifest to {args.output_manifest}")


if __name__ == "__main__":
    main()
