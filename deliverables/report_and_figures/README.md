# Final Report Submission

This directory is the final report submission package. It contains the compiled
course report, its LaTeX source, embedded figures, and the result artifacts
needed to verify the reported controlled experiments.

- `report.pdf`: current compiled course report.
- `report_source.tex` and `report_source.pdf`: source and matching compiled PDF
  for editors that preview a PDF with the same base name.
- `figures/`: figures embedded by the report.
- `additional_figures/`: complete Pareto, copy-paste, and controlled-run figures.
- `calibration/`: previous calibration outputs plus `paper_controlled/`, the
  five-fold calibration and bootstrap outputs for the controlled rerun.
- `results/`: Pareto/copy-paste summaries and the controlled-run per-sample
  generation and OPT-2.7B PPL CSV files.

The report additionally incorporates the completed controlled rerun: OPT-1.3B,
499 effective C4/realnewslike prompts, 200+-5 tokens, standard KGW detection,
and continuation PPL scored by unwatermarked OPT-2.7B. The copy-paste TPR@FPR
values are clearly labelled as diagnostics based on the constructed 0% negative
samples; controlled-run detector claims use separate five-fold calibration.

To preview, open `report.pdf` or `report_source.pdf` directly. Do not use the
Code Runner play button on the `.tex` file: it reports that LaTeX is an
unsupported code language. In VS Code, use a LaTeX extension's ``View LaTeX
PDF`` command after compiling with XeLaTeX.
