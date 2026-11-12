# Understanding the all-components IRMF bound

For a shorter explanation with proof notation and diagrams, start with the
[five-page guide](../irmf_intuitive_guide/irmf_intuitive_guide.pdf).

[Read the 12-page PDF](irmf_all_components_guide.pdf). Edit `main.tex` to change
the guide. The diagrams and charts use TikZ and PGFPlots, so the PDF stays sharp
when zoomed or printed. `plot_data.tex` contains the generated plot coordinates.

The guide teaches the restricted theorem in
[`irmf_all_components_bound.tex`](../irmf_all_components_bound.tex). It keeps the
added assumptions explicit. The submitted n=1000 experiment does not satisfy
the theorem's conservative sample-size condition.

## Rebuild

From the repository root, regenerate the plot coordinates from the existing
audit results:

```bash
.venv/bin/python research/recursive-component-bounds/build_visual_guide.py
```

This reads the audit CSV/JSON files and calculates the linear frequency
responses. It does not rerun or modify the Monte Carlo experiments.

Compile from this directory, keeping intermediate files outside the project:

```bash
mkdir -p /tmp/imf-visual-guide-build
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/imf-visual-guide-build main.tex
cp /tmp/imf-visual-guide-build/main.pdf irmf_all_components_guide.pdf
```

For Overleaf, upload `main.tex` and `plot_data.tex` and select `main.tex` as the
main document. The two links to the companion proof PDFs work when those files
are retained in the parent directory, as in this repository.

## Validation

The delivered PDF has 12 pages. It was compiled with `latexmk`, checked for
warnings and overfull boxes, and rendered with Poppler for page-by-page visual
inspection. The numerical plots use the audited archive summaries; they are
illustrations and finite-design checks, not interval-certified proof constants.
