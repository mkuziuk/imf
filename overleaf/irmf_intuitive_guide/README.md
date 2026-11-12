# IRMF: the proof in five steps

[Read the five-page guide](irmf_intuitive_guide.pdf).

The guide keeps the proof notation and diagrams and omits worked numerical
arithmetic. Short explanations at the point of use motivate the deterministic
reference, curvature replacement, error splits, centering, and accumulated
remainder. It covers:

1. Score, curvature, and the Fisher leading term in Proposition 4.1.
2. The Newton and curvature-replacement contributions to its remainder.
3. The exact recursion and the distinction between `R`, `Phi`, and `Q`.
4. Quadratic cancellation and remainder feedback.
5. Induction closure and the population-target correction.

The final bound is conditional on the assumptions in the
[full theorem](../irmf_all_components_bound.pdf).

## Rebuild

`main.tex` contains all text and vector diagrams. No generated data file is needed.

```bash
mkdir -p /tmp/imf-intuitive-guide-build
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/imf-intuitive-guide-build main.tex
cp /tmp/imf-intuitive-guide-build/main.pdf irmf_intuitive_guide.pdf
```

For Overleaf, upload `main.tex`. Companion links use the repository's relative paths.

## Earlier numerical example

`worked_example.json` and `example_data.tex` retain the previous eight-observation
calculation. They are not used by this guide. Regenerate them from the repository root:

```bash
.venv/bin/python research/recursive-component-bounds/build_concrete_guide.py
```

## Validation

Compiled with local `latexmk` and the desktop compiler. The five pages were
rendered and visually checked; the TeX log has no warnings or overfull boxes.
