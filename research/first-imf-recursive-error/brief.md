# Investigation Brief

## Question

Why is the first recursive IMF error much larger than stages 2--8 in both the linear and robust runs, and does the relevant theory actually predict a constant scaled error across recursive IMF components?

## Bounded Tasks

1. **Code-path audit:** derive the exact stage-wise error operators implemented in the notebook and check definitions, indexing, bandwidth schedule, normalization, boundaries, and reference/calculated comparability.
2. **Numerical diagnosis:** reproduce the reported metrics and run targeted checks that isolate first-stage effects, including theoretical variance calculations, residual-filter effects, boundary effects, Monte Carlo variability, and robust-versus-linear behavior.
3. **Theory audit:** locate the source of the `a^(k/2)` scaling, record the exact theorem or heuristic, assumptions, error norm, estimator, and whether it applies to single-pass smoothers or recursively extracted IMF components.
4. **Synthesis/review:** reconcile theory and experiment, identify bugs versus expected finite-sample behavior, and recommend precise validation steps without modifying the original notebook.

## Success Criteria

- Reproduce the first-stage values in the notebook.
- Give an algebraic explanation for stage 1 versus later recursive stages.
- Quantify how much of the effect is predicted by the implemented linear filters.
- Test whether the scale factor and stage indexing match the stated theory.
- State a qualified verdict on “scaled error should be constant,” including norm, asymptotic/finite-sample status, and required assumptions.
- Leave a runnable, non-destructive diagnostic artifact plus concise evidence memos.

## Expected Outputs

- `agents/code-path.md`
- `agents/numerical-diagnosis.md`
- `agents/theory.md`
- `review.md`
- `executive-summary.md`
- optional diagnostic tables/figures and a companion notebook in this research directory
