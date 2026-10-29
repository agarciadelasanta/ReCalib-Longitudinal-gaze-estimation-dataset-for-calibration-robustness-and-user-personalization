# Hierarchical dataset statistical analysis

The `statistical_analysis` package compares participants and sessions using the
flattened ReCalib sample metadata. It is intentionally independent of model
predictions. Its outputs describe dataset variability; they do not claim that a
particular difference causes model error or negative transfer.

## Quick start

Run the complete descriptive analysis without resampling:

```bash
python -m statistical_analysis dataset_summary_recalib.csv \
  --output temp/statistical_analysis \
  --bootstrap 0 \
  --permutations 0
```

Run inferential analysis for selected feature families:

```bash
python -m statistical_analysis dataset_summary_recalib.csv \
  --output temp/statistical_analysis_head_gaze \
  --families head_rotation head_translation gaze_direction \
  --bootstrap 1000 \
  --permutations 5000 \
  --seed 42
```

The CLI defaults to 500 bootstrap and 1,000 permutation iterations. Runtime is
approximately linear in the number of selected families and resampling
iterations. Use the descriptive command to inspect all families quickly, then
run high-resolution inference for the families needed in a final analysis.

Use `python -m statistical_analysis --help` for every option.

## Interactive explorer

After generating a report bundle, create and open a standalone explorer with:

```bash
python visualization/explore_statistical_analysis.py \
  --report-dir temp/statistical_analysis \
  --open
```

The generated `statistical_analysis_explorer.html` is self-contained and can be
opened without running a server. It provides linked filters for feature family,
KPI, task, view, and matrix statistic, with the target-conditioned view selected
by default when it is available. The plots cover:

- participant-by-participant KPI matrices;
- each participant's mean distance to all peers;
- within-participant longitudinal session instability;
- per-session calibration-to-test mismatch distributions;
- the exact pairwise estimate, confidence interval, `p`, and `q` values.

An exploratory report created with `--bootstrap 0 --permutations 0` exposes
estimates only. Confidence interval, `p`, and `q` selections become available
when the corresponding inferential outputs contain values.

See [How to Interpret the Statistical Analysis Explorer](statistical_analysis_explorer_interpretation.md)
for a plot-by-plot reading guide and the main interpretation limitations.

## Analysis population and hierarchy

- Rows where `discarded=True` are excluded from every calculation and output.
- The 9-point calibration and 16-point test tasks are analyzed separately.
- Each recording session receives equal weight within its participant.
- Frames are not treated as independent replicates.
- The raw view describes the observed data distribution.
- The primary target-conditioned view compares behavior at shared normalized
  screen targets. Target coordinates are divided by the effective
  `2736 x 1824` pixel surface and rounded to four decimal places for matching.
- Target coverage is meaningful only in the raw view.

For each session, or each session-target cell in the conditioned view, the tool
calculates robust descriptors: the median, IQR, 10/25/50/75/90 percentiles, and
Spearman dependence structure. User comparisons operate on those equally
weighted hierarchical descriptors rather than on a pooled collection of
correlated frames.

## Feature families

| Family | Primary representation |
| --- | --- |
| `target_coverage` | Screen target coordinates normalized to `[0, 1]` |
| `gaze_direction` | Unit gaze vector converted to pitch/yaw radians |
| `gaze_origin` | 3D origin in millimetres |
| `gaze_point` | Screen-plane gaze point, with constant `z` removed |
| `head_rotation` | Joint Euler-angle vector in radians |
| `head_translation` | Joint translation vector in millimetres |
| `landmark_position` | Facial-landmark centroid in pixels |
| `landmark_scale` | RMS landmark radius in pixels |
| `landmark_shape` | Centred, RMS-scale-normalized landmark coordinates |
| `quality` | Joint QA measurements in their native definitions |

Original columns remain auditable through `column_catalog.csv`. On the current
CSV, camera rotation, image/screen dimensions, gender, age, distance, and
contact-lens attributes are missing on every accepted row. They are reported as
unusable and are not silently imputed.

## KPI vector

There is deliberately no global domain-gap score in Phase 1. Every feature
family receives a separate vector:

- `location_shift`: Euclidean difference between session-balanced robust
  locations after per-dimension robust scaling and dimension normalization.
- `dispersion_shift`: dimension-normalized magnitude of the log ratio between
  robust IQRs.
- `distribution_shift`: normalized Wasserstein-1 for scalar families, or
  robust-scaled multivariate energy distance for multidimensional families.
- `dependence_shift`: dimension-normalized distance between Spearman
  correlation structures. It is not applicable to scalar families and is
  therefore `NaN` for `landmark_scale`.

All distances are non-negative, with zero representing equality under the
selected descriptor. They are standardized effect distances, not probabilities
and not artificially bounded to `[0, 1]`.

The user profile additionally reports the median and IQR of each session's
distance from its participant reference. Calibration-to-test mismatch is
reported per session and as an equal-session user mean.

## Statistical inference

- Confidence intervals use clustered bootstrap resampling of whole sessions.
- Permutation tests exchange whole-session participant labels.
- `p_value` is secondary to effect magnitude and its confidence interval.
- Benjamini-Hochberg `q_value` correction is performed separately within every
  view/task/family/KPI matrix.
- The random seed makes the complete result deterministic.

When resampling is disabled, confidence intervals and `p/q` values are left
missing rather than replaced with misleading values.

## Report bundle

| Output | Contents |
| --- | --- |
| `manifest.json` | Input, configuration, row counts, and generated files |
| `pairwise_distances.csv` | Long-form pairwise KPI estimates and inference |
| `user_profiles.csv` | Within-user longitudinal instability |
| `calibration_test_mismatch.csv` | Per-session and user-level task mismatch |
| `feature_catalog.csv` | Derived feature definitions and source columns |
| `column_catalog.csv` | Usage, missingness, cardinality, and exclusion reason for every input column |
| `matrix_index.csv` | Lookup table for matrix files |
| `matrices/.../*.csv` | Symmetric user-by-user estimate matrices |

## Phase 1 boundary

These results can nominate unusual users, sessions, and feature families. They
cannot yet explain why a model benefits or suffers from adaptation. That claim
requires Phase 2 to join held-out baseline and adapted predictions to the same
user/session/sample identities and test whether these pre-analysis KPIs predict
negative transfer.
