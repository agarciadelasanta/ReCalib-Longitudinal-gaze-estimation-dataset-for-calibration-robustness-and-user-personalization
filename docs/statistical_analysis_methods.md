# Statistical analyses and methods

#recalib #statistical-analysis #statistical-methods #gaze-estimation #dataset-analysis

This document is the canonical explanation of the statistical analyses
implemented by the `statistical_analysis` package and used by the report runs
under `temp/`. It describes what is calculated, the statistical unit, and how
each result should be interpreted. The analyses characterize dataset domains;
they do not use gaze-model predictions and cannot establish that a domain
difference causes model error or negative transfer.

## Analysis workflows

The package supports two related workflows:

1. **Within-ReCalib analysis** compares participant pairs separately for the
   9-point calibration and 16-point test tasks. It also measures each
   participant's longitudinal session instability and the mismatch between
   calibration and test data from the same session.
2. **Independent-dataset analysis** compares two complete dataset-summary CSVs.
   It does not pair rows, users, sessions, task names, or task schedules across
   datasets.

Both workflows use the same feature representations, session descriptors, and
four distance KPIs. Their weighting and inference units differ where noted
below.

## Population, hierarchy, and missing data

Rows whose `discarded` value is boolean true or the case-insensitive text
`true`, `1`, or `yes` are excluded. Each feature family then uses complete cases
for only the columns it needs; a missing value in one family does not exclude a
row from every other family.

The within-ReCalib hierarchy is:

```text
participant
└── session
    ├── 9-point calibration samples
    └── 16-point test samples
```

Frames from a session are correlated and are not treated as independent
replicates. In the raw view, one descriptor unit is one participant-session. In
the target-conditioned view, it is one participant-session-target cell. This
gives sessions equal weight within a participant instead of allowing longer
recordings to dominate.

The independent-dataset workflow adds another level: sessions are first given
equal weight within users, and users are then given equal weight within each
dataset. Consequently, neither frame count nor the number of sessions belonging
to a user determines that user's final dataset weight.

## Feature representations

The analyses operate on semantic feature families rather than treating every
CSV column as an unrelated variable.

| Family | Representation used in the analysis | Dimensions |
| --- | --- | ---: |
| `target_coverage` | Requested screen coordinates divided by screen width and height | 2 |
| `head_rotation` | Euler-angle head rotation in radians | 3 |
| `gaze_direction` | Normalized 3D gaze vector converted to pitch and yaw in radians | 2 |
| `gaze_origin` | Three-dimensional gaze origin in millimetres | 3 |
| `gaze_point` | Screen-plane gaze intersection in millimetres; constant `z` is excluded | 2 |
| `head_translation` | Three-dimensional head translation in millimetres | 3 |
| `landmark_position` | Centroid of 18 selected facial landmarks in pixels | 2 |
| `landmark_scale` | RMS landmark radius around the facial centroid in pixels | 1 |
| `landmark_shape` | The 18 landmarks after centering and division by RMS radius | 36 |
| `quality` | Nine QA measurements retained in their native units | 9 |

### Exact CSV variables used

`target_coverage` is the implemented family name; there is no
`task_coverage` variable. Task membership is stored separately in `task_type`.
The leading-underscore names below are temporary variables derived in memory,
not columns that must exist in the input CSV.

| Feature family | Real input CSV column names | Variables analyzed |
| --- | --- | --- |
| `target_coverage` | `pog_px_x`, `pog_px_y`; optionally `screen_width_px`, `screen_height_px` | `_pog_norm_x`, `_pog_norm_y` |
| `head_rotation` | `head_rot_x`, `head_rot_y`, `head_rot_z` | The same three columns |
| `gaze_direction` | `gaze_vector_x`, `gaze_vector_y`, `gaze_vector_z` | `_gaze_pitch`, `_gaze_yaw` |
| `gaze_origin` | `gaze_origin_x`, `gaze_origin_y`, `gaze_origin_z` | The same three columns |
| `gaze_point` | `gaze_pog_x`, `gaze_pog_y` | The same two columns |
| `head_translation` | `head_trans_x`, `head_trans_y`, `head_trans_z` | The same three columns |
| `landmark_position` | The 36 `lm_<id>_x` and `lm_<id>_y` columns listed below | `_landmark_center_x`, `_landmark_center_y` |
| `landmark_scale` | The same 36 landmark columns | `_landmark_scale` |
| `landmark_shape` | The same 36 landmark columns | `_landmark_shape_0` through `_landmark_shape_35` |
| `quality` | `qa_max_x`, `qa_min_x`, `qa_max_y`, `qa_min_y`, `qa_mean_x`, `qa_mean_y`, `qa_head_bbox_surface`, `qa_ear_left`, `qa_ear_right` | The same nine columns |

The grouping and filtering variables used by the within-ReCalib analysis are
`user_id`, `session_id`, `task_type`, and `discarded`. The independent-dataset
comparison requires `user_id`, `session_id`, and `discarded`, but deliberately
does not group or align data by `task_type`.

### Target coverage

The real target-coordinate columns are `pog_px_x` and `pog_px_y`. They are
converted to normalized coordinates as follows:

```text
_pog_norm_x = pog_px_x / effective screen width
_pog_norm_y = pog_px_y / effective screen height
```

The effective width and height come from `screen_width_px` and
`screen_height_px` when those columns exist and contain positive values. A
missing, non-positive, or unavailable dimension falls back independently to
width `2736` or height `1824`. For target-conditioned matching, the analysis
also creates `_target_x = round(_pog_norm_x, 4)` and
`_target_y = round(_pog_norm_y, 4)`.

### Head rotation

`head_rotation` directly analyzes the joint vector
`[head_rot_x, head_rot_y, head_rot_z]`. No component is analyzed separately and
no additional transformation is applied. Values are expected to be Euler
angles in radians.

### Gaze direction

The source vector is `[gaze_vector_x, gaze_vector_y, gaze_vector_z]`. Each row
is first divided by its Euclidean norm. A vector with norm at or below `1e-12`
becomes missing. For normalized components `(gx, gy, gz)`, the analyzed angles
are:

```text
_gaze_pitch = -asin(clip(gy, -1, 1))
_gaze_yaw   = atan2(-gx, -gz)
```

Thus, the analysis uses pitch and yaw in radians, not the three raw vector
components or their magnitude.

### Gaze origin

`gaze_origin` directly analyzes
`[gaze_origin_x, gaze_origin_y, gaze_origin_z]` as one three-dimensional vector.
The expected unit is millimetres.

### Gaze point

`gaze_point` directly analyzes `[gaze_pog_x, gaze_pog_y]` in millimetres.
`gaze_pog_z` is not used because the screen-plane z coordinate is constant and
therefore carries no useful distributional information.

### Head translation

`head_translation` directly analyzes
`[head_trans_x, head_trans_y, head_trans_z]` as one three-dimensional vector.
The expected unit is millimetres.

### Landmark source columns

All three landmark families use the same 18 MediaPipe landmark IDs:

```text
4, 33, 61, 129, 133, 168, 263, 308, 358,
362, 469, 470, 471, 472, 474, 475, 476, 477
```

For every ID, both `lm_<id>_x` and `lm_<id>_y` are used. The complete real
column list is:

```text
lm_4_x, lm_4_y,
lm_33_x, lm_33_y,
lm_61_x, lm_61_y,
lm_129_x, lm_129_y,
lm_133_x, lm_133_y,
lm_168_x, lm_168_y,
lm_263_x, lm_263_y,
lm_308_x, lm_308_y,
lm_358_x, lm_358_y,
lm_362_x, lm_362_y,
lm_469_x, lm_469_y,
lm_470_x, lm_470_y,
lm_471_x, lm_471_y,
lm_472_x, lm_472_y,
lm_474_x, lm_474_y,
lm_475_x, lm_475_y,
lm_476_x, lm_476_y,
lm_477_x, lm_477_y
```

### Landmark position

For each sample, `landmark_position` takes the arithmetic mean of the 18 x
coordinates and the arithmetic mean of the 18 y coordinates:

```text
_landmark_center_x = mean(lm_<id>_x for all selected IDs)
_landmark_center_y = mean(lm_<id>_y for all selected IDs)
```

These two values represent the selected landmarks' centroid in pixels.

### Landmark scale

After subtracting the centroid, the analysis calculates the root-mean-square
radial distance of the 18 landmark points:

```text
_landmark_scale = sqrt(mean((x_i - center_x)^2 + (y_i - center_y)^2))
```

This scalar is an apparent facial-size measure in pixels. It is not a bounding
box width, face area, or physical camera distance.

### Landmark shape

Every centered landmark coordinate is divided by `_landmark_scale`. The result
is flattened in the landmark order shown above, with x before y for every ID:

```text
_landmark_shape_(2j)     = (x_j - center_x) / landmark_scale
_landmark_shape_(2j + 1) = (y_j - center_y) / landmark_scale
```

For example, `_landmark_shape_0` and `_landmark_shape_1` are the normalized x
and y coordinates of landmark 4; `_landmark_shape_2` and
`_landmark_shape_3` belong to landmark 33. The final pair,
`_landmark_shape_34` and `_landmark_shape_35`, belongs to landmark 477. This
representation removes overall 2D position and scale, but does not perform a
rotation or Procrustes alignment.

### Quality variables

The nine quality values are used directly and jointly, without conversion:

- `qa_max_x`, `qa_min_x`, `qa_max_y`, and `qa_min_y`;
- `qa_mean_x` and `qa_mean_y`;
- `qa_head_bbox_surface`;
- `qa_ear_left` and `qa_ear_right`.

Their definitions and units remain those of the source CSV. Because these
measurements are heterogeneous, robust per-dimension scaling is especially
important. This family is supported by the within-ReCalib workflow but excluded
from the independent-dataset comparison.

The exact source columns and the families enabled for a particular run are
recorded in `feature_catalog.csv`. The within-ReCalib report also writes
`column_catalog.csv`, which records whether every source column was used,
missing, constant, structural, or outside the analysis contract.

For target normalization, valid per-row `screen_width_px` and
`screen_height_px` values are used when available; otherwise the implementation
falls back to `2736 × 1824`. Normalized coordinates are rounded to four decimal
places when matching targets.

## Raw and target-conditioned analyses

The **raw view** compares distributions exactly as captured. Its differences
can include unequal target coverage as well as differences in participant
behavior, geometry, or acquisition conditions.

The **target-conditioned view** calculates a distance separately at every
normalized target shared by the two groups and then takes the unweighted mean
over those targets. It asks whether the groups behave differently for the same
requested screen locations. `target_coverage` is not analyzed in this view
because conditioning on target would remove the quantity that family measures.

The within-ReCalib workflow stratifies both views by `task_type`. The
independent-dataset workflow combines all accepted samples and ignores source
task labels. It returns `no_common_targets` rather than a zero estimate when a
conditioned comparison has no shared targets.

## Robust session descriptors

For every descriptor unit and feature dimension, the implementation computes:

- location: the sample median;
- dispersion: the interquartile range (IQR);
- distribution: the concatenated 10th, 25th, 50th, 75th, and 90th percentiles;
- dependence: the upper triangle of the Spearman correlation matrix.

Spearman dependence is undefined for a scalar family, so
`landmark_scale` has no `dependence_shift` estimate.

Distances are standardized using a robust scale calculated across descriptor
units. The scale is the IQR of each descriptor dimension, with standard
deviation as a fallback and `1` as the final fallback for a constant dimension.
This makes the estimates unitless, but it does not make values from different
families or KPIs directly comparable.

## Pairwise and domain-gap distance KPIs

Every comparison produces four separate, non-negative KPIs. Zero means equality
under that particular summary; there is no universal cutoff for a small or
large value and no composite global score.

### Location shift

`location_shift` compares the mean location vectors of the units supplied to
the KPI calculation. These are session medians in the within-ReCalib workflow
and session-balanced user descriptors in the independent-dataset workflow. The
difference is divided dimension by dimension by the robust location scale, then
its Euclidean norm is divided by the square root of the number of feature
dimensions.

It answers: **are the typical feature values in different places?** A high
value can indicate a different typical gaze direction, head pose, face
position, or other family-specific center.

### Dispersion shift

`dispersion_shift` compares the median session IQR in each group. For every
dimension it calculates the absolute log-ratio implicitly through the norm:

```text
|| log((median IQR_A + 1e-9) / (median IQR_B + 1e-9)) || / sqrt(d)
```

It answers: **does one group show a different amount of within-session
variability?** Because the sign disappears in the norm, the estimate gives the
size of the difference, not which group is more variable. Inspect source
summaries when direction matters.

### Distribution shift

`distribution_shift` compares the session quantile descriptors and captures
differences beyond one center or spread statistic.

- For a scalar feature family, it is the one-dimensional Wasserstein-1 distance
  between the pooled session quantiles, divided by the robust location scale.
- For a multidimensional family, it is the square root of the energy-distance
  expression applied to robust-scaled quantile vectors:

```text
sqrt(max(0, 2 mean_distance(A, B)
            - mean_distance(A, A)
            - mean_distance(B, B)))
```

It answers: **do the fuller robust distributions differ?** For multivariate
families, this can reflect several simultaneous changes and should be
decomposed with the other KPIs.

### Dependence shift

`dependence_shift` is the Euclidean distance between the groups' mean vectors
of pairwise Spearman correlations, divided by the square root of the number of
correlation terms.

It answers: **do feature dimensions co-vary differently?** It detects changes
such as a different relationship among head-pose axes even when their marginal
locations are similar.

## Within-participant longitudinal instability

This analysis measures how much a participant changes across sessions. In the
raw view, every session's median feature vector is compared with the mean of
that participant's session medians. In the conditioned view, a participant
reference is built separately for each target; cell distances are calculated
against those references and averaged within each session.

The standardized Euclidean session distances are summarized as:

- `longitudinal_instability_median`: the participant's typical session drift;
- `longitudinal_instability_iqr`: how uneven that drift is across sessions.

These are descriptive location-based measures. The IQR is a spread measure,
not a confidence interval, and this analysis does not use the selected
pairwise KPI.

## Calibration-to-test mismatch

For every participant-session containing both task types, the analysis compares
the raw 9-point calibration descriptor with the raw 16-point test descriptor.
It calculates all four KPIs using scales estimated from the combined
calibration and test session units.

The report contains one estimate per session and a
`user_equal_session_mean` across that participant's sessions. When bootstrap
iterations are enabled and more than one session is available, the user mean
also receives a percentile confidence interval by resampling its session
estimates with replacement.

This measures representativeness of calibration data, not gaze-model
calibration accuracy. `target_coverage` and `gaze_point` mismatch values should
not be interpreted: the 9-point and 16-point grids differ by design and can
produce near-zero scaling denominators and extreme artifacts.

## Bootstrap confidence intervals

Pairwise confidence intervals use a nonparametric clustered bootstrap. For each
participant, the same number of whole sessions is sampled with replacement,
all target cells belonging to a selected session remain together, and the KPI
is recalculated. The reported interval consists of the configured lower and
upper empirical quantiles (95% by default).

For independent datasets, sessions are resampled within every user before user
descriptors and the dataset gap are rebuilt. This preserves equal user weight.
The robust scaling factors are held fixed at their values from the observed
descriptor units during resampling.

A confidence interval describes resampling uncertainty under this hierarchy;
it is not a range containing 95% of frames or users.

## Permutation tests

Within-ReCalib pairwise tests randomly reassign whole participant-session
clusters between the two participant labels while preserving the original
number of sessions assigned to the first label. For independent datasets, the
same mechanism reassigns whole users between dataset labels. The observed and
permuted KPI estimates use the same fixed robust scales.

The test is one-sided because every KPI is a non-negative distance. Its Monte
Carlo p-value is:

```text
(number of permuted distances >= observed distance + 1)
--------------------------------------------------------
              (number of valid permutations + 1)
```

The added one in numerator and denominator prevents a reported p-value of zero.
A small p-value means the observed separation is unusual under exchangeability;
it does not say the effect is large, useful, or causal.

## Multiple-testing correction

The implementation converts p-values to Benjamini-Hochberg false-discovery-rate
(FDR) q-values. The correction families are:

- within-ReCalib: all participant pairs within one combination of view, task,
  feature family, and KPI;
- independent datasets: all feature-family and KPI tests within one view.

A q-value is the FDR-adjusted counterpart of a p-value. Effect magnitude and
its interval should remain the primary evidence; `q < 0.05` is at most a
secondary flag.

## Independent-dataset domain-gap analysis

The independent workflow first builds session descriptors, then aggregates
them into one descriptor per user (or per user-target cell):

- location, distribution quantiles, and dependence vectors are averaged over
  sessions;
- dispersion vectors are combined with the median;
- each resulting user descriptor has equal weight in its dataset.

The same four KPIs then compare the two sets of user descriptors. Because users
are not paired and task labels are ignored, the estimate includes differences
in participant composition, capture conditions, protocol, behavior, and target
distribution. It is an overall dataset-domain gap, not a longitudinal or
before/after effect.

## How to read inferential results

Use the outputs in this order:

1. Check coverage, status, and whether the view is meaningful.
2. Read the KPI estimate relative to other comparisons from the same family,
   KPI, task, and view.
3. Use the confidence interval to assess estimate stability.
4. Use the q-value, then the unadjusted p-value, as secondary evidence.

Do not compare numeric magnitudes across different feature families or KPIs.
Do not interpret missing values as zero. In particular, scalar dependence,
disabled inference, insufficient data, and absent common targets intentionally
produce missing estimates or inferential fields.

## Reports currently under `temp/`

The manifests present at the time this document was written describe two
inferential runs, both using 500 bootstrap iterations, 1,000 permutations, a
95% confidence level, seed 42, raw and target-conditioned views, and all
non-quality feature families:

- `temp/statistical_analysis_inference`: task-stratified participant,
  longitudinal, and calibration/test analyses of
  `dataset_summary_recalib.csv`;
- `temp/overall_dataset_comparison_final`: independent overall comparison of
  `dataset_summary_recalib.csv` with `dataset_summary_targets.csv`.

Treat each run's `manifest.json` as authoritative if a report is regenerated
with different settings.

## Output tables

| Workflow | Table | Analysis represented |
| --- | --- | --- |
| Within-ReCalib | `pairwise_distances.csv` | Four KPIs for every participant pair, task, family, and view |
| Within-ReCalib | `user_profiles.csv` | Median and IQR longitudinal instability |
| Within-ReCalib | `calibration_test_mismatch.csv` | Per-session and equal-session user mismatch |
| Within-ReCalib | `matrices/...` | Symmetric matrix forms of estimates, intervals, p-values, and q-values |
| Independent datasets | `dataset_distances.csv` | Four overall domain-gap KPIs per family and view |
| Independent datasets | `comparison_coverage.csv` | Samples, sessions, users, descriptor units, and shared targets |

For commands and report layout, see [Hierarchical dataset statistical
analysis](statistical_analysis.md). For plot interpretation, see [How to
Interpret the Statistical Analysis Explorer](statistical_analysis_explorer_interpretation.md).
For the two-CSV workflow, see [Comparing Two Independent Dataset
CSVs](independent_dataset_comparison.md).
