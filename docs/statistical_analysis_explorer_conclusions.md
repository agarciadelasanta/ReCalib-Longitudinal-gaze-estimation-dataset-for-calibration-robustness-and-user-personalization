# ReCalib Statistical Analysis Explorer: final conclusions

#recalib #statistical-analysis #user-variability #calibration #longitudinal-analysis #gaze-estimation

## Executive conclusions

The ReCalib dataset contains substantial and statistically supported
participant variability. This variability is multidimensional: different users
are unusual in typical geometry, within-session spread, full distribution, or
dependence structure. There is no single universally anomalous participant and
no defensible one-number ranking of user domain shift.

The main participant profiles are:

- **User 01:** the clearest location and full-distribution outlier for
  camera-to-face geometry, especially during calibration;
- **User 04:** primarily a variability and feature-dependence outlier rather
  than a location outlier;
- **User 07:** the strongest gaze-direction outlier in both tasks, with
  additional test-domain geometry differences;
- **User 08:** the broadest calibration-to-test representativeness problem;
- **User 02:** the strongest test head-rotation and normalized-landmark-shape
  outlier, together with high longitudinal drift in those families;
- **User 06:** a calibration-to-test mismatch concentrated in gaze direction.

Calibration is not equally representative of test behavior for every user.
User 08 has the largest mismatch in 14 of the 27 valid feature/KPI summaries,
while users 02, 04, and 06 show more specialized mismatch patterns. A single
static calibration strategy is therefore unlikely to be equally robust across
all participants and sessions.

**Practical verdict:** the results justify evaluating user- and session-aware
adaptation or recalibration in ReCalib. They do not prove that adaptation will
reduce gaze error, because this statistical report contains no model
predictions. That claim requires a paired baseline-versus-adapted evaluation.

## Final run analyzed

These conclusions use the inferential report stored in:

```text
temp/statistical_analysis_inference/
```

The run analyzed:

| Quantity | Value |
| --- | ---: |
| Input samples | 151,974 |
| Excluded discarded samples | 1,949 |
| Accepted samples | 150,025 |
| Participants | 9 |
| Participant-session combinations | 188 |
| Accepted 9-point calibration samples | 23,916 |
| Accepted 16-point test samples | 126,109 |

It used nine feature families, raw and target-conditioned views, 500
session-bootstrap iterations, 1,000 whole-session permutation iterations, 95%
confidence intervals, and random seed 42. Unlike the earlier exploratory run,
this report contains confidence intervals, p-values, and FDR-adjusted q-values.
The `quality` family was not included in this final run and no quality
conclusion should be inferred from it.

## 1. Between-user variability is widespread

The target-conditioned view is the most informative test of participant
variability because it compares users at equivalent requested screen targets.
After excluding deterministic `gaze_point` results and protocol-only
`target_coverage`, the final run contains 972 valid pairwise tests in each
task/view combination.

| View | Task | FDR-significant pairs | Percentage |
| --- | --- | ---: | ---: |
| Raw | 9-point | 667 / 972 | 68.6% |
| Target-conditioned | 9-point | 640 / 972 | 65.8% |
| Raw | 16-point | 764 / 972 | 78.6% |
| Target-conditioned | 16-point | 766 / 972 | 78.8% |

The prevalence of significant target-conditioned differences shows that user
heterogeneity is not merely the result of unequal target coverage. The larger
fraction in the 16-point task also suggests that participant domains separate
more strongly during test than during the shorter calibration task.

The evidence is especially consistent for complete distributions:

- 16-point `distribution_shift` is significant for all 36 participant pairs
  in head rotation and landmark shape, 35/36 pairs in gaze direction, gaze
  origin, and head translation, 33/36 in landmark position, and 23/36 in
  landmark scale;
- 9-point `distribution_shift` is significant for all 36 pairs in head
  rotation and landmark shape, 34/36 in gaze origin and head translation,
  33/36 in gaze direction, 27/36 in landmark position, and 16/36 in landmark
  scale.

Dependence differences are more selective. Normalized landmark shape is the
exception: its dependence shift is significant for 35/36 test pairs and all
36 calibration pairs. For most other families, location, dispersion, and
distribution provide stronger evidence than correlation structure.

## 2. Which users differ most from their peers

For each target-conditioned task/family/KPI matrix, the explorer can rank users
by their mean distance to all eight peers. Excluding deterministic gaze point,
the number of matrices in which each user ranks first is:

| User | First-place matrices | Dominant type |
| --- | ---: | --- |
| 01 | 14 | 8 location, 5 distribution, 1 dependence |
| 04 | 12 | 6 dispersion, 6 dependence |
| 07 | 8 | 5 distribution, 2 location, 1 dependence |
| 08 | 6 | 4 dispersion, 1 location, 1 distribution |
| 02 | 5 | 2 dispersion, 2 distribution, 1 location |
| 03 | 3 | 1 location, 1 dispersion, 1 distribution |
| 06 | 3 | 3 dependence |
| 05 | 2 | 1 location, 1 dependence |
| 00 | 1 | 1 dispersion |

This count is a profile summary, not a composite effect size or hypothesis
test. It shows why one global outlier score would be misleading: user 01 is
mainly shifted in center and distribution, while user 04 is unusual in spread
and dependence.

### Target-conditioned distribution outliers

The strongest mean distance-to-peer profile for each family is:

| Task | Family | User | Mean distance to peers | Significant incident pairs |
| --- | --- | ---: | ---: | ---: |
| 16-point | Gaze direction | 07 | 0.930 | 8 / 8 |
| 16-point | Gaze origin | 07 | 1.750 | 8 / 8 |
| 16-point | Head rotation | 02 | 2.339 | 8 / 8 |
| 16-point | Head translation | 07 | 1.798 | 8 / 8 |
| 16-point | Landmark position | 07 | 1.536 | 8 / 8 |
| 16-point | Landmark scale | 01 | 1.077 | 6 / 8 |
| 16-point | Landmark shape | 02 | 4.392 | 8 / 8 |
| 9-point | Gaze direction | 07 | 0.765 | 8 / 8 |
| 9-point | Gaze origin | 01 | 2.028 | 8 / 8 |
| 9-point | Head rotation | 08 | 2.359 | 8 / 8 |
| 9-point | Head translation | 01 | 2.167 | 8 / 8 |
| 9-point | Landmark position | 01 | 1.940 | 8 / 8 |
| 9-point | Landmark scale | 01 | 0.925 | 7 / 8 |
| 9-point | Landmark shape | 03 | 4.232 | 8 / 8 |

Distances in different feature-family rows are not numerically comparable.
Within a row, however, the ranking identifies the user most broadly separated
from peers. The final column shows that nearly all of these outlier profiles are
supported by significant pairwise comparisons after correction.

## 3. Coherent participant-pair differences

Repeated pairings across related families provide stronger domain evidence than
one isolated heatmap cell.

### Users 01 and 05: camera-to-face geometry

This pair has the largest target-conditioned distribution separation for gaze
origin and landmark scale in both tasks, and for head translation in the test
task:

| Task | Family | Estimate | 95% CI | q-value |
| --- | --- | ---: | --- | ---: |
| 16-point | Gaze origin | 2.105 | [1.892, 2.472] | 0.00116 |
| 16-point | Head translation | 2.169 | [1.923, 2.512] | 0.00116 |
| 16-point | Landmark scale | 1.812 | [1.388, 2.202] | 0.00240 |
| 9-point | Gaze origin | 2.459 | [2.197, 2.826] | 0.00128 |
| 9-point | Landmark scale | 1.511 | [1.136, 1.897] | 0.00327 |

The agreement among eye-origin, head-position, and apparent face-scale
variables indicates a coherent camera/face geometry axis.

### Users 07 and 08: gaze direction

This is the most separated gaze-direction pair in both tasks:

- 16-point: estimate `1.169`, CI `[1.035, 1.324]`, `q = 0.00120`;
- 9-point: estimate `1.012`, CI `[0.882, 1.175]`, `q = 0.00138`.

Because these are target-conditioned results, requested target composition does
not explain the difference. The pair likely represents different gaze behavior
or gaze/head compensation strategies for the same requested targets.

### Users 00 and 02: test pose and facial shape

In the 16-point test task, this pair has the largest distribution separation
for head rotation (`2.923`, CI `[2.808, 3.214]`) and normalized landmark shape
(`5.191`, CI `[4.915, 5.715]`). Both have `q = 0.000999`.

### Users 03 and 05: calibration pose and facial shape

In the 9-point calibration task, this pair has the largest distribution
separation for head rotation (`3.161`, CI `[3.031, 3.326]`) and landmark shape
(`5.811`, CI `[5.593, 6.139]`). Both have `q = 0.000999`.

## 4. Within-user longitudinal variability

Longitudinal instability measures how far each session moves from that user's
own reference. The most unstable user in each target-conditioned family is:

| Task | Family | User | Median session distance |
| --- | --- | ---: | ---: |
| 16-point | Gaze direction | 07 | 0.113 |
| 16-point | Gaze origin | 04 | 0.640 |
| 16-point | Head rotation | 02 | 1.860 |
| 16-point | Head translation | 01 | 0.679 |
| 16-point | Landmark position | 04 | 0.669 |
| 16-point | Landmark scale | 05 | 0.667 |
| 16-point | Landmark shape | 02 | 1.748 |
| 9-point | Gaze direction | 07 | 0.097 |
| 9-point | Gaze origin | 01 | 0.630 |
| 9-point | Head rotation | 08 | 0.766 |
| 9-point | Head translation | 01 | 0.673 |
| 9-point | Landmark position | 07 | 0.694 |
| 9-point | Landmark scale | 05 | 0.610 |
| 9-point | Landmark shape | 08 | 0.553 |

The main conclusions are:

- user 02 combines strong between-user test differences with the largest
  within-user drift in test head rotation and landmark shape;
- user 07 is consistently the most longitudinally variable in gaze direction;
- user 01 shows calibration/test geometry differences as well as substantial
  session drift in gaze origin and head translation;
- user 08 is particularly unstable during calibration for head rotation and
  landmark shape;
- face scale varies most across sessions for user 05 in both tasks.

The instability IQR adds another warning: user 00 has the widest test-session
spread in gaze origin, head translation, and landmark scale, while user 08 has
the widest calibration-session spread in gaze origin, head rotation, head
translation, landmark position, and landmark shape. Thus, a user can have a
moderate typical drift but still contain especially unstable sessions.

These longitudinal values are descriptive. Their IQRs are variability
intervals, not confidence intervals, and `user_profiles.csv` does not contain
permutation p-values. The rankings identify calibration-risk candidates but do
not formally test one user's instability against another's.

## 5. Calibration-versus-test mismatch

This analysis compares each session's raw 9-point calibration descriptors with
its 16-point test descriptors, then reports an equal-session mean for every
user. Target coverage and gaze point are excluded from the conclusions because
their different protocol-defined grids create non-interpretable scaling
artifacts.

The largest location and full-distribution mismatch for each valid family is:

| Family | Largest location mismatch | Estimate (95% CI) | Largest distribution mismatch | Estimate (95% CI) |
| --- | ---: | --- | ---: | --- |
| Gaze direction | 06 | 1.485 [1.281, 1.687] | 06 | 2.496 [2.358, 2.633] |
| Gaze origin | 08 | 1.028 [0.721, 1.326] | 08 | 3.061 [2.758, 3.385] |
| Head rotation | 08 | 1.442 [0.899, 2.071] | 02 | 4.498 [4.252, 4.767] |
| Head translation | 08 | 0.990 [0.746, 1.258] | 08 | 3.077 [2.817, 3.394] |
| Landmark position | 08 | 1.186 [0.828, 1.586] | 08 | 2.789 [2.559, 3.049] |
| Landmark scale | 08 | 0.480 [0.292, 0.670] | 08 | 0.658 [0.460, 0.846] |
| Landmark shape | 08 | 1.567 [0.917, 2.354] | 02 | 8.277 [7.837, 8.744] |

Three distinct mismatch profiles emerge:

- **User 08 has the broadest mismatch.** It is the largest result in 14 of 27
  valid family/KPI cells, spanning gaze origin, head pose/position, face
  position and scale, and landmark shape. Its calibration samples are the
  strongest candidate for being poorly representative of later test behavior.
- **User 04 is a dispersion/dependence mismatch case.** It leads 7 of 27 cells,
  including head-rotation and landmark dispersion and several dependence
  metrics. Center-only calibration diagnostics would miss this behavior.
- **Users 02 and 06 have specialized mismatches.** User 02 leads the
  head-rotation and landmark-shape distribution shifts; user 06 leads gaze
  direction location, dispersion, and distribution.

The reported confidence intervals resample sessions within a user. They show
the stability of that user's mean mismatch across recorded sessions, but there
is no calibration/test permutation test or q-value in this table. The analysis
therefore supports robust descriptive ranking, not formal claims that one
user's mismatch is significantly greater than another's.

## 6. Raw versus target-conditioned conclusions

Across informative task/family/KPI matrices, the median Spearman correlation
between raw and target-conditioned pair rankings is `0.904`. Location and full-
distribution rankings are usually very stable, particularly for gaze origin,
head translation, landmark scale, and landmark shape.

This reinforces the conclusion that much of the user variability persists after
controlling for requested target. It is therefore not solely a target-coverage
artifact.

Conditioning matters more for several dispersion and dependence matrices,
especially in the 9-point task. Examples include landmark-shape dependence,
gaze-direction dispersion, and landmark-position dependence. These low or
negative raw/conditioned rank correlations mean that target composition can
change which participant pairs appear most different for those KPIs. Use the
target-conditioned matrix for user-domain interpretation in these cases.

## 7. Results that must not be interpreted

### Target-conditioned gaze point

Every target-conditioned `gaze_point` estimate is zero apart from floating-
point noise, and all q-values equal 1. The screen intersection is effectively
determined by the conditioning target, so these matrices contain no participant
information.

### Calibration/test target coverage and gaze point

The 9-point and 16-point protocols intentionally use different target grids.
Near-zero robust scale denominators create extremely large standardized
calibration/test mismatch values for `target_coverage` and `gaze_point`. These
are protocol/scaling artifacts and must be excluded from participant
conclusions.

### Cross-family magnitude comparisons

A landmark-shape distance of 4 is not necessarily more important than a gaze-
direction distance of 1. Each family has its own dimension, descriptor scale,
and interpretation. Compare participants within the same family, task, view,
and KPI.

## 8. Implications for calibration and adaptation

The results support three design decisions:

1. **Personalization should be evaluated.** Pairwise participant differences
   are widespread even after target conditioning, so one population model may
   not represent every user's domain equally well.
2. **Calibration should be session-aware.** Longitudinal drift and the broad
   user 08 mismatch show that one calibration may not remain representative
   across behavior or sessions.
3. **Adaptation diagnostics should be multidimensional.** User 01 needs a
   geometry-focused interpretation, user 04 a spread/dependence interpretation,
   user 07 a gaze-direction interpretation, and user 02 a pose/shape and drift
   interpretation. A single domain score would obscure these mechanisms.

The most defensible next experiment is to join these profiles with held-out
baseline and adapted angular errors. Test whether higher participant distance,
longitudinal instability, or calibration/test mismatch predicts adaptation
benefit or negative transfer. Use participant/session-aware resampling and keep
adaptation samples separate from evaluation samples.

## 9. Final verdict

ReCalib exhibits enough inter-user heterogeneity, longitudinal session drift,
and calibration-to-test mismatch to justify a user/session adaptation study.
Users 08, 02, 07, 01, 04, and 06 provide distinct high-value stress cases.

The statistical explorer does **not** establish that adaptation is required for
acceptable gaze accuracy or that it will help every user. It establishes that
the underlying domains differ and that calibration representativeness is user-
dependent. Adaptation should therefore be treated as a well-supported
experimental requirement, with its actual benefit decided by model-error
results.

## Source files

The conclusions are derived from:

- `temp/statistical_analysis_inference/manifest.json`;
- `temp/statistical_analysis_inference/pairwise_distances.csv`;
- `temp/statistical_analysis_inference/user_profiles.csv`;
- `temp/statistical_analysis_inference/calibration_test_mismatch.csv`;
- `temp/statistical_analysis_inference/feature_catalog.csv`.

For formulas and exact source variables, see [Statistical analyses and
methods](statistical_analysis_methods.md). For guidance on reading the plots,
see [How to Interpret the Statistical Analysis
Explorer](statistical_analysis_explorer_interpretation.md).
