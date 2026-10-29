# First Conclusions from the Exploratory Statistical Analysis

## 1. Run analyzed

This document summarizes the exploratory report stored under:

```text
temp/statistical_analysis/
```

The run used:

- all 10 feature families;
- raw and target-conditioned views;
- separate 9-point and 16-point analyses;
- equal weighting of sessions within participants;
- no bootstrap iterations;
- no permutation iterations.

The analyzed population was:

| Quantity | Value |
| --- | ---: |
| Input samples | 151,974 |
| Excluded discarded samples | 1,949 |
| Accepted samples | 150,025 |
| Participants | 9 |
| Participant-session combinations | 188 |
| Accepted 9-point samples | 23,916 |
| Accepted 16-point samples | 126,109 |

Because resampling was disabled, the report supports exploratory ranking but not
claims of statistical significance. Confidence intervals, `p` values, and
FDR-adjusted `q` values are not available in this run.

## 2. Main conclusion

The dataset does not contain one universally unusual participant. Instead,
different participants are distinct along different domain dimensions.

The most important exploratory profiles are:

| Participant | Dominant pattern |
| --- | --- |
| **01** | Robust location and full-distribution differences in camera/face geometry and quality |
| **04** | Different dispersion and feature-dependence structure despite less unusual average locations |
| **07** | Gaze-direction differences in both task types and test-set geometry shifts |
| **08** | Calibration head-pose difference, test dispersion, and broad calibration-to-test mismatch |
| **02** | Test head-rotation and normalized facial-shape differences plus longitudinal instability |

This supports treating participant domain shift as multidimensional. A single
unvalidated composite score would hide the distinction between location,
dispersion, distribution shape, and dependence structure.

## 3. Frequency of exploratory extremes

For each target-conditioned task/family/KPI cell, the participant with the
greatest mean distance to all other participants was identified. Deterministic
zero-valued `gaze_point` cells were excluded from this count.

| Participant | Number of cells ranked most extreme | Main type of extreme |
| --- | ---: | --- |
| 01 | 18 | 10 location, 7 distribution, 1 dependence |
| 04 | 14 | 7 dispersion, 7 dependence |
| 07 | 8 | 5 distribution, 2 location, 1 dependence |
| 08 | 7 | 5 dispersion, 1 distribution, 1 location |
| 02 | 5 | 2 dispersion, 2 distribution, 1 location |
| 06 | 4 | 4 dependence |
| 03 | 3 | Location, dispersion, and distribution |
| 05 | 2 | Location and dependence |
| 00 | 1 | Dispersion |

This count is not a composite effect size. It only summarizes how frequently a
participant appears at the top of separate KPI rankings.

## 4. Target-conditioned full-distribution findings

The target-conditioned distribution result is the most useful initial measure
of participant difference because it controls for requested screen target.

### 4.1 Test tasks: 16-point

| Feature family | Most distant participant on average | Mean distance to peers |
| --- | ---: | ---: |
| Gaze direction | 07 | 0.930 |
| Gaze origin | 07 | 1.751 |
| Head rotation | 02 | 2.339 |
| Head translation | 07 | 1.798 |
| Landmark position | 07 | 1.536 |
| Landmark scale | 01 | 1.077 |
| Normalized landmark shape | 02 | 4.392 |
| Quality | 01 | 2.270 |

### 4.2 Calibration tasks: 9-point

| Feature family | Most distant participant on average | Mean distance to peers |
| --- | ---: | ---: |
| Gaze direction | 07 | 0.765 |
| Gaze origin | 01 | 2.028 |
| Head rotation | 08 | 2.359 |
| Head translation | 01 | 2.167 |
| Landmark position | 01 | 1.940 |
| Landmark scale | 01 | 0.925 |
| Normalized landmark shape | 03 | 4.232 |
| Quality | 01 | 2.806 |

The magnitudes in different rows must not be compared directly because each
family has its own robust scaling and dimensionality.

## 5. Coherent participant-pair differences

Several participant pairs recur across related feature families.

### Participant 01 versus participant 05

This pair has the greatest target-conditioned distribution separation for:

- 16-point gaze origin: `2.105`;
- 16-point head translation: `2.169`;
- 16-point landmark scale: `1.812`;
- 16-point quality: `2.587`;
- 9-point gaze origin: `2.459`;
- 9-point landmark scale: `1.511`.

This coherent group suggests a camera-to-face geometry and acquisition-quality
axis rather than an isolated gaze-label anomaly.

### Participant 07 versus participant 08

This is the most separated gaze-direction pair in both task types:

- 16-point: `1.169`;
- 9-point: `1.012`.

This result suggests different gaze/head compensation behavior for equivalent
screen targets.

### Participant 00 versus participant 02

In the 16-point test task, this is the strongest contrast for:

- normalized landmark-shape distribution: `5.191`;
- head-rotation distribution: `2.923`.

### Participant 03 versus participant 05

In the 9-point calibration task, this is the strongest contrast for:

- normalized landmark-shape distribution: `5.811`;
- head-rotation distribution: `3.161`.

These repeated pairings are more persuasive than isolated large matrix cells,
although they still require bootstrap uncertainty estimates.

## 6. Participant-specific interpretations

### 6.1 Participant 01: geometry and quality domain

Participant 01 is most frequently extreme in robust location and full
distribution. The pattern includes:

- gaze origin;
- head translation;
- landmark position;
- landmark scale;
- QA measurements.

The pattern is especially strong during the 9-point calibration task and remains
visible in the 16-point test task. The most plausible dataset-level hypothesis
is a distinct camera/face distance, placement, apparent scale, or image-quality
domain.

### 6.2 Participant 04: variability and dependence domain

Participant 04 is not primarily distinguished by its mean feature positions.
Instead, it repeatedly has the greatest:

- 9-point dispersion differences;
- 16-point Spearman dependence differences.

This means that ordinary comparisons of averages could miss its domain shift.
The participant may occupy similar central positions while varying differently
or coupling pose, geometry, and quality variables differently.

### 6.3 Participant 07: gaze-direction domain

Participant 07 has the greatest target-conditioned gaze-direction location and
distribution differences in both task types. It also shows strong 16-point
differences in:

- gaze origin;
- head translation;
- landmark position.

Its gaze-direction difference remains after controlling for requested target,
so target coverage alone does not explain it.

### 6.4 Participant 08: calibration representativeness risk

Participant 08 is the most unusual calibration participant for head-rotation
location and distribution. During the test task, it also has large dispersion
differences in gaze origin, head translation, landmark scale, and quality.

Most importantly, participant 08 has the greatest valid calibration-to-test
location mismatch across gaze origin, head rotation, head translation, landmark
position, landmark scale, landmark shape, and quality.

This makes participant 08 the strongest candidate for the hypothesis that its
calibration data does not adequately represent its later test behavior.

### 6.5 Participant 02: pose, shape, and session drift

Participant 02 is the strongest 16-point distribution outlier for:

- head rotation: `2.339` mean distance to peers;
- normalized landmark shape: `4.392` mean distance to peers.

It also has the greatest 16-point longitudinal instability for both families.
This combines a cross-participant domain difference with substantial variation
across the participant's own sessions.

## 7. Longitudinal instability

Notable target-conditioned session-instability results include:

| Task | Family | Most unstable participant | Median session distance |
| --- | --- | ---: | ---: |
| 16-point | Gaze direction | 07 | 0.113 |
| 16-point | Gaze origin | 04 | 0.640 |
| 16-point | Head rotation | 02 | 1.860 |
| 16-point | Head translation | 01 | 0.679 |
| 16-point | Landmark position | 04 | 0.669 |
| 16-point | Landmark scale | 05 | 0.667 |
| 16-point | Landmark shape | 02 | 1.748 |
| 16-point | Quality | 02 | 0.590 |
| 9-point | Gaze direction | 07 | 0.097 |
| 9-point | Gaze origin | 01 | 0.630 |
| 9-point | Head rotation | 08 | 0.766 |
| 9-point | Head translation | 01 | 0.673 |
| 9-point | Landmark position | 07 | 0.694 |
| 9-point | Landmark scale | 05 | 0.610 |
| 9-point | Landmark shape | 08 | 0.553 |
| 9-point | Quality | 07 | 0.651 |

Longitudinal instability is particularly relevant to calibration because a
calibration strategy learned on one session may fail to represent later sessions
even for the same participant.

## 8. Calibration-to-test mismatch

After excluding the invalid coverage-related rows described below, the most
important equal-session mismatch results are:

| Feature family | Largest location mismatch | Estimate | Largest distribution mismatch | Estimate |
| --- | ---: | ---: | ---: | ---: |
| Gaze direction | 06 | 1.485 | 06 | 2.496 |
| Gaze origin | 08 | 1.029 | 08 | 3.062 |
| Head rotation | 08 | 1.442 | 02 | 4.499 |
| Head translation | 08 | 0.990 | 08 | 3.077 |
| Landmark position | 08 | 1.187 | 08 | 2.789 |
| Landmark scale | 08 | 0.480 | 08 | 0.658 |
| Landmark shape | 08 | 1.567 | 02 | 8.277 |
| Quality | 08 | 0.944 | 08 | 3.887 |

The results expose three different mismatch profiles:

- participant 08 has a broad shift in typical calibration/test location;
- participant 02 has a particularly strong change in pose and facial-shape
  distribution;
- participant 06 has a gaze-direction-specific mismatch.

## 9. Raw versus target-conditioned results

Across informative task/family/KPI cells, participant rankings from the raw and
target-conditioned views have a median Spearman correlation of approximately
`0.867`.

This indicates that most participant differences persist after controlling for
screen target and therefore are unlikely to be caused solely by unequal target
coverage.

Some 16-point dependence and gaze-direction dispersion rankings change
substantially after conditioning. Those cases should be interpreted from the
target-conditioned view rather than the raw matrices.

## 10. Non-informative and anomalous results

### Target-conditioned gaze point

`gaze_point` distances become zero after conditioning on the requested target.
This is expected because the geometrical screen intersection is effectively
determined by the target label. These zero-valued rankings contain no participant
information and must be ignored.

### Calibration/test target coverage and gaze point

The calibration and test tasks use different target grids. Their central target
coordinates have an almost zero robust between-session scale, producing very
large standardized mismatch values—for example values near `789` for location
and `47` for distribution.

These values are a scaling artifact and must not be interpreted as evidence of
an enormous participant-domain shift. They should be excluded from
calibration-to-test conclusions until the metric implementation handles
protocol-defined target-grid differences separately.

## 11. What the exploratory run can and cannot conclude

The run supports the following statements:

- participant domains differ along several interpretable feature families;
- participants 01, 04, 07, 08, and 02 have distinct types of domain difference;
- some differences persist after conditioning on requested target;
- participant 08 has the broadest calibration-to-test representativeness risk;
- participant 02 has substantial pose/shape domain shift and longitudinal drift.

The run does not yet support:

- claims of statistical significance;
- claims that one participant definitively requires adaptation;
- claims that any KPI causes adaptation failure;
- a global ranking of domain gap across unrelated feature families;
- an explanation of negative transfer without model-result integration.

## 12. Recommended next analysis

The next step should calculate uncertainty only for the most informative
families, excluding deterministic coverage-related metrics:

```bash
python -m statistical_analysis dataset_summary_recalib.csv \
  --output temp/statistical_analysis_inference \
  --families gaze_direction gaze_origin head_rotation head_translation \
             landmark_position landmark_scale landmark_shape quality \
  --bootstrap 100 \
  --permutations 200 \
  --seed 42
```

This preliminary inferential run should take approximately 45-75 minutes on the
development machine. After reviewing interval stability, publication-level
iterations can be run only for the families and participant contrasts that
remain important.

## 13. Source result files

- [`pairwise_distances.csv`](../temp/statistical_analysis/pairwise_distances.csv)
- [`user_profiles.csv`](../temp/statistical_analysis/user_profiles.csv)
- [`calibration_test_mismatch.csv`](../temp/statistical_analysis/calibration_test_mismatch.csv)
- [`matrix_index.csv`](../temp/statistical_analysis/matrix_index.csv)
- [`manifest.json`](../temp/statistical_analysis/manifest.json)
