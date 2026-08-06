# Statistical Analysis Script: Technical Explanation

## 1. Purpose and scope

The `statistical_analysis` package compares the statistical domains represented
by participants and sessions in `dataset_summary_recalib.csv`.

Its purpose is to answer questions such as:

- Which participants have substantially different gaze, pose, facial geometry,
  or quality distributions?
- Which participants are internally unstable across recording sessions?
- Does a session's 9-point calibration task represent its subsequent 16-point
  test tasks?
- Which feature families should later be investigated as possible predictors of
  successful adaptation or negative transfer?

The package does **not** consume model predictions in Phase 1. Its results are
descriptive dataset-domain distances, not evidence that a feature difference
caused model error or that domain adaptation is required.

## 2. Public interfaces

The package exposes two Python interfaces:

```python
analyze_frame(frame, config) -> AnalysisResult
analyze_csv(input_path, output_dir, config) -> AnalysisManifest
```

The first performs an in-memory analysis of a Pandas `DataFrame`. The second
loads a CSV and writes the complete report bundle.

The command-line interface is:

```bash
python -m statistical_analysis INPUT.csv --output OUTPUT_DIRECTORY
```

Use the following command to see all available options:

```bash
python -m statistical_analysis --help
```

## 3. Recommended execution workflow

### 3.1 Exploratory analysis

This produces all descriptive distances without statistical resampling:

```bash
python -m statistical_analysis dataset_summary_recalib.csv \
  --output temp/statistical_analysis \
  --bootstrap 0 \
  --permutations 0
```

On the development machine, the complete exploratory run took approximately 71
seconds.

### 3.2 Targeted inferential analysis

After identifying relevant feature families, confidence intervals and
permutation tests can be calculated for those families:

```bash
python -m statistical_analysis dataset_summary_recalib.csv \
  --output temp/statistical_analysis_inference \
  --families gaze_direction head_rotation head_translation landmark_shape quality \
  --bootstrap 200 \
  --permutations 500 \
  --seed 42
```

Runtime grows approximately linearly with the number of selected families and
resampling iterations. Landmark shape is the most computationally expensive
family.

### 3.3 Explore the report interactively

Generate a self-contained HTML explorer from either report directory:

```bash
python visualization/explore_statistical_analysis.py \
  --report-dir temp/statistical_analysis \
  --open
```

The explorer does not recompute statistics. It reads the exported long-form
tables and plots the selected feature family, KPI, task, view, and available
estimate or inferential statistic. Closing and reopening the generated HTML does
not require Python or a local web server.

## 4. Analysis population

The script applies the following population rules:

1. Every row where `discarded=True` is excluded.
2. Calibration (`9-point`) and test (`16-point`) tasks are analyzed separately.
3. Sessions receive equal weight within a participant.
4. Frames are not treated as independent statistical replicates.
5. Missing measurements are removed only from the feature family that requires
   them.
6. Columns that are empty or constant among accepted rows are catalogued as
   unusable instead of being imputed.

For the current dataset, this results in:

- 151,974 input samples;
- 1,949 excluded samples;
- 150,025 analyzed samples;
- 9 participants;
- 188 participant-session combinations;
- 23,916 accepted calibration samples;
- 126,109 accepted test samples.

## 5. Hierarchical statistical unit

The acquisition hierarchy is:

```text
Participant
└── Session
    ├── 9-point calibration task
    └── 16-point test tasks
        └── Samples/frames
```

Consecutive frames from one task are correlated. Pooling them as independent
observations would artificially reduce uncertainty and would give longer
sessions more influence. The script therefore summarizes each session before
constructing participant comparisons.

For the target-conditioned analysis, the elementary unit is a session-target
cell rather than an individual frame.

## 6. Raw and target-conditioned views

### Raw view

The raw view describes the dataset as captured. It includes differences caused
by both participant behavior and target coverage.

### Target-conditioned view

The target-conditioned view compares participants only at common requested
screen targets. It is the primary diagnostic view because gaze direction, eye
landmarks, and head pose naturally depend on the requested target.

Screen coordinates are divided by the effective `2736 x 1824` pixel surface and
rounded to four decimal places to construct target identifiers.

The two views answer different questions:

- Raw: "How different are the captured distributions?"
- Target-conditioned: "How differently do participants behave for equivalent
  target requests?"

Target coverage itself is analyzed only in the raw view.

## 7. Geometry-aware feature families

The input columns are reorganized into semantic families instead of being
treated as 82 unrelated variables.

| Family | Derived representation | Interpretation |
| --- | --- | --- |
| `target_coverage` | Normalized target X/Y | Requested screen-region coverage |
| `gaze_direction` | Unit vector converted to pitch/yaw | Direction of gaze independent of vector magnitude |
| `gaze_origin` | 3D position in millimetres | Estimated origin between the eyes |
| `gaze_point` | Screen-plane X/Y in millimetres | Geometrical target intersection |
| `head_rotation` | Joint 3D Euler-angle vector | Head orientation |
| `head_translation` | Joint 3D translation in millimetres | Head/camera-relative position |
| `landmark_position` | Landmark centroid | Face position in the image |
| `landmark_scale` | RMS landmark radius | Apparent face size/distance |
| `landmark_shape` | Centered and scale-normalized landmarks | Facial shape after removing position and scale |
| `quality` | Nine QA measurements | Eye visibility, face size, and geometric quality |

The original columns and their usage remain visible in `column_catalog.csv`.

### Unusable accepted-row columns

The current CSV has no usable accepted-row values for:

- `rot_cam_x`, `rot_cam_y`, and `rot_cam_z`;
- image and screen dimensions;
- gender, age, distance, and contact-lens attributes.

`gaze_pog_z` is constant and is also excluded from distance calculations.

## 8. Session descriptors

For every session or session-target cell, the script calculates:

- median location;
- interquartile range;
- 10th, 25th, 50th, 75th, and 90th percentiles;
- Spearman correlation structure for multivariate families.

These descriptors retain location, scale, distribution-shape, and dependence
information while preventing thousands of correlated frames from acting as
independent replicates.

## 9. KPI definitions

Every participant pair receives an interpretable vector of separate distances.
There is no global composite domain-gap score in Phase 1.

### 9.1 Location shift

`location_shift` measures the Euclidean difference between session-balanced
robust locations after robust per-dimension scaling. It is normalized by the
square root of the feature dimension.

A value of zero means that the compared robust centers coincide. Larger values
indicate a greater standardized location difference.

### 9.2 Dispersion shift

`dispersion_shift` measures the magnitude of the log ratio between robust
interquartile ranges:

```text
dispersion shift = || log(IQR_A / IQR_B) || / sqrt(number of dimensions)
```

It distinguishes participants with similar centers but different variability.

### 9.3 Distribution shift

`distribution_shift` measures differences beyond center and scale:

- scalar families use normalized Wasserstein-1 distance;
- multivariate families use robust-scaled energy distance.

### 9.4 Dependence shift

`dependence_shift` measures the distance between Spearman correlation
structures. It detects changes in how feature dimensions move together.

It is not applicable to scalar families such as `landmark_scale`, for which the
result is intentionally missing.

## 10. User-level longitudinal profiles

For every participant, task type, view, and feature family, the script reports:

- `longitudinal_instability_median`;
- `longitudinal_instability_iqr`.

These represent the typical session-to-participant distance and its variability.
They identify participants whose domain changes substantially across days.

## 11. Calibration-to-test mismatch

For every session, the script compares its 9-point calibration descriptors with
its 16-point test descriptors. It reports:

- one row per session;
- an equal-session participant mean;
- a participant-level bootstrap interval when bootstrapping is enabled.

This analysis measures whether the calibration task represents the subsequent
test distribution. It does not measure model calibration accuracy.

## 12. Statistical inference

When enabled:

- 95% confidence intervals resample whole sessions within participants;
- permutation tests exchange whole-session participant labels;
- `p_value` reports the session-level permutation result;
- `q_value` applies Benjamini-Hochberg false-discovery-rate correction within
  each view/task/family/KPI matrix;
- the configured random seed makes results reproducible.

Effect magnitude and its interval are primary. The `p` and `q` values are
secondary because a statistical difference is not necessarily large or useful.

When inference is disabled, the interval and significance fields remain empty.

## 13. Output bundle

| File | Purpose |
| --- | --- |
| `manifest.json` | Input path, configuration, counts, and generated-file inventory |
| `pairwise_distances.csv` | Complete long-form participant comparisons |
| `user_profiles.csv` | Longitudinal instability by participant |
| `calibration_test_mismatch.csv` | Per-session and participant-level task mismatch |
| `feature_catalog.csv` | Definitions and source columns for feature families |
| `column_catalog.csv` | Usage, missingness, cardinality, and exclusion reason for every input column |
| `matrix_index.csv` | Lookup table for all symmetric matrices |
| `matrices/...` | Participant-by-participant estimate, CI, p-value, and q-value matrices |

Each estimate matrix is symmetric and has a zero diagonal. Confidence and
significance diagonals remain empty because self-comparisons are not tested.

## 14. Interpretation rules

1. Compare numbers only within the same view, task, family, and KPI.
2. Do not compare a landmark-shape distance numerically with a gaze-direction
   distance.
3. A large exploratory distance is a hypothesis, not a statistically supported
   conclusion.
4. A statistically supported domain difference is not automatically relevant to
   model performance.
5. Negative-transfer explanations require Phase 2 to join these KPIs to paired
   baseline and adapted model results.

## 15. Known limitation in calibration/test coverage metrics

Calibration and test target grids are intentionally different. Their
session-level target centers can have an almost zero robust scaling denominator.
This produces abnormally large calibration/test mismatch values for
`target_coverage` and the deterministic `gaze_point` family.

Those calibration/test rows must not be interpreted in the current version.
Target coverage remains valid as a raw inter-participant comparison, and
target-conditioned `gaze_point` is expected to be zero.
