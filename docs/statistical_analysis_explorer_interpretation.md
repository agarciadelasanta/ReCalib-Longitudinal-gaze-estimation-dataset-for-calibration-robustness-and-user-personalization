# How to Interpret the Statistical Analysis Explorer

## 1. What the explorer answers

The explorer visualizes differences in the **dataset itself**. It helps answer:

- Which participants differ from each other for a particular feature family?
- Is the difference mainly in location, variability, distribution shape, or
  dependence between variables?
- Is a participant consistently unusual relative to several peers?
- Does a participant change substantially between recording sessions?
- Does the 9-point calibration task represent the 16-point test task?

The explorer does not use model predictions. A large dataset distance may later
help explain adaptation behavior, but it does not by itself prove that domain
adaptation is needed or that it will improve angular error.

All discarded samples were removed before the report was produced. Sessions,
not frames, are the statistical units and receive equal weight within each
participant.

## 2. The most important interpretation rule

The participant matrix is a **distance matrix**, not a correlation matrix:

- `0` means that the two participant summaries are equal for the selected KPI;
- a larger value means greater statistical separation;
- values are non-negative;
- the matrix is symmetric, so user A versus B equals user B versus A;
- the zero diagonal is the comparison of each participant with themself.

There is no universal threshold separating a small and large distance. Interpret
a value relative to the other participant pairs in the **same feature family,
KPI, task, and view**. Do not directly compare, for example, a landmark-shape
distance of `4` with a gaze-direction distance of `1`.

Heatmap colors are rescaled for every selected matrix. A similarly dark cell in
two different matrices does not imply the same numerical distance; always check
the color scale and the displayed value.

## 3. Recommended filter order

Read the controls from left to right.

| Control | Meaning | Recommended use |
| --- | --- | --- |
| **Feature family** | The physical or statistical aspect being compared | Select the possible source of domain shift |
| **KPI** | The kind of difference measured within that family | Start with `distribution_shift`, then decompose it |
| **Task** | `9-point` calibration or `16-point` test data | Examine both separately |
| **View** | Raw or conditioned on shared screen targets | Use `target_conditioned` as the primary diagnostic |
| **Matrix value** | Estimate or inferential result | Start with `estimate`; use CI, `p`, and `q` secondarily |

### Raw versus target-conditioned

The views answer different questions:

- **Raw:** How different are the distributions exactly as they were captured?
- **Target-conditioned:** How differently do participants behave for equivalent
  requested screen targets?

Target-conditioned is primary for gaze, pose, landmarks, and quality because it
reduces the influence of different target composition. If a participant is
unusual in raw data but not after conditioning, target coverage probably
explains part of the apparent difference. If the difference remains after
conditioning, it is more plausibly participant- or acquisition-specific.

`target_coverage` is available only in the raw view.

### Calibration versus test task

- **9-point:** differences among participant calibration distributions;
- **16-point:** differences among participant test distributions.

A participant that is unusual only in the 9-point task may have an atypical
calibration domain. A participant that is unusual only in the 16-point task may
change after calibration or behave differently during the larger test grid.

## 4. Feature-family meanings

| Family | What a difference can represent |
| --- | --- |
| `gaze_direction` | Different pitch/yaw gaze behavior for the requested targets |
| `gaze_origin` | Different estimated eye-origin position relative to the camera |
| `head_rotation` | Different head orientation or gaze/head compensation strategy |
| `head_translation` | Different head position, camera distance, or seating geometry |
| `landmark_position` | Different face location in the image |
| `landmark_scale` | Different apparent face size, often related to camera distance |
| `landmark_shape` | Different normalized facial geometry after removing position and scale |
| `quality` | Different visibility, bounding-box, eye-aspect, or acquisition-quality behavior |
| `target_coverage` | Different requested screen-region coverage in the raw data |
| `gaze_point` | Screen-plane target intersection; target-conditioned values are deterministic |

Related families should be interpreted together. For example, simultaneous
differences in gaze origin, head translation, landmark position, landmark
scale, and quality suggest a coherent camera/face geometry domain rather than
five independent anomalies.

## 5. KPI meanings

Each KPI describes a different kind of statistical difference.

| KPI | Question answered | High value suggests |
| --- | --- | --- |
| `location_shift` | Are the typical values in different places? | Different robust centers or typical pose/geometry |
| `dispersion_shift` | Is one distribution more variable than the other? | Different spread, consistency, or range |
| `distribution_shift` | Are the complete distributions different? | A combined difference beyond only the center |
| `dependence_shift` | Do dimensions vary together differently? | Different coupling between axes or measurements |

A useful reading sequence is:

1. Use `distribution_shift` to locate broadly separated participants.
2. Check `location_shift` to determine whether typical values moved.
3. Check `dispersion_shift` to determine whether variability changed.
4. Check `dependence_shift` to identify a different multivariate relationship.

`dependence_shift` is not defined for scalar families such as
`landmark_scale`, so it can be absent.

## 6. How to read each plot

### Participant-by-participant matrix

Each off-diagonal cell compares two participants. Hover over a cell to see the
estimate and, when available, its confidence interval, `p` value, and `q` value.
Select a cell to keep the pair details visible below the matrix.

Look for:

- one dark row/column: one participant differs from many peers;
- one isolated dark cell: a specific participant pair differs strongly;
- several dark cells among a subgroup: a possible cluster or acquisition
  domain;
- uniformly light cells: little separation for the selected KPI.

The three summary values above the plots show the most extreme pair, its value,
and the median of all pairwise values for the current matrix. For `p` and `q`,
the smallest rather than largest pair is reported.

### Mean selected matrix value by participant

For every participant, this plot averages their pairwise values against all
other participants.

With `estimate` selected:

- a high bar indicates a participant that is broadly different from the group;
- a low bar indicates a participant close to most peers;
- one large pair alone may be diluted by the remaining comparisons.

Use this as a ranking aid, not as a global domain-gap score. The average remains
specific to the selected family, KPI, task, and view. Averaged `p` or `q` values
do not have a useful inferential interpretation, so use this ranking primarily
with `estimate`.

### Longitudinal session instability

This plot asks whether each participant is stable across their own sessions.
The bar is the median standardized distance between a session and that
participant's reference. The whisker length represents the IQR of those session
distances; it is a variability indicator, **not a confidence interval**.

- high bar: the participant typically changes substantially between sessions;
- long whisker: session behavior is inconsistent, with some sessions much more
  unusual than others;
- low bar and short whisker: stable participant domain across sessions.

This plot is based on session location descriptors. It follows the selected
family, task, and view, but it does not change with the selected pairwise KPI.

### Calibration-to-test mismatch by session

Every point is one session's distance between its 9-point calibration data and
16-point test data. The box summarizes the distribution of session mismatch for
each participant.

- high center: calibration is systematically unlike test for that participant;
- wide box or isolated high points: mismatch depends strongly on the session;
- low, compact distribution: calibration generally represents test well.

This plot uses the selected feature family and KPI. It compares calibration and
test directly, so the task and raw/target-conditioned controls do not redefine
this plot. Its calculation uses raw session descriptors and gives each session
equal weight.

### Pairwise values table

The table provides the exact values behind the participant matrix:

- **Estimate:** observed KPI distance;
- **95% CI:** session-bootstrap uncertainty interval;
- **p:** whole-session permutation-test probability;
- **q:** FDR-adjusted `p` value within the selected comparison matrix;
- **Sessions:** number of sessions available for both participants;
- **Samples:** accepted sample counts used to form the session summaries.

Sample counts provide context but do not act as statistical weights.

## 7. Confidence intervals, p values, and q values

Interpret inferential outputs in this order:

1. **Estimate:** Is the difference large enough to matter relative to other
   pairs in this matrix?
2. **95% CI:** Is the estimated magnitude reasonably precise, or is the interval
   very wide?
3. **q value:** Does the evidence remain after correcting for the many pairwise
   tests?
4. **p value:** Use as supporting information before FDR correction.

A small `p` or `q` value does not imply a large or practically important domain
difference. Conversely, a large exploratory estimate with a wide interval may
be important but uncertain. A conventional `q < 0.05` threshold can be used as
a secondary flag, not as the definition of domain shift.

The current `temp/statistical_analysis` report was generated with zero bootstrap
and permutation iterations. Therefore, its explorer supports descriptive
ranking only: confidence intervals, `p`, and `q` values are unavailable. The
matrix-value filter exposes them automatically when an inferential report
contains them.

## 8. A participant-investigation workflow

To investigate why a participant looks statistically unusual:

1. Select `target_conditioned`, `16-point`, and `distribution_shift`.
2. Move across feature families and note where the participant has a high
   peer-average bar or several large matrix cells.
3. For those families, inspect location, dispersion, and dependence shifts to
   characterize the difference.
4. Switch to `9-point` and determine whether the same pattern was already
   present during calibration.
5. Compare the raw and target-conditioned matrices. A disappearing difference
   points toward target composition; a persistent difference points toward the
   participant or acquisition setup.
6. Inspect longitudinal instability to see whether the participant is
   consistently unusual or changes across sessions.
7. Inspect calibration-to-test mismatch to determine whether calibration data
   represents later test behavior.
8. Repeat the analysis with bootstrap and permutation iterations before making
   a statistically supported claim.

For example, a participant with high target-conditioned head-rotation
distribution shift, high longitudinal head-rotation instability, and high
calibration/test head-rotation mismatch has three distinct signals: difference
from peers, change across their own sessions, and poor calibration/test
representativeness. This is a stronger dataset-level hypothesis than a single
large matrix cell, but it still does not establish an effect on model error.

## 9. Results that must not be interpreted

### Target-conditioned gaze point

Target-conditioned `gaze_point` distances are expected to be zero because the
screen intersection is determined by the target label. These values contain no
participant information.

### Calibration/test target coverage and gaze point

The 9-point and 16-point protocols intentionally use different target grids.
Near-zero robust scaling denominators can make calibration/test mismatch values
for `target_coverage` and `gaze_point` extremely large. These are scaling
artifacts, not evidence of enormous participant shift. The explorer displays a
warning when these families are selected.

## 10. What can be concluded

The explorer can support statements such as:

- participant A is more separated from peers than participant B for a specific
  feature family and KPI;
- a difference remains after controlling for requested target;
- a participant has substantial within-user session drift;
- calibration and test distributions are poorly matched for a feature family.

It cannot yet support statements such as:

- the statistical difference caused higher angular error;
- a particular adaptation method failed because of this KPI;
- adaptation is guaranteed to help an unusual participant;
- one feature family's numerical distance is larger than another family's in a
  directly comparable sense.

Those questions require the later model-analysis phase to join baseline and
adapted prediction results to the participant/session statistical profiles.

## 11. Rebuilding the explorer

The HTML is a snapshot of one report directory. If the statistical analysis is
rerun, regenerate the explorer so it embeds the new results:

```powershell
& C:\Users\irisbond\.conda\envs\mamu_data_curation\python.exe `
  visualization\explore_statistical_analysis.py `
  --report-dir temp\statistical_analysis `
  --open
```

For an inferential report, replace the report directory with its corresponding
output path.
