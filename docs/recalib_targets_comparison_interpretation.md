# Understanding the ReCalib vs Targets Comparison

## 1. Scope of this report

This document explains the exploratory comparison stored in:

```text
temp/overall_dataset_comparison/
```

The report compares `dataset_summary_recalib.csv` and
`dataset_summary_targets.csv` as two **independent overall domains**.

- All accepted samples are included together.
- Source task names and task schedules are ignored.
- Samples, sessions, and users are not paired between datasets.
- Sessions receive equal weight within each user.
- Users receive equal weight within each dataset.
- Rows marked `discarded=True` are excluded.
- Discard-description text and QA-derived quality fields are outside the
  comparison and do not appear in its analytical tables or compatibility audit.

This weighting means that ReCalib's larger sample count does not automatically
give it more influence than Targets.

## 2. Population compared

| Dataset | Accepted samples | Excluded discarded samples | Users | User-session recordings |
| --- | ---: | ---: | ---: | ---: |
| ReCalib | 150,025 | 1,949 | 9 | 188 |
| Targets | 64,763 | 0 | 84 | 206 |

The participant populations differ substantially in size. The result therefore
describes a whole-dataset domain gap, including participant composition and
acquisition differences. It is not a paired before/after comparison.

## 3. What the four KPIs mean

Every feature family has its own four-part distance vector.

| KPI | Question answered |
| --- | --- |
| `location_shift` | Are the robust typical values located differently? |
| `dispersion_shift` | Does the within-session spread differ? |
| `distribution_shift` | Do the fuller user/session descriptor distributions differ? |
| `dependence_shift` | Do the feature dimensions vary together differently? |

All reported distances are non-negative:

- zero means equality under that KPI;
- larger values mean greater separation;
- the values are standardized effect distances, not probabilities;
- there is no universal threshold for a small or large domain gap.

Do not compare two numbers merely because they are shown on the same chart.
Interpret each value in the context of its feature family, representation, and
KPI. Repeated evidence across related families is more informative than one
isolated large number.

## 4. Exploratory KPI vector

The current run used `--bootstrap 0 --permutations 0`, so it contains effect
estimates only. Confidence intervals, `p` values, and `q` values are not yet
available.

| Feature family | Location | Dispersion | Distribution | Dependence |
| --- | ---: | ---: | ---: | ---: |
| Target coverage | 2.3423 | 0.6931 | 7.1123 | 0.0798 |
| Gaze direction | 0.1840 | 0.1912 | 1.4108 | 0.0570 |
| Gaze origin | 0.8123 | 0.9583 | 1.7111 | 0.1163 |
| Gaze point | 0.1529 | 0.0000 | 1.9088 | 0.0798 |
| Head rotation | 0.4237 | 1.2745 | 1.2653 | 0.0352 |
| Head translation | 0.8320 | 1.1696 | 1.7554 | 0.1312 |
| Landmark position | 0.8605 | 2.3279 | 1.9371 | 0.0182 |
| Landmark scale | 0.8999 | 1.9778 | 0.8943 | N/A |
| Landmark shape | 0.5565 | 0.4870 | 2.1308 | 0.2789 |

`landmark_scale` is scalar, so a dependence structure does not exist. Its
missing dependence value is expected and is not a failed calculation.

## 5. Main interpretation

### 5.1 The target protocols are substantially different

Target coverage has a location shift of `2.3423` and a distribution shift of
`7.1123`. This is direct evidence that the two datasets request or cover
different screen regions and target distributions.

This is primarily a **protocol-domain difference**. It should not be interpreted
as participant behavior or camera appearance. The raw gaze-point distribution
gap (`1.9088`) is also likely influenced by this target-layout difference.

### 5.2 Camera-to-face geometry differs coherently

Several related families show the same broad pattern:

| Family | Location | Dispersion | Distribution |
| --- | ---: | ---: | ---: |
| Gaze origin | 0.8123 | 0.9583 | 1.7111 |
| Head translation | 0.8320 | 1.1696 | 1.7554 |
| Landmark position | 0.8605 | 2.3279 | 1.9371 |
| Landmark scale | 0.8999 | 1.9778 | 0.8943 |

The repeated location and dispersion differences suggest a coherent
camera/participant geometry gap: face position, apparent face size, head
position, or camera-relative eye origin differ between the datasets.

The especially large landmark-position and landmark-scale dispersion values
mean the datasets differ strongly in how variable face framing and apparent
scale are, not only in their typical values.

### 5.3 Head-pose variability differs more than its typical center

Head rotation has:

- location shift: `0.4237`;
- dispersion shift: `1.2745`;
- distribution shift: `1.2653`;
- dependence shift: `0.0352`.

The dispersion gap is much more prominent than the dependence gap. A reasonable
dataset-level hypothesis is that head-pose range or consistency differs, while
the relationship between rotation axes remains comparatively similar.

### 5.4 Gaze-direction centers are relatively close, but the full domains differ

Gaze direction has modest location (`0.1840`) and dispersion (`0.1912`) shifts,
but its full distribution shift is `1.4108`.

This means similar typical gaze direction and spread do not imply identical
domains. Differences can remain in user-level distribution structure or tails.
The result should be interpreted together with the large target-coverage gap,
because raw gaze distributions depend on which targets were requested.

### 5.5 Normalized facial shape is another distinct domain axis

Landmark shape removes face position and global scale before comparison. Its
distribution shift is `2.1308`, and its dependence shift is `0.2789`.

Because position and scale have already been removed, this pattern cannot be
explained only by where the face appears or how large it is. It indicates a
remaining difference in normalized landmark geometry or in how landmark
components vary together across the dataset populations.

## 6. How to use the HTML explorer

Open:

```text
temp/overall_dataset_comparison/dataset_comparison_explorer.html
```

The explorer contains four plots, one for each KPI. Every bar represents one
feature family's ReCalib-versus-Targets distance.

### Displayed value

The current exploratory report exposes only `estimate`. An inferential rerun
will also expose:

- `ci_low` and `ci_high`: session-bootstrap confidence limits;
- `p_value`: whole-user permutation-test result;
- `q_value`: false-discovery-rate-adjusted result.

Effect magnitude and its confidence interval are primary. Statistical
significance is secondary.

### Comparison coverage

Use the coverage table to verify how many users, sessions, samples, and
descriptor units contributed to each feature. A value marked as unavailable
must not be treated as a zero gap.

### Column compatibility

The compatibility table reports only columns relevant to the comparison or
neutral source metadata. A selected feature is interpretable only when its
required columns have usable values in both datasets.

## 7. What this exploratory run supports

The current results support these dataset-level statements:

- ReCalib and Targets have substantially different target protocols.
- Their camera-to-face geometry and face-framing variability differ coherently.
- Head-pose variability differs more than head-rotation dependence structure.
- A gaze-direction gap remains beyond a simple difference in average direction.
- Normalized facial-landmark shape provides an additional domain distinction.

These are descriptive findings. Without resampling, the report cannot yet claim
that an observed gap is statistically stable across alternative session
samples.

## 8. What the comparison cannot establish

This report does not establish:

- that either dataset is better;
- that any gap causes gaze-estimation error;
- that domain adaptation is necessary or sufficient;
- that an adaptation method will improve every user;
- that target-protocol differences are equivalent to biometric domain shift.

Connecting these distances to baseline and adapted angular error requires the
later model-analysis phase.

## 9. Recommended inferential follow-up

Target coverage and gaze point primarily reflect protocol geometry. For a first
inferential run focused on appearance, pose, and gaze behavior, use:

```powershell
& C:\Users\irisbond\.conda\envs\mamu_data_curation\python.exe `
  -m statistical_analysis compare `
  dataset_summary_recalib.csv `
  dataset_summary_targets.csv `
  --output temp\overall_dataset_comparison_inference `
  --name-a ReCalib `
  --name-b Targets `
  --families gaze_direction gaze_origin head_rotation head_translation landmark_position landmark_scale landmark_shape `
  --bootstrap 200 `
  --permutations 500 `
  --seed 42
```

Regenerate the explorer from that output directory to inspect confidence
intervals and adjusted significance values.
