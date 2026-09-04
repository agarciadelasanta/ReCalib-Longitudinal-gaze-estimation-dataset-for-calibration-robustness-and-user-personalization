# Comparing Two Independent Dataset CSVs

## Purpose

The independent-dataset comparison measures how large the statistical domain
gap is between two compatible dataset-summary CSV files. Rows,
samples, users, and sessions are **not paired** between files, even when their
identifiers happen to match.

The output is an interpretable KPI vector for every feature family and analysis
view. All accepted samples in each CSV are combined into its overall domain. It
deliberately does not collapse unrelated feature families into one global score.

For a worked interpretation of the current ReCalib-versus-Targets report, see
[recalib_targets_comparison_interpretation.md](recalib_targets_comparison_interpretation.md).

## Statistical unit and population

- Rows with `discarded=True` are excluded separately in both datasets.
- Sessions receive equal weight when constructing each user's descriptors.
- Users then receive equal weight when constructing each dataset's domain.
- Frame counts do not determine statistical weight.
- A session identifier is combined with its user identifier internally, so
  repeated session numbers across users cannot collide.
- Comparison-critical headers (`user_id`, `session_id`, `discarded`, and the
  selected feature sources) must exist in both files. Column order may differ.
- Discard-description text and QA-derived quality fields are outside this
  comparison and are omitted from its tables and compatibility audit.
- Different users, sessions, sample counts, task names, numbers of tasks, and
  task schedules are allowed.
- `task_type` and `task_id` are retained only as source metadata and never used
  to divide or align the datasets.

Because the datasets are independent, the measured gap includes differences in
participant composition as well as capture conditions, protocol, behavior, and
label distributions. It should be interpreted as a whole-dataset domain gap,
not a paired before/after change.

## Exploratory command

Use zero resampling iterations for a quick descriptive comparison:

```powershell
& C:\Users\irisbond\.conda\envs\mamu_data_curation\python.exe `
  -m statistical_analysis compare `
  path\to\dataset_a.csv `
  path\to\dataset_b.csv `
  --output temp\dataset_comparison `
  --name-a Dataset-A `
  --name-b Dataset-B `
  --bootstrap 0 `
  --permutations 0
```

The PowerShell backtick must be the final character on its line. The same
command can also be entered on one line.

## Inferential command

After identifying the relevant families, add session-level bootstrap confidence
intervals and whole-user permutation tests:

```powershell
& C:\Users\irisbond\.conda\envs\mamu_data_curation\python.exe `
  -m statistical_analysis compare `
  path\to\dataset_a.csv `
  path\to\dataset_b.csv `
  --output temp\dataset_comparison_inference `
  --name-a Dataset-A `
  --name-b Dataset-B `
  --families gaze_direction gaze_origin head_rotation head_translation landmark_shape `
  --bootstrap 1000 `
  --permutations 5000 `
  --seed 42
```

The defaults are 500 bootstrap and 1,000 permutation iterations. Run the
exploratory command first because a complete all-family inferential comparison
can be computationally expensive.

## Raw and target-conditioned gaps

- `raw` measures the complete observed difference between the datasets.
- `target_conditioned` compares behavior only at normalized screen targets
  present in both datasets, then averages the target-level distances equally.

Target-conditioned is the primary diagnostic for gaze, head pose, and landmarks
when both datasets contain comparable requested screen targets. If
a raw gap disappears after conditioning, different target coverage explains
much of the difference. If it remains large, the difference is more likely
associated with participant composition or acquisition setup.

Target coordinates are normalized with each row's valid `screen_width_px` and
`screen_height_px`. When those fields are unavailable, the ReCalib fallback of
`2736 x 1824` is used. Verify this fallback is appropriate for both datasets
before interpreting target-conditioned results.

`target_coverage` is raw-only. A target-conditioned comparison with no shared
targets has status `no_common_targets` and no estimate.

The independent-dataset command defaults to `--views raw` because unrelated
datasets often use different target grids. Request both views explicitly with
`--views raw target_conditioned` only when normalized targets are meaningfully
shared.

## KPI vector

| KPI | Interpretation |
| --- | --- |
| `location_shift` | Difference in robust typical feature location |
| `dispersion_shift` | Difference in robust within-session spread |
| `distribution_shift` | Difference in the fuller session-descriptor distribution |
| `dependence_shift` | Difference in Spearman relationships between dimensions |

All estimates are non-negative. Zero means equality under the selected KPI;
larger means more separation. Compare magnitudes only within the same feature
family, KPI, and view. There is no universal small/large cutoff.

## Inference

- `ci_low` and `ci_high` form the configured session-bootstrap interval.
- `p_value` comes from permuting whole users between datasets while keeping all
  of a user's session information together.
- `q_value` uses Benjamini-Hochberg FDR correction across all family/KPI tests
  within each view.

Effect size and its uncertainty are primary. The `p` and `q` values are
secondary; a small `q` value does not make a tiny gap practically important.

## Output bundle

| File | Contents |
| --- | --- |
| `manifest.json` | Input paths, configuration, population counts, weighting, and files |
| `dataset_distances.csv` | Complete domain-gap KPI vector and inferential results |
| `comparison_coverage.csv` | Users, sessions, samples, descriptor units, and common targets |
| `column_compatibility.csv` | Exact headers, roles, data types, usable counts, and compatibility status |
| `feature_catalog.csv` | Feature-family definitions and source columns |

Rows with `status=insufficient_data` or `status=no_common_targets` must not be
interpreted as zero domain gap.

## Interactive explorer

Generate and open the comparison explorer after the report has been written:

```powershell
& C:\Users\irisbond\.conda\envs\mamu_data_curation\python.exe `
  visualization\explore_dataset_comparison.py `
  --report-dir temp\dataset_comparison `
  --open
```

The explorer displays all four KPI families simultaneously for the selected
view. It also shows exact estimates, confidence intervals, `p` and `q` values,
status, comparison coverage, and the complete header compatibility audit.

## Recommended reading workflow

1. Start with target-conditioned `distribution_shift` when common-target
   coverage is sufficient; otherwise start with the raw view.
2. Identify feature families with comparatively large gaps.
3. Use location, dispersion, and dependence shifts to characterize those gaps.
4. Compare target-conditioned and raw views.
5. Inspect common-target, user, and session coverage before interpreting a
   difference.
6. Review `column_compatibility.csv`, especially any non-compatible feature
   sources.
7. Run inference for the families that remain important.

The result quantifies dataset separation. Relating that separation to baseline
or adapted angular error remains a separate model-analysis phase.
