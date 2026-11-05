# Statistical analysis

Model-independent comparisons of the released labels. Both tools read a summary CSV written by `dataset_conversion/create_dataset_summary.py` and use the kept samples only.

## Hierarchical comparison of participants and sessions

Compares participants, sessions, calibration and test distributions and label families (head pose, gaze, targets). Sessions are weighted equally; raw and target-conditioned views are available; the output includes user-by-user matrices with optional session-bootstrap confidence intervals, session-label permutation tests and FDR correction.

```bash
python -m statistical_analysis dataset_summary_recalib.csv --output temp/statistical_analysis --bootstrap 0 --permutations 0
python visualization/explore_statistical_analysis.py --report-dir temp/statistical_analysis --open
```

The options (feature families, views, bootstrap and permutation settings) are listed by `python -m statistical_analysis --help`.

## `within_session_tasks.py`: calibration vs test tasks within a session

For every session, compares the calibration task with the three test tasks: head-rotation spread within each task, shift between the calibration and test medians (rotation and gaze-origin position), overlap of the test head poses with the calibration pose, and the differences between test tasks. Writes per-session and per-participant metrics, the statistical tests and the figure behind Table 5 of the paper.

```bash
python statistical_analysis/within_session_tasks.py --help
```
