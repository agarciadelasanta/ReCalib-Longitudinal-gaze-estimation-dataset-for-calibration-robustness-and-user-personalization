# ReCalib: Longitudinal gaze estimation dataset for calibration robustness and user-specific personalization

Companion code of the **ReCalib** dataset: a longitudinal RGB gaze dataset for research on **calibration robustness** and **user-specific personalization** in camera-based gaze estimation. It contains about 150,000 images of 9 participants recorded with the front camera of a Microsoft Surface Pro 7+ in 188 sessions over several weeks. Every session starts with a 9-point calibration task followed by 16-point test tasks at other screen positions.

**Keywords:** gaze estimation, eye tracking, augmentative and alternative communication, calibration, personalization, dataset.

* **Dataset (DOI):** [https://doi.org/10.82518/A0TND2](https://doi.org/10.82518/A0TND2), controlled access under the ReCalib Data Usage Agreement (see [Data access](#data-access))
* **Contact:** Alejandro García de la Santa Ramos (agarciadelasan001@ikasle.ehu.eus, a.garcia@irisbond.com)

---

## Repository contents

All the code used to produce the released data is here, except the acquisition software (it depends on the proprietary SDK of the infrared eye tracker used to supervise the recordings). Each folder has its own README.

| Folder | Contents |
|---|---|
| [`labelling/`](labelling/README.md) | Annotation of a sample from its image, the acquisition setup and the target position: landmarks, head pose, gaze vector and quality metrics. Re-running it on the released images reproduces the released annotations. Includes the per-participant face-scale factors and the validation on EVE |
| [`dataset_conversion/`](dataset_conversion/README.md) | Conversion of the labelled recordings into the released layout, quality flags and summary files |
| [`visualization/`](visualization/README.md) | Inspection of a single sample (2D and 3D) and of the dataset distributions |
| [`statistical_analysis/`](statistical_analysis/README.md) | Comparisons between participants, sessions and calibration/test tasks |
| [`evaluation/`](evaluation/README.md) | Zero-shot evaluation and fine-tuning of ETH-XGaze under the reuse scenarios below. Uses the data, does not produce it |
| `example/` | One image with its annotation (`07_00_02_img-0039`), identical to the released sample, to try the labelling, the single-sample visualization and the ETH-XGaze model without downloading the dataset |
| `docs/` | Landmark indices, discard codes, camera calibration and other reference files |
| `dataset_summary.csv` | One row per released sample (labels and flags, no images), used by the visualization tools |

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/agarciadelasanta/ReCalib-A-multi-session-gaze-dataset-for-calibration-robustness-and-user-adaptation.git ReCalib
cd ReCalib
python -m pip install -r requirements-visualization.txt    # visualization and summary tools
python -m pip install -r requirements.txt                  # + ETH-XGaze inference, training and evaluation
python -m pip install -r labelling/requirements.txt        # labelling (MediaPipe)
```

Quick start with the bundled example:

```bash
python visualization/visualize_sample.py
```

---

## Dataset layout

```text
ReCalib/
├── setup.json                          # screen, camera calibration and conventions
├── recalib_index.csv                   # one row per sample (labels, flags)
├── sessions.csv                        # one row per task, with its date and time
├── incomplete_tasks.csv                # tasks with missing target points
├── user_00/
│   └── session_00_00/
│       ├── task_00_00_00/              # 9-point calibration task
│       │   ├── 00_00_00_img-0001.png   # frontal RGB image (1920×1080)
│       │   └── 00_00_00_img-0001.json  # annotation
│       ├── task_00_00_01/              # 16-point test tasks
│       └── ...
└── ...
```

Each annotation holds the screen target (`pog_px`), the 3D gaze (`gaze`: origin between the eyes, target on the screen, unit vector; camera frame, mm), the same gaze origin and vector from the ETH-XGaze preprocessing (`gaze_eth_xgaze`: face centre and unit vector to the same target; null when its face detector fails), the head pose (`head_pose`: rotation, translation and 18 key landmarks), quality metrics, and `discard_info` for samples flagged by the quality pipeline (null otherwise). The paper's Data Records section and [`labelling/README.md`](labelling/README.md) describe every field.

## Reuse scenarios

1. **Cross-dataset transfer:** evaluate a model trained elsewhere on ReCalib.
2. **Cross-user adaptation:** leave-one-participant-out.
3. **User personalization:** adapt on a participant's earlier sessions, evaluate on later ones.
4. **Session-level calibration:** adapt on the calibration task of a session, evaluate on its test tasks.

Keep the test participant out of training in cross-user experiments, use only the calibration task for adaptation in session-level experiments, and exclude flagged samples (`discard_info`) for the recommended subset.

---

## Data access

The dataset contains unmasked facial images, so the complete dataset is available under controlled access: requests are made through the repository and require a verified institutional identity and the signed ReCalib Data Usage Agreement. One session (`recalib_sample.tar`) is openly available under CC BY 4.0, with the separate consent of its participant.

All participants gave written informed consent, and the study was approved by the Ethics Committee for Research with Human Beings (CEISH) of the University of the Basque Country (UPV/EHU), code PI_2026_022.

## Citation

If you use the dataset or the code, please cite the dataset (the paper reference will be added on publication):

```bibtex
@misc{recalib_dataset_2026,
  author    = {Alejandro Garcia de la Santa Ramos and Ane Zulaika and Iñigo Perona and Jose Luis Jodra and Arantxa Villanueva},
  title     = {{Replication Data for: Longitudinal gaze estimation dataset for calibration robustness and user-specific personalization}},
  publisher = {DIPC data repository},
  year      = {2026},
  doi       = {10.82518/A0TND2},
  url       = {https://doi.org/10.82518/A0TND2}
}
```

## Authors

| Name | Affiliation | ORCID |
| :--- | :--- | :--- |
| **Alejandro García de la Santa Ramos** | University of the Basque Country (UPV/EHU) / Irisbond | [0000-0002-1357-6135](https://orcid.org/0000-0002-1357-6135) |
| **Ane Zulaika** | Irisbond Crowdbonding SL, Donostia, Spain | [0009-0004-3277-8033](https://orcid.org/0009-0004-3277-8033) |
| **Iñigo Perona** | University of the Basque Country (UPV/EHU) | [0000-0002-9246-3736](https://orcid.org/0000-0002-9246-3736) |
| **Jose Luis Jodra** | University of the Basque Country (UPV/EHU) | [0000-0003-2453-9521](https://orcid.org/0000-0003-2453-9521) |
| **Arantxa Villanueva** | Public University of Navarre (UPNA), Pamplona, Spain | [0000-0001-9822-2530](https://orcid.org/0000-0001-9822-2530) |

## Funding

* **Basque Government:** ADIAN Project and IT1437-22.
* **Basque Government BIKAINTEK:** grant 027-B2/2023.
* **Spanish Ministry of Science and Innovation (MCIN/AEI/10.13039/501100011033):** grant PID2021-123087OB-I00, co-funded by the European Regional Development Fund (ERDF) "A way of making Europe".

## License

* **Code:** MIT License.
* **Dataset:** ReCalib Data Usage Agreement; no Creative Commons licence applies to the restricted files.
* **Open Session (`recalib_sample.tar`):** CC BY 4.0.
* **Open sample (`example\07_00_02_img-0039.png`):** CC BY 4.0.
