<div align="center">

# PPG–IMU Heart Rate Estimation with Deep Learning

**Wearable heart-rate estimation under motion using PPG + 3-axis accelerometer sensor fusion**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://www.tensorflow.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-1D%20CNN%20%7C%20BiGRU-purple.svg)](#models)
[![Task](https://img.shields.io/badge/Task-HR%20Regression%20%2B%20Activity%20Classification-green.svg)](#project-overview)

</div>

---

## Project Overview

Wearable **photoplethysmography (PPG)** is widely used for heart-rate monitoring, but motion can introduce artifacts that degrade signal quality.

This project investigates whether combining PPG with **3-axis accelerometer (IMU)** data improves heart-rate estimation during different activities.

The project contains two Deep Learning tasks:

1. **Heart-rate regression**
   - PPG-only baseline
   - PPG + IMU early fusion
   - Two-branch PPG + IMU fusion
   - CNN + BiGRU PPG + IMU fusion

2. **Activity classification**
   - IMU-only 1D CNN
   - IMU-only CNN + BiGRU

The **BLE chest strap** is used as the independent reference source for heart rate.

---

## Key Results

### Heart-Rate Estimation

| Model | MAE ↓ | RMSE ↓ | R² ↑ | Pearson r ↑ |
|---|---:|---:|---:|---:|
| PPG-only CNN | 10.080 BPM | 12.548 BPM | 0.719 | 0.870 |
| Early Fusion CNN | 4.956 BPM | 6.830 BPM | 0.917 | 0.958 |
| Two-Branch CNN | 4.248 BPM | 6.799 BPM | 0.917 | 0.959 |
| **CNN + BiGRU Fusion** | **3.865 BPM** | **6.243 BPM** | **0.930** | **0.966** |
| Activity-mean diagnostic | 4.316 BPM | 6.604 BPM | 0.922 | 0.969 |

**Selected final HR model:** `CNN + BiGRU PPG + IMU Fusion`

The final fusion model reduced MAE from **10.08 BPM** for the PPG-only baseline to **3.87 BPM**.

> **Important:** Strong overall performance does not imply equally strong HR tracking within every activity. Activity-specific analysis showed that cycling and walking remained more difficult, so per-activity metrics are reported alongside overall metrics.

### Activity Classification

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| **IMU 1D CNN** | **88.36%** | **88.70%** | **0.8826** | **0.8842** |
| CNN + BiGRU | 88.01% | 86.90% | 0.8637 | 0.8763 |

**Selected final activity model:** `IMU 1D CNN`

The main classification difficulty was distinguishing **High Tempo Cycling** from **Low Tempo Cycling**.

---

## Final Result Visuals

### HR Prediction on the Test Set

![HR test predictions](results/figures/hr_test_predictions.png)

### Predicted vs True Heart Rate

![Predicted vs true HR](results/figures/hr_predicted_vs_true.png)

### Activity Classification Confusion Matrix

![Activity confusion matrix](results/figures/activity_confusion_matrix.png)

---

## Sensors and Data

### Wearable Device

**Bangle.js 2 smartwatch**

Recorded channels:

- `PPG_Raw`
- `ACC_X`
- `ACC_Y`
- `ACC_Z`
- `Watch_HR_BPM`
- `Watch_Confidence_Percent`

### Reference Sensor

BLE chest strap:

- `Chest_HR_BPM`
- `Chest_RR_ms`

For HR regression, **chest-strap-derived HR** is used as the ground-truth target.

`Watch_HR_BPM` and `Watch_Confidence_Percent` are **not used as model inputs**.

---

## Activities

The dataset contains six activity classes:

| Activity | Description |
|---|---|
| Eating | Low-motion daily activity |
| High Tempo Cycling | Higher-intensity cycling |
| Low Tempo Cycling | Lower-intensity cycling |
| Sitting | Mostly stationary activity |
| Sleep | Very low-motion recording |
| Walking | Repetitive locomotion |

Some activities contain multiple recording sessions. Session boundaries are preserved during preprocessing.

---

## Preprocessing Pipeline

The raw logger can contain repeated sensor values because rows may be stored faster than the physical sensor updates.

These repeated values are **kept**.

### Windowing

Each model sample represents an exact **8-second interval** based on `Timestamp_ms`.

- Window duration: **8 seconds**
- Model points per window: **320**
- Representation: nearest existing recorded row for each desired time point
- Signal-value interpolation: **not used**
- Duplicate-value removal: **not used**
- Maximum allowed nearest-row timing error: **100 ms**

Windows exceeding the timing-error threshold are rejected.

### Dataset Split

Splitting is chronological within each recording session:

- **70% Train**
- **15% Validation**
- **15% Test**

No random window shuffling is performed before splitting.

Final usable windows:

| Split | Windows |
|---|---:|
| Train | 1322 |
| Validation | 281 |
| Test | 292 |
| **Total** | **1895** |

### Normalization

Normalization parameters are computed **only from the training set**.

The same training statistics are then applied to validation and test inputs.

---

<a id="models"></a>
## Models

### PPG-Only CNN

```text
PPG
 ↓
Conv1D
 ↓
Batch Normalization
 ↓
Pooling
 ↓
Conv1D
 ↓
Conv1D
 ↓
Global Average Pooling
 ↓
Dense
 ↓
Heart Rate
```

### Early PPG + IMU Fusion

```text
PPG + ACC_X + ACC_Y + ACC_Z
                ↓
              1D CNN
                ↓
           Heart Rate
```

### Two-Branch Fusion

```text
PPG --------> CNN branch ----\
                              ├──> Feature Fusion ──> HR
ACC_X/Y/Z --> CNN branch ----/
```

### Final CNN + BiGRU Fusion

```text
PPG --------> Conv1D ──> BiGRU ----\
                                      ├──> Feature Fusion ──> HR
ACC_X/Y/Z --> Conv1D ──> BiGRU ----/
```

### Activity Classifier

```text
ACC_X + ACC_Y + ACC_Z
          ↓
        Conv1D
          ↓
        Conv1D
          ↓
        Conv1D
          ↓
 Global Average Pooling
          ↓
        Dense
          ↓
  6-class Softmax
```

---

## Repository Structure

```text
PPG-IMU-HR-Estimation-DeepLearning/
│
├── data/
│   └── raw/
│       ├── eating/
│       ├── high_tempo_cycling/
│       ├── low_tempo_cycling/
│       ├── sitting/
│       ├── sleep/
│       └── walking/
│
├── notebooks/
│   ├── 01_activity_overview.ipynb
│   ├── 02_preprocessing_windowing.ipynb
│   ├── 03_ppg_only_model.ipynb
│   ├── 04_ppg_imu_fusion_model.ipynb
│   ├── 05_two_branch_fusion_model.ipynb
│   ├── 06_cnn_bigru_fusion_model.ipynb
│   ├── 07_activity_classification.ipynb
│   ├── 08_activity_cnn_bigru.ipynb
│   └── 09_final_training_and_export.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Notebook Workflow

| Notebook | Purpose |
|---|---|
| `01_activity_overview.ipynb` | Exploratory signal analysis |
| `02_preprocessing_windowing.ipynb` | Exact 8-second windowing, QC and splitting |
| `03_ppg_only_model.ipynb` | PPG-only regression baseline |
| `04_ppg_imu_fusion_model.ipynb` | Early PPG + IMU fusion |
| `05_two_branch_fusion_model.ipynb` | Separate PPG and IMU CNN branches |
| `06_cnn_bigru_fusion_model.ipynb` | Final CNN + BiGRU HR model |
| `07_activity_classification.ipynb` | Final IMU 1D CNN classifier |
| `08_activity_cnn_bigru.ipynb` | CNN + BiGRU classifier comparison |
| `09_final_training_and_export.ipynb` | Final model training and export |

---

## Final Exported Artifacts

After running `09_final_training_and_export.ipynb`:

```text
results/
├── models/
│   ├── hr_cnn_bigru.keras
│   ├── hr_normalization.npz
│   ├── activity_cnn.keras
│   ├── activity_normalization.npz
│   └── activity_labels.json
│
├── metrics/
│   ├── hr_metrics.csv
│   ├── hr_per_activity_metrics.csv
│   ├── activity_metrics.csv
│   ├── activity_classification_report.csv
│   └── final_summary.csv
│
└── figures/
    ├── hr_training_history.png
    ├── hr_test_predictions.png
    ├── hr_predicted_vs_true.png
    ├── activity_training_history.png
    └── activity_confusion_matrix.png
```

---

## Installation

Python **3.10** is recommended.

```bash
git clone https://github.com/AliFurkanYerek/PPG-IMU-HR-Estimation-DeepLearning.git
cd PPG-IMU-HR-Estimation-DeepLearning
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

1. Place the raw `.xlsx` recordings in the corresponding `data/raw/` activity folders.
2. Open the repository in VS Code or Jupyter.
3. Select the project's Python virtual environment.
4. Run the notebooks in numerical order.
5. Run `09_final_training_and_export.ipynb` to generate the final models, metrics and figures.

---

## Evaluation Strategy

### Heart-Rate Regression

Metrics:

- MAE
- RMSE
- R²
- Pearson correlation
- Per-activity metrics

An **activity-mean diagnostic baseline** is also evaluated to detect whether strong overall scores are driven primarily by activity-dependent HR levels.

### Activity Classification

Metrics:

- Accuracy
- Balanced Accuracy
- Macro F1
- Weighted F1
- Per-class precision / recall / F1
- Confusion matrix

---

## Limitations

- The dataset contains a limited number of independent recording sessions.
- Class sizes are imbalanced.
- Eating contains relatively few test windows.
- High- and low-tempo cycling produce similar IMU patterns.
- Strong overall HR-regression performance does not always correspond to strong within-activity HR tracking.
- The chronological split evaluates later portions of the available recordings and should not be interpreted as subject-independent external validation.
- Additional subjects and independent sessions would be required for stronger generalization claims.

---

## Future Work

Possible extensions include:

- collecting data from additional participants
- increasing the number of independent sessions per activity
- subject-independent evaluation
- improved motion-artifact suppression
- signal-quality estimation
- attention-based sensor fusion
- external validation on another wearable dataset
- comparison with timestamp-resampled recordings

---

## Project Context

Developed during an **Erasmus traineeship at Hanze University of Applied Sciences, Groningen, the Netherlands**.

The project focuses on:

- wearable sensing
- PPG signal analysis
- motion artifacts
- IMU-based motion information
- multimodal sensor fusion
- Deep Learning for heart-rate estimation

---

## Author

**Ali Furkan Yerek**  
Computer Engineering  
Balıkesir University
