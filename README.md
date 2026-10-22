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

---

## Activities

The dataset contains six activity classes:| Activity | Description |
|---|---|
| Eating | Low-motion daily activity |
| High Tempo Cycling | Higher-intensity cycling |
| Low Tempo Cycling | Lower-intensity cycling |
| Sitting | Mostly stationary activity |
| Sleep | Very low-motion recording |
| Walking | Repetitive locomotion |

Some activities contain multiple recording sessions. Session boundaries are preserved during preprocessing.

---

---

## Activity-Level Accelerometer and Chest-Strap Visual Analysis

This section adds the exploratory signal figures from **`01_activity_overview.ipynb`**.

### Why these plots are included

These figures help explain **why IMU information can help heart-rate estimation** and how the six activities differ before any Deep Learning model is trained.

- The **accelerometer movement plot** is based on the **magnitude** of the 3-axis acceleration:

\[
ACC_{mag}=\sqrt{ACC_X^2 + ACC_Y^2 + ACC_Z^2}
\]

- This magnitude was used in the **exploratory analysis and visualization** to summarize wrist/body motion in one signal.
- In the **final HR models**, the IMU information is provided as the three original channels **`ACC_X`**, **`ACC_Y`**, and **`ACC_Z`**. The models do **not** use only `ACC_mag` as input.
- The chest-strap plots show the **reference HR target** used for model training and evaluation.
- The representative plots below show the same **0–120 second time range**, so activities can be compared visually.

### Cross-activity comparison plots

#### Chest-strap HR across activities

This figure compares the **mean chest-strap heart rate** of each activity together with its standard deviation.

![Chest-Strap HR Across Activities](results/figures/01_chest_hr_comparison.png)

#### Movement intensity across activities

This figure compares the **mean accelerometer magnitude** of each activity together with its standard deviation.

![Movement Intensity Across Activities](results/figures/01_movement_comparison.png)

### Activity-specific accelerometer and chest-strap plots

Each activity includes:

- **Movement intensity plot:** based on `ACC_mag = sqrt(ACC_X^2 + ACC_Y^2 + ACC_Z^2)`
- **Chest-strap HR plot:** reference HR values used as the regression target

> Note: repeated identical chest-strap HR values are not plotted as separate points in these representative figures. This makes HR changes easier to read.

### Eating

#### Movement intensity (accelerometer magnitude)
![Eating ACC magnitude](results/figures/01_acc_eating.png)

#### Chest-strap heart rate
![Eating chest-strap HR](results/figures/01_chest_eating.png)

### High Tempo Cycling

#### Movement intensity (accelerometer magnitude)
![High Tempo Cycling ACC magnitude](results/figures/01_acc_high_tempo_cycling.png)

#### Chest-strap heart rate
![High Tempo Cycling chest-strap HR](results/figures/01_chest_high_tempo_cycling.png)

### Low Tempo Cycling

#### Movement intensity (accelerometer magnitude)
![Low Tempo Cycling ACC magnitude](results/figures/01_acc_low_tempo_cycling.png)

#### Chest-strap heart rate
![Low Tempo Cycling chest-strap HR](results/figures/01_chest_low_tempo_cycling.png)

### Sitting

#### Movement intensity (accelerometer magnitude)
![Sitting ACC magnitude](results/figures/01_acc_sitting.png)

#### Chest-strap heart rate
![Sitting chest-strap HR](results/figures/01_chest_sitting.png)

### Sleep

#### Movement intensity (accelerometer magnitude)
![Sleep ACC magnitude](results/figures/01_acc_sleep.png)

#### Chest-strap heart rate
![Sleep chest-strap HR](results/figures/01_chest_sleep.png)

### Walking

#### Movement intensity (accelerometer magnitude)
![Walking ACC magnitude](results/figures/01_acc_walking.png)

#### Chest-strap heart rate
![Walking chest-strap HR](results/figures/01_chest_walking.png)

### Interpretation of these plots

These figures show that the six activities differ not only in **heart-rate level**, but also in **movement intensity**.

- **Sleep** and **Sitting** show relatively low movement.
- **Walking** and both **Cycling** classes show stronger movement.
- **High Tempo Cycling** shows high HR together with considerable motion, which helps explain why this activity is challenging for heart-rate prediction.
- Because PPG is sensitive to motion artifacts, these movement differences provide a practical reason for testing **PPG + IMU fusion models** in addition to a PPG-only baseline.

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

### Timestamp Grid and the Meaning of 320 Points

We first divide each recording session into time windows of **8 seconds (8,000 ms)**. For each window, the input representation has **320 target time positions**, spaced by:

$$\Delta t = \frac{8\text{ seconds}}{320} = 0.025\text{ seconds} = 25\text{ ms}.$$

For a window beginning at timestamp $t_0$, the target positions can be written as:

$$t_k = t_0 + 25k\text{ ms}, \qquad k=0,1,\ldots,319.$$

For each target position $t_k$, we select the **existing recorded row whose timestamp is closest** to it. The nearest row is accepted only if the timing difference is at most **100 ms**. Otherwise, the window is rejected by the preprocessing quality rule. We do **not** create new signal values by linear interpolation, and we do **not** remove repeated values. As a result, a recorded sample may be selected for more than one target position.

**Important:** The 25 ms interval is the spacing of our **model input representation**. It **does not prove** that the original sensor was physically sampled at exactly 40 Hz. Raw timestamps and actual sensor update behavior are separate from our chosen fixed-length representation.

This approach gives each CNN or CNN + BiGRU model the same time dimension. An 8-second segment is long enough to contain several heartbeats and a short period of movement, but still gives a local prediction instead of one prediction for an entire long recording. **We did not test multiple window lengths**, so the choice should not be described as experimentally optimal.

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

### Why We Used a Chronological 70/15/15 Split

The order of sensor windows matters because nearby windows in a recording can have similar heart rate, movement, and signal conditions. If all windows were randomly mixed before splitting, training and test examples could come from very close times. This could produce an overly optimistic evaluation.

Instead, for **each recording session**, we keep its chronological order and use approximately the first **70%** for training, the next **15%** for validation, and the last **15%** for testing. The final totals are **1,322 training**, **281 validation**, and **292 test** windows. The training set is used to learn weights, the validation set helps us monitor training and choose settings, and the test set is kept for final reporting.

This design reduces temporal mixing, but **it does not mean that test data comes from entirely new people or independent recording sessions**. Our results should be interpreted as performance on held-out later parts of the available sessions, not proof of subject-independent generalization.

### Normalization

Normalization parameters are computed **only from the training set**.

The same training statistics are then applied to validation and test inputs.

### Why Normalization Uses Training Data Only

PPG and the three accelerometer channels have different numerical ranges. If we directly train on these raw scales, channels with larger numbers may influence optimization differently. Standardization puts the input channels on a more comparable scale:

$$z = \frac{x - \mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}.$$

Here $x$ is an input-channel value, and $\mu_{\mathrm{train}}$ and $\sigma_{\mathrm{train}}$ are the mean and standard deviation calculated **only from the training data** for that channel. Validation and test windows use those *same* training-set values. We do **not** calculate new normalization statistics from validation or test data, because doing so would reveal information about the evaluation data during model preparation (**data leakage**).

---

<a id="models"></a>

---

## Why an 8-Second Window?

Eight seconds was selected to give the models a fixed-length segment that contains multiple PPG pulse patterns and some motion context, while remaining short enough for window-level analysis. Each window is represented by 320 positions, using the nearest existing timestamped rows and without value interpolation. **This was a design decision; the project did not compare alternative window lengths systematically, so eight seconds is not claimed to be optimal.**

---

---

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

### Why We Compared Four HR Architectures

All four models predict a **continuous heart-rate value in BPM**, using the same test split. Their main difference is **when and how the PPG and accelerometer information is combined**:

| Model | Input | How it works | Why we tested it |
|---|---|---|---|
| **PPG-only 1D CNN** | PPG only, 320 × 1 | Convolutional layers learn short local PPG patterns; a regression output predicts HR. | Establish a baseline without motion information. |
| **Early Fusion 1D CNN** | PPG + ACC X/Y/Z, 320 × 4 | All four channels enter one CNN together. | Check whether adding motion signals improves PPG-only prediction. |
| **Two-Branch CNN** | Separate PPG and three-channel ACC inputs | One CNN branch learns PPG features, another learns motion features; their outputs are combined for HR prediction. | Allow each sensor type to have its own feature extraction before fusion. |
| **CNN + BiGRU Fusion** | Separate PPG and ACC inputs | CNN layers extract local patterns; bidirectional GRU layers learn relationships across positions in the 8-second window; features are then combined. | Test whether modeling additional temporal context improves the two-branch approach. |

**CNN (Convolutional Neural Network)** detects local patterns in a signal. **GRU (Gated Recurrent Unit)** is a recurrent layer that can use information from different parts of a sequence. **BiGRU** processes a sequence in both directions within the available window; it does **not** mean that the system predicts future measurements beyond that window.

**What happened in our tests?** Overall MAE decreased from **10.080 BPM** with PPG-only CNN, to **4.956 BPM** with Early Fusion, **4.248 BPM** with Two-Branch CNN, and **3.865 BPM** with CNN + BiGRU. Thus, adding ACC information improved the overall test results, and CNN + BiGRU had the lowest overall MAE. However, the best overall model was **not** the best on every activity: for example, Two-Branch CNN had a slightly lower MAE for Sitting and Sleep. We therefore report both overall and activity-specific performance.

**Activity classification is a separate task.** In that task, we use only ACC X/Y/Z and classify one of six activities. We compare an IMU-only CNN with an IMU-only CNN + BiGRU. This classification task uses **accuracy, precision, recall, and F1**, while HR regression uses **MAE, RMSE, R², Pearson correlation, and ±5/±10 BPM tolerance rates**.

---

## Detailed Per-Activity HR Analysis

The results below compare four HR estimation models on the six recorded activities. The reference is the **BLE chest strap**. **Within ±5 BPM** and **Within ±10 BPM** describe how many predictions fall inside an error tolerance; they are **not classification accuracy**. Every percentage is calculated using the number of test windows shown for that activity.

### Four-model MAE comparison

| Activity | Test windows | PPG-only CNN | Early Fusion CNN | Two-Branch CNN | CNN + BiGRU |
|---|---:|---:|---:|---:|---:|
| Eating | 8 | 14.27 | 8.89 | 10.13 | 5.23 |
| High Tempo Cycling | 38 | 12.67 | 9.25 | 9.23 | 8.63 |
| Low Tempo Cycling | 56 | 10.60 | 5.34 | 5.09 | 4.35 |
| Sitting | 43 | 9.28 | 3.86 | 3.49 | 3.96 |
| Sleep | 73 | 13.27 | 3.71 | 1.35 | 1.46 |
| Walking | 74 | 5.22 | 3.90 | 3.71 | 3.22 |

![Four-model MAE comparison by activity](results/figures/hr_four_models_per_activity_mae.png)

**Within ±5 BPM comparison**

![HR predictions within 5 BPM](results/figures/hr_four_models_per_activity_within5.png)

**Within ±10 BPM comparison**

![HR predictions within 10 BPM](results/figures/hr_four_models_per_activity_within10.png)

### How the metrics were calculated

For ground-truth chest strap HR $y_i$, predicted HR $\hat{y}_i$, $N$ windows, and reference mean $\bar{y}$:

**Mean Absolute Error (MAE):**

$$\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|$$

**Root Mean Squared Error (RMSE):**

$$\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}$$

**Coefficient of determination (R²):**

$$R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}$$

**Percentage of predictions within a tolerance of $t$ BPM:**

$$P_{\pm t}=\frac{100}{N}\sum_{i=1}^{N}\mathbf{1}(|y_i-\hat{y}_i|\leq t)$$

We use $t=5$ and $t=10$ BPM. Pearson correlation is also reported, because it measures linear association rather than absolute prediction error.

**Normalization:** Model input channels are standardized with statistics calculated only on the training data:

$$z=\frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}$$

### Activity-by-activity comparison

#### Eating (n=8)

| Model | MAE (BPM) | Within ±5 BPM | Within ±10 BPM |
|---|---:|---:|---:|
| PPG-only CNN | 14.27 | 50.00% | 62.50% |
| Early Fusion CNN | 8.89 | 25.00% | 75.00% |
| Two-Branch CNN | 10.13 | 50.00% | 62.50% |
| CNN + BiGRU | 5.23 | 50.00% | 100.00% |

**Chest-strap HR versus predictions for each of the four models:**

**PPG-only CNN**

![Eating — Chest Strap vs PPG-only CNN](results/figures/ppg_only_actual_vs_predicted_eating.png)

**Early Fusion CNN**

![Eating — Chest Strap vs Early Fusion CNN](results/figures/early_fusion_actual_vs_predicted_eating.png)

**Two-Branch CNN**

![Eating — Chest Strap vs Two-Branch CNN](results/figures/two_branch_actual_vs_predicted_eating.png)

**CNN + BiGRU**

![Eating — Chest Strap vs CNN + BiGRU](results/figures/cnn_bigru_actual_vs_predicted_eating.png)


#### High Tempo Cycling (n=38)

| Model | MAE (BPM) | Within ±5 BPM | Within ±10 BPM |
|---|---:|---:|---:|
| PPG-only CNN | 12.67 | 28.95% | 52.63% |
| Early Fusion CNN | 9.25 | 23.68% | 63.16% |
| Two-Branch CNN | 9.23 | 34.21% | 63.16% |
| CNN + BiGRU | 8.63 | 31.58% | 73.68% |

**Chest-strap HR versus predictions for each of the four models:**

**PPG-only CNN**

![High Tempo Cycling — Chest Strap vs PPG-only CNN](results/figures/ppg_only_actual_vs_predicted_high_tempo_cycling.png)

**Early Fusion CNN**

![High Tempo Cycling — Chest Strap vs Early Fusion CNN](results/figures/early_fusion_actual_vs_predicted_high_tempo_cycling.png)

**Two-Branch CNN**

![High Tempo Cycling — Chest Strap vs Two-Branch CNN](results/figures/two_branch_actual_vs_predicted_high_tempo_cycling.png)

**CNN + BiGRU**

![High Tempo Cycling — Chest Strap vs CNN + BiGRU](results/figures/cnn_bigru_actual_vs_predicted_high_tempo_cycling.png)


#### Low Tempo Cycling (n=56)

| Model | MAE (BPM) | Within ±5 BPM | Within ±10 BPM |
|---|---:|---:|---:|
| PPG-only CNN | 10.60 | 30.36% | 44.64% |
| Early Fusion CNN | 5.34 | 46.43% | 87.50% |
| Two-Branch CNN | 5.09 | 62.50% | 85.71% |
| CNN + BiGRU | 4.35 | 58.93% | 96.43% |

**Chest-strap HR versus predictions for each of the four models:**

**PPG-only CNN**

![Low Tempo Cycling — Chest Strap vs PPG-only CNN](results/figures/ppg_only_actual_vs_predicted_low_tempo_cycling.png)

**Early Fusion CNN**

![Low Tempo Cycling — Chest Strap vs Early Fusion CNN](results/figures/early_fusion_actual_vs_predicted_low_tempo_cycling.png)

**Two-Branch CNN**

![Low Tempo Cycling — Chest Strap vs Two-Branch CNN](results/figures/two_branch_actual_vs_predicted_low_tempo_cycling.png)

**CNN + BiGRU**

![Low Tempo Cycling — Chest Strap vs CNN + BiGRU](results/figures/cnn_bigru_actual_vs_predicted_low_tempo_cycling.png)


#### Sitting (n=43)

| Model | MAE (BPM) | Within ±5 BPM | Within ±10 BPM |
|---|---:|---:|---:|
| PPG-only CNN | 9.28 | 32.56% | 60.47% |
| Early Fusion CNN | 3.86 | 76.74% | 93.02% |
| Two-Branch CNN | 3.49 | 76.74% | 95.35% |
| CNN + BiGRU | 3.96 | 81.40% | 95.35% |

**Chest-strap HR versus predictions for each of the four models:**

**PPG-only CNN**

![Sitting — Chest Strap vs PPG-only CNN](results/figures/ppg_only_actual_vs_predicted_sitting.png)

**Early Fusion CNN**

![Sitting — Chest Strap vs Early Fusion CNN](results/figures/early_fusion_actual_vs_predicted_sitting.png)

**Two-Branch CNN**

![Sitting — Chest Strap vs Two-Branch CNN](results/figures/two_branch_actual_vs_predicted_sitting.png)

**CNN + BiGRU**

![Sitting — Chest Strap vs CNN + BiGRU](results/figures/cnn_bigru_actual_vs_predicted_sitting.png)


#### Sleep (n=73)

| Model | MAE (BPM) | Within ±5 BPM | Within ±10 BPM |
|---|---:|---:|---:|
| PPG-only CNN | 13.27 | 2.74% | 24.66% |
| Early Fusion CNN | 3.71 | 75.34% | 100.00% |
| Two-Branch CNN | 1.35 | 100.00% | 100.00% |
| CNN + BiGRU | 1.46 | 100.00% | 100.00% |

**Chest-strap HR versus predictions for each of the four models:**

**PPG-only CNN**

![Sleep — Chest Strap vs PPG-only CNN](results/figures/ppg_only_actual_vs_predicted_sleep.png)

**Early Fusion CNN**

![Sleep — Chest Strap vs Early Fusion CNN](results/figures/early_fusion_actual_vs_predicted_sleep.png)

**Two-Branch CNN**

![Sleep — Chest Strap vs Two-Branch CNN](results/figures/two_branch_actual_vs_predicted_sleep.png)

**CNN + BiGRU**

![Sleep — Chest Strap vs CNN + BiGRU](results/figures/cnn_bigru_actual_vs_predicted_sleep.png)


#### Walking (n=74)

| Model | MAE (BPM) | Within ±5 BPM | Within ±10 BPM |
|---|---:|---:|---:|
| PPG-only CNN | 5.22 | 56.76% | 89.19% |
| Early Fusion CNN | 3.90 | 81.08% | 94.59% |
| Two-Branch CNN | 3.71 | 81.08% | 94.59% |
| CNN + BiGRU | 3.22 | 86.49% | 95.95% |

**Chest-strap HR versus predictions for each of the four models:**

**PPG-only CNN**

![Walking — Chest Strap vs PPG-only CNN](results/figures/ppg_only_actual_vs_predicted_walking.png)

**Early Fusion CNN**

![Walking — Chest Strap vs Early Fusion CNN](results/figures/early_fusion_actual_vs_predicted_walking.png)

**Two-Branch CNN**

![Walking — Chest Strap vs Two-Branch CNN](results/figures/two_branch_actual_vs_predicted_walking.png)

**CNN + BiGRU**

![Walking — Chest Strap vs CNN + BiGRU](results/figures/cnn_bigru_actual_vs_predicted_walking.png)

## Detailed Activity Classification Results

The selected **IMU-only 1D CNN** classified **258 of 292** test windows correctly (**88.36%** overall accuracy). The following class-level scores provide a more complete picture than overall accuracy alone.

| Activity | Test windows | Precision | Recall / classwise correct rate | F1 |
|---|---:|---:|---:|---:|
| Eating | 8 | 1.000 | 100.00% | 1.000 |
| High Tempo Cycling | 38 | 0.600 | 78.95% | 0.682 |
| Low Tempo Cycling | 56 | 0.818 | 64.29% | 0.720 |
| Sitting | 43 | 0.952 | 93.02% | 0.941 |
| Sleep | 73 | 0.973 | 100.00% | 0.986 |
| Walking | 74 | 0.973 | 95.95% | 0.966 |

Here, per-class correct rate is **recall** rather than the model's overall accuracy. For classification:

$$\mathrm{Precision}=\frac{TP}{TP+FP},\qquad \mathrm{Recall}=\frac{TP}{TP+FN}$$

$$F_1=\frac{2\cdot\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}$$

**Macro F1** is the average of the six class-level F1 scores. The CNN + BiGRU activity classifier achieved **88.01%** overall accuracy and **0.8637** Macro F1, slightly below the selected 1D CNN. Its class-by-class precision/recall/F1 comparison should be added after those outputs are checked.

---

---

## Final Result Visuals

### HR Prediction on the Test Set

![HR test predictions](results/figures/hr_test_predictions.png)

### Predicted vs True Heart Rate

![Predicted vs true HR](results/figures/hr_predicted_vs_true.png)

### Activity Classification Confusion Matrix

![Activity confusion matrix](results/figures/activity_confusion_matrix.png)

---

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
│   ├── 09_final_training_and_export.ipynb
│   └── 10_detailed_model_evaluation.ipynb
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
| `10_detailed_model_evaluation.ipynb` | Per-activity regression comparison charts |

---

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

---

## Running the Project

1. Place the raw `.xlsx` recordings in the corresponding `data/raw/` activity folders.
2. Open the repository in VS Code or Jupyter.
3. Select the project's Python virtual environment.
4. Run the notebooks in numerical order.
5. Run `09_final_training_and_export.ipynb` to generate the final models, metrics and figures.

---

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

---

## Author

**Ali Furkan Yerek**  
Computer Engineering  
Balıkesir University
