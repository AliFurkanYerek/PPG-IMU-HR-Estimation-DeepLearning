from pathlib import Path
import numpy as np
import pandas as pd


WINDOW_SECONDS = 8.0
WINDOW_MS = int(WINDOW_SECONDS * 1000)
MODEL_POINTS = 320
MAX_ALLOWED_TIME_ERROR_MS = 100.0

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

ACTIVITY_ORDER = [
    "Eating",
    "High_Tempo_Cycling",
    "Low_Tempo_Cycling",
    "Sitting",
    "Sleep",
    "Walking",
]

MODEL_INPUT_COLUMNS = [
    "PPG_Raw",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
]

TARGET_COLUMN = "Chest_HR_BPM"


def discover_activity_files(activity: str, folder: Path):
    """
    Return the Excel recording files for one activity.

    Sitting and Walking use explicitly split session files when available.
    """
    files = sorted(folder.glob("*.xlsx"))

    if activity == "Sitting":
        split_files = sorted(folder.glob("Sitting_*.xlsx"))
        if split_files:
            return split_files

    if activity == "Walking":
        split_files = sorted(folder.glob("Walking_*.xlsx"))
        if split_files:
            return split_files

    return files


def load_session(path: Path) -> pd.DataFrame:
    """
    Load one raw recording without removing duplicated sensor values.
    """
    numeric_columns = [
        "Timestamp_ms",
        "PPG_Raw",
        "ACC_X",
        "ACC_Y",
        "ACC_Z",
        "Chest_HR_BPM",
    ]

    df = pd.read_excel(path)

    missing = set(numeric_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not df["Timestamp_ms"].dropna().is_monotonic_increasing:
        df = (
            df.sort_values("Timestamp_ms", kind="stable")
            .reset_index(drop=True)
        )

    return df


def nearest_row_indices(timestamps, desired_times):
    """
    For each desired timestamp, return the index of the nearest existing row.
    No signal-value interpolation is performed.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    desired_times = np.asarray(desired_times, dtype=np.float64)

    positions = np.searchsorted(
        timestamps,
        desired_times,
        side="left",
    )

    positions = np.clip(
        positions,
        0,
        len(timestamps) - 1,
    )

    previous = np.clip(
        positions - 1,
        0,
        len(timestamps) - 1,
    )

    next_dist = np.abs(
        timestamps[positions] - desired_times
    )

    prev_dist = np.abs(
        timestamps[previous] - desired_times
    )

    choose_previous = prev_dist <= next_dist

    nearest = positions.copy()
    nearest[choose_previous] = previous[choose_previous]

    return nearest


def build_exact_windows(
    df: pd.DataFrame,
    activity: str,
    session_id: str,
    window_seconds: float = WINDOW_SECONDS,
    model_points: int = MODEL_POINTS,
    max_allowed_time_error_ms: float = MAX_ALLOWED_TIME_ERROR_MS,
):
    """
    Build exact timestamp-defined windows for HR estimation.

    Important:
    - duplicated sensor values are kept
    - no interpolation
    - no filtering
    - no normalization
    - windows with timing error above the QC threshold are rejected
    """
    window_ms = int(window_seconds * 1000)

    work = (
        df.loc[df["Timestamp_ms"].notna()]
        .copy()
        .reset_index(drop=True)
    )

    timestamps = work["Timestamp_ms"].to_numpy(dtype=np.float64)

    if len(timestamps) < 2:
        return None

    session_start = float(timestamps[0])
    session_end = float(timestamps[-1])

    n_candidate_windows = int(
        np.floor(
            (session_end - session_start)
            / window_ms
        )
    )

    X_ppg = []
    X_fusion = []
    y = []
    metadata = []

    rejected_no_rows = 0
    rejected_input = 0
    rejected_target = 0
    rejected_timing = 0

    for window_id in range(n_candidate_windows):
        start_ms = session_start + window_id * window_ms
        end_ms = start_ms + window_ms

        left = np.searchsorted(
            timestamps,
            start_ms,
            side="left",
        )

        right = np.searchsorted(
            timestamps,
            end_ms,
            side="left",
        )

        if right <= left:
            rejected_no_rows += 1
            continue

        w = work.iloc[left:right].copy()

        local_ts = w["Timestamp_ms"].to_numpy(
            dtype=np.float64
        )

        if len(local_ts) < 2:
            rejected_no_rows += 1
            continue

        desired_times = np.linspace(
            start_ms,
            end_ms,
            model_points,
            endpoint=False,
            dtype=np.float64,
        )

        selected_local_idx = nearest_row_indices(
            local_ts,
            desired_times,
        )

        selected = w.iloc[selected_local_idx]

        model_values = selected[
            MODEL_INPUT_COLUMNS
        ].to_numpy(dtype=np.float32)

        if not np.isfinite(model_values).all():
            rejected_input += 1
            continue

        target_values = (
            w[TARGET_COLUMN]
            .dropna()
            .to_numpy(dtype=np.float32)
        )

        if len(target_values) == 0:
            rejected_target += 1
            continue

        target_hr = float(np.mean(target_values))

        selected_ts = selected[
            "Timestamp_ms"
        ].to_numpy(dtype=np.float64)

        timing_error_ms = np.abs(
            selected_ts - desired_times
        )

        if float(np.max(timing_error_ms)) > max_allowed_time_error_ms:
            rejected_timing += 1
            continue

        X_ppg.append(model_values[:, 0:1])
        X_fusion.append(model_values)
        y.append(target_hr)

        metadata.append({
            "Activity": activity,
            "Session_ID": session_id,
            "Window_ID": window_id,
            "Window_Start_ms": start_ms,
            "Window_End_ms": end_ms,
            "Window_Duration_sec": window_seconds,
            "Raw_Rows_in_Window": len(w),
            "Unique_Selected_Rows": len(
                np.unique(selected_local_idx)
            ),
            "Mean_Time_Error_ms": float(
                np.mean(timing_error_ms)
            ),
            "Max_Time_Error_ms": float(
                np.max(timing_error_ms)
            ),
            "Chest_Samples": len(target_values),
            "Target_HR_BPM": target_hr,
        })

    return {
        "X_ppg": np.asarray(X_ppg, dtype=np.float32),
        "X_fusion": np.asarray(X_fusion, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "metadata": pd.DataFrame(metadata),
        "candidate_windows": n_candidate_windows,
        "rejected_no_rows": rejected_no_rows,
        "rejected_input": rejected_input,
        "rejected_target": rejected_target,
        "rejected_timing": rejected_timing,
    }


def chronological_split_indices(
    n_windows: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
):
    """
    Chronological split within one recording session.
    """
    n_train = int(np.floor(n_windows * train_ratio))
    n_val = int(np.floor(n_windows * val_ratio))

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_windows)

    return train_idx, val_idx, test_idx


def fit_channel_normalizer(X_train: np.ndarray):
    """
    Compute per-channel mean/std from the training set only.
    """
    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True,
    )

    std = X_train.std(
        axis=(0, 1),
        keepdims=True,
    )

    if np.any(std == 0):
        raise ValueError(
            "At least one training channel has zero standard deviation."
        )

    return mean, std


def apply_channel_normalizer(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
):
    """
    Apply previously fitted training statistics.
    """
    return (X - mean) / std
