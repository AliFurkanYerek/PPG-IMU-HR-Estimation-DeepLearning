import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def regression_metrics(
    y_true,
    y_pred,
):
    """
    Standard HR-regression metrics.
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    if (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        correlation = np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    else:
        correlation = np.nan

    return {
        "MAE_BPM": mae,
        "RMSE_BPM": rmse,
        "R2": r2,
        "Correlation": correlation,
        "True_Mean": np.mean(y_true),
        "True_Std": np.std(y_true),
        "Pred_Mean": np.mean(y_pred),
        "Pred_Std": np.std(y_pred),
    }


def regression_metrics_by_activity(
    y_true,
    y_pred,
    activities,
    activity_order,
):
    """
    Compute HR-regression metrics separately for each activity.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    activities = np.asarray(activities)

    rows = []

    for activity in activity_order:
        mask = activities == activity

        if not np.any(mask):
            continue

        metrics = regression_metrics(
            y_true[mask],
            y_pred[mask],
        )

        rows.append({
            "Activity": activity,
            "Test_Windows": int(mask.sum()),
            **metrics,
        })

    return pd.DataFrame(rows)


def activity_mean_baseline(
    y_train,
    train_activities,
    test_activities,
    activity_order,
):
    """
    Diagnostic baseline:
    predict each test sample using that activity's mean HR from training data.

    This intentionally uses activity labels and is not a deployable HR estimator.
    """
    y_train = np.asarray(y_train)
    train_activities = np.asarray(train_activities)
    test_activities = np.asarray(test_activities)

    activity_mean_map = {}

    for activity in activity_order:
        mask = train_activities == activity

        if np.any(mask):
            activity_mean_map[activity] = float(
                np.mean(y_train[mask])
            )

    predictions = np.array(
        [
            activity_mean_map[a]
            for a in test_activities
        ],
        dtype=float,
    )

    return predictions, activity_mean_map


def classification_metrics(
    y_true,
    y_pred,
):
    """
    Overall classification metrics.
    """
    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "Macro_F1": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
        "Weighted_F1": f1_score(
            y_true,
            y_pred,
            average="weighted",
        ),
    }


def classification_report_dataframe(
    y_true,
    y_pred,
    class_names,
):
    """
    Return sklearn classification report as a DataFrame.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).T


def classification_confusion_matrix(
    y_true,
    y_pred,
    num_classes,
    normalize=False,
):
    """
    Return raw or row-normalized confusion matrix.
    """
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
    )

    if normalize:
        row_sums = cm.sum(
            axis=1,
            keepdims=True,
        )

        row_sums[row_sums == 0] = 1

        cm = (
            cm.astype(float)
            / row_sums
        )

    return cm
