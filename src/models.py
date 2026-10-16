import tensorflow as tf


def build_ppg_only_cnn(
    input_length: int = 320,
):
    """
    PPG-only 1D CNN regression baseline.
    """
    inputs = tf.keras.layers.Input(
        shape=(input_length, 1),
        name="ppg_input",
    )

    x = tf.keras.layers.Conv1D(
        32,
        7,
        padding="same",
        activation="relu",
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(
        64,
        5,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(
        128,
        3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    outputs = tf.keras.layers.Dense(
        1,
        name="predicted_hr",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="PPG_Only_1D_CNN",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def build_early_fusion_cnn(
    input_length: int = 320,
):
    """
    Early-fusion PPG + three-axis IMU regression model.
    """
    inputs = tf.keras.layers.Input(
        shape=(input_length, 4),
        name="ppg_imu_input",
    )

    x = tf.keras.layers.Conv1D(
        32,
        7,
        padding="same",
        activation="relu",
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(
        64,
        5,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(
        128,
        3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    outputs = tf.keras.layers.Dense(
        1,
        name="predicted_hr",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="PPG_IMU_Early_Fusion_CNN",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def build_two_branch_cnn(
    input_length: int = 320,
):
    """
    Two-branch PPG + IMU CNN regression model.
    """
    ppg_input = tf.keras.layers.Input(
        shape=(input_length, 1),
        name="ppg_input",
    )

    p = tf.keras.layers.Conv1D(
        32, 7, padding="same", activation="relu"
    )(ppg_input)
    p = tf.keras.layers.BatchNormalization()(p)
    p = tf.keras.layers.MaxPooling1D(2)(p)
    p = tf.keras.layers.Conv1D(
        64, 5, padding="same", activation="relu"
    )(p)
    p = tf.keras.layers.BatchNormalization()(p)
    p = tf.keras.layers.MaxPooling1D(2)(p)
    p = tf.keras.layers.Conv1D(
        96, 3, padding="same", activation="relu"
    )(p)
    p = tf.keras.layers.GlobalAveragePooling1D()(p)
    p = tf.keras.layers.Dense(64, activation="relu")(p)

    imu_input = tf.keras.layers.Input(
        shape=(input_length, 3),
        name="imu_input",
    )

    m = tf.keras.layers.Conv1D(
        32, 7, padding="same", activation="relu"
    )(imu_input)
    m = tf.keras.layers.BatchNormalization()(m)
    m = tf.keras.layers.MaxPooling1D(2)(m)
    m = tf.keras.layers.Conv1D(
        64, 5, padding="same", activation="relu"
    )(m)
    m = tf.keras.layers.BatchNormalization()(m)
    m = tf.keras.layers.MaxPooling1D(2)(m)
    m = tf.keras.layers.Conv1D(
        96, 3, padding="same", activation="relu"
    )(m)
    m = tf.keras.layers.GlobalAveragePooling1D()(m)
    m = tf.keras.layers.Dense(64, activation="relu")(m)

    fused = tf.keras.layers.Concatenate(
        name="feature_fusion"
    )([p, m])

    fused = tf.keras.layers.Dropout(0.30)(fused)
    fused = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(fused)
    fused = tf.keras.layers.Dropout(0.20)(fused)

    output = tf.keras.layers.Dense(
        1,
        name="predicted_hr",
    )(fused)

    model = tf.keras.Model(
        inputs=[ppg_input, imu_input],
        outputs=output,
        name="Two_Branch_PPG_IMU_CNN",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def build_cnn_bigru_hr_model(
    input_length: int = 320,
):
    """
    Final HR regression model:
    PPG branch + IMU branch, both using CNN + BiGRU.
    """
    ppg_input = tf.keras.layers.Input(
        shape=(input_length, 1),
        name="ppg_input",
    )

    p = tf.keras.layers.Conv1D(
        32, 7, padding="same", activation="relu"
    )(ppg_input)
    p = tf.keras.layers.BatchNormalization()(p)
    p = tf.keras.layers.MaxPooling1D(2)(p)
    p = tf.keras.layers.Conv1D(
        64, 5, padding="same", activation="relu"
    )(p)
    p = tf.keras.layers.BatchNormalization()(p)
    p = tf.keras.layers.MaxPooling1D(2)(p)

    p = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            48,
            return_sequences=False,
            dropout=0.10,
        ),
        name="ppg_bigru",
    )(p)

    p = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(p)

    imu_input = tf.keras.layers.Input(
        shape=(input_length, 3),
        name="imu_input",
    )

    m = tf.keras.layers.Conv1D(
        32, 7, padding="same", activation="relu"
    )(imu_input)
    m = tf.keras.layers.BatchNormalization()(m)
    m = tf.keras.layers.MaxPooling1D(2)(m)
    m = tf.keras.layers.Conv1D(
        64, 5, padding="same", activation="relu"
    )(m)
    m = tf.keras.layers.BatchNormalization()(m)
    m = tf.keras.layers.MaxPooling1D(2)(m)

    m = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            48,
            return_sequences=False,
            dropout=0.10,
        ),
        name="imu_bigru",
    )(m)

    m = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(m)

    fused = tf.keras.layers.Concatenate(
        name="feature_fusion"
    )([p, m])

    fused = tf.keras.layers.Dropout(0.30)(fused)
    fused = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(fused)
    fused = tf.keras.layers.Dropout(0.20)(fused)

    output = tf.keras.layers.Dense(
        1,
        name="predicted_hr",
    )(fused)

    model = tf.keras.Model(
        inputs=[ppg_input, imu_input],
        outputs=output,
        name="CNN_BiGRU_PPG_IMU_Fusion",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def build_activity_cnn(
    input_length: int = 320,
    num_classes: int = 6,
):
    """
    Final IMU-only activity classifier.
    """
    inputs = tf.keras.layers.Input(
        shape=(input_length, 3),
        name="imu_input",
    )

    x = tf.keras.layers.Conv1D(
        32, 7, padding="same", activation="relu"
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(
        64, 5, padding="same", activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(
        128, 3, padding="same", activation="relu"
    )(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="activity_probabilities",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="IMU_Activity_1D_CNN",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_activity_cnn_bigru(
    input_length: int = 320,
    num_classes: int = 6,
):
    """
    Comparison IMU-only CNN + BiGRU activity classifier.
    """
    inputs = tf.keras.layers.Input(
        shape=(input_length, 3),
        name="imu_input",
    )

    x = tf.keras.layers.Conv1D(
        32, 7, padding="same", activation="relu"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(
        64, 5, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            64,
            return_sequences=False,
            dropout=0.10,
        ),
        name="activity_bigru",
    )(x)

    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="activity_probabilities",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="IMU_CNN_BiGRU_Activity_Classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
