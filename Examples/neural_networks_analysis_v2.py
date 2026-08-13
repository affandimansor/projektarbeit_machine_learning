import setup_utils_path
from utils.src.data_utils import ReadCSVDataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# -------------------------------------------------
# REPRODUZIERBARKEIT
# -------------------------------------------------

# ÄNDERUNG: globalen Zufalls-Seed für TensorFlow/Keras,
# NumPy und Python setzen. "random_state" wird hier nicht verwendet.
tf.keras.utils.set_random_seed(42)

# ÄNDERUNG: TensorFlow-Operationen möglichst deterministisch ausführen.
# Dadurch werden insbesondere mögliche Unterschiede durch nicht-
# deterministische TensorFlow-Operationen reduziert.
tf.config.experimental.enable_op_determinism()


# -------------------------------------------------
# Die Datenmenge aus csv-Dateien auslesen
# -------------------------------------------------

X_train_std, X_test_std, y_train, y_test = ReadCSVDataset(
    trainds="standardized_trainingdata_credit_score.csv",
    testds="standardized_testdata_credit_score.csv"
)


# -------------------------------------------------
# Die neuronalen Schichten definieren
# -------------------------------------------------

input_shape = (7,)

model = Sequential([
    Input(shape=input_shape),
    Dense(40, activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])


# -------------------------------------------------
# Das Modell kompilieren
# -------------------------------------------------

adam_optimizer = Adam(learning_rate=1e-04)

model.compile(
    optimizer=adam_optimizer,
    loss=tf.keras.losses.MSE,
    metrics=['accuracy']
)


# -------------------------------------------------
# Das Modell trainieren und seine Leistung bewerten
# -------------------------------------------------

history = model.fit(
    X_train_std,
    y_train,
    epochs=50,
    batch_size=1,
    validation_split=0.2,

    # ERKLÄRUNG: Die Trainingsdaten werden weiterhin gemischt.
    # Durch den oben gesetzten Seed erfolgt das Mischen reproduzierbar.
    shuffle=True,

    verbose=0
)


# -------------------------------------------------
# Die Leistung des Modells bewerten
# -------------------------------------------------

loss, accuracy = model.evaluate(X_test_std, y_test)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")