import setup_utils_path
from utils.src.data_utils import ReadCSVDataset
from utils.src.result_utils import CheckModelFit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# -------------------------------------------------
# Die Reproduzierbarkeit konfigurieren
# -------------------------------------------------
tf.keras.utils.set_random_seed(42)              # Vergleichbar mit random_state=42
tf.config.experimental.enable_op_determinism()  # Damit TensorFlow nur deterministische Operationen durchfuehrt. Somit wird
                                                # die Reproduzierbarkeit der Ergebnisse gewaehrleistet

# -----------------------------------------
# Die Datenmenge aus csv-Dateien auslesen
# -----------------------------------------
X_train_std, X_test_std, y_train, y_test = ReadCSVDataset(trainds="standardized_trainingdata_credit_score.csv",
                                                          testds="standardized_testdata_credit_score.csv")

# -----------------------------------------
# Die neuronale Schichten definieren
# -----------------------------------------
# Ein neuronales Modell durch die Klasse Sequential definiere
model = Sequential([
    Input(shape=(7,)),              # Definiere die Anzahl der Merkmale in der Eingabeschicht
    Dense(40, activation='relu'),   # Eingabeschicht mit 40 Neuronen, entspricht 40 Proben bzw. Samples
    Dense(10, activation='relu'),   # Versteckte Schicht mit 10 Neuronen
    Dense(3, activation='sigmoid')  # Ausgabeschicht mit 30 Neuronen
])

# -----------------------------------------
# Das Modell kompilieren
# -----------------------------------------
# Definiere einen Optimizer
adam_optimizer = Adam(learning_rate=1e-04)

# Das neuronale Modell kompilieren bzw. aufbauen
model.compile(optimizer=adam_optimizer,
              loss=tf.keras.losses.MSE,
              metrics=['accuracy'])

# -----------------------------------------
# Das Modell trainieren
# -----------------------------------------
history = model.fit(X_train_std,
          y_train,
          epochs=50,
          batch_size=1,         # Alle Gewichte werden nach Bearbeitung eines bzw. jedes Elements oder Datenpunktes aktualisiert
          validation_split=0.2, # Die Daten fuer Validierung aufsplitten
          shuffle=True,         # Die Daten wird weiterhin gemischt, durch den oben definierten
                                # Random State wird die Mischung reproduzierbar durchgefuehrt      
          verbose=0)            # Progressbar auf das Terminal nicht anzeigen

# Auf den letzten accuracy_score des Trainings zugreifen. Annahme, dass dieser der beste accuracy_score ist.
acc_train = history.history['accuracy'][-1]

# -----------------------------------------------------
# Das Modellfitting analysieren
# -----------------------------------------------------
# Die Leistung des Modells mit den Testdaten bewerten
loss_test, acc_test = model.evaluate(X_test_std, y_test)

# Bestimme das Modellfitting und gebe es auf das Terminal aus
fit, threshold = CheckModelFit(acc_train*100, acc_test*100)
print(f"Neural Networks: {fit}; underfit threshold = {threshold}%")