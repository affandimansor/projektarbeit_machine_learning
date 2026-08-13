import setup_utils_path
from utils.src.data_utils import ReadCSVDataset
from utils.src.result_utils import CheckModelFit
from utils.src.result_utils import CreateVisualizeConfusionMatrix
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

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
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
model = Sequential([
    Input(shape=(7,)),              # Definiere die Anzahl der Merkmale in der Eingabeschicht
    Dense(40, activation='relu'),   # Eingabeschicht mit 40 Neuronen, entspricht 40 Proben bzw. Samples
    Dense(10, activation='relu'),   # Versteckte Schicht mit 10 Neuronen
    Dense(3, activation='softmax')  # Ausgabeschicht mit 3 Neuronen, welche der Anzahl der Klassen bezeichnung entsprechen
                                    # Die Aktivierungsfunktion softmax ist fuer Multiclass Zielvariable wie in diesem Fall geeignet
                                    #? Die Aktivierungsfunktion Softmax braucht mindestens 2 Neurons, um richtigen Output auszugeben
 ])

# -----------------------------------------
# Das Modell kompilieren
# -----------------------------------------
# Definiere einen Optimizer
adam_optimizer = Adam(learning_rate=1e-04)

# Das neuronale Modell kompilieren bzw. aufbauen
model.compile(optimizer=adam_optimizer,
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
              loss=tf.keras.losses.SparseCategoricalCrossentropy)   # sparse_categorical_crossentropy ist eine richtige Wahl fuer den Fall, dass richtiges Label bzw. die Zielvariable 
                                                                    # durch Integers dargestellt (aber nicht One Hot Coding!) ist, also (0, 1, 2, ...)
                                                                    #? Dem One Hot Coding langt die Loss- bzw. Straffunktion categorical_crossentropy, da das richtiges Label als binaer
                                                                    #? statt Integers dargestellt ist, siehe https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy

# -----------------------------------------
# Das Modell trainieren 
# -----------------------------------------
history = model.fit(X_train_std,
          y_train,
          epochs=100,           # Hoehere Epochs ergeben sich kein genaueres Ergebnis
          batch_size=1,         # Alle Gewichte werden nach Bearbeitung eines bzw. jedes Elements oder Datenpunktes aktualisiert
          validation_split=0.2, # Die Daten fuer Validierung aufsplitten
          shuffle=True,         # Die Daten wird weiterhin gemischt, durch den oben definierten
                                # Random State wird die Mischung reproduzierbar durchgefuehrt      
          verbose=0)            # Progressbar auf das Terminal nicht anzeigen

print(history.history.keys())   # Alle Keywords aus dem History Objekt anzeigen lassen, falls unsicher, welche in dem History vorhanden sind

# Auf den letzten accuracy_score des Trainings zugreifen, da das Modell die Parameter aus diesem Stand enthaelt
acc_train = history.history['sparse_categorical_accuracy'][-1]

# Ermittelt den Index, bei dem die maximale accuracy erstmals auftaucht. Dieser kann im Nachhinein als die Anzahl der Epochs verwendet werden.
accuracies = history.history['sparse_categorical_accuracy']
print("first max at the index: %f " % accuracies.index(max(accuracies)))

# -----------------------------------------
# Das Model auf den Testdaten validieren
# -----------------------------------------
# Vorhersagen anhand X_test_std treffen
# Das Ergebnis ist die Wahrscheinlichkeiten der Klassenzugehoerigkeit, welche deren Summe 1 betraegt
y_pred = model.predict(X_test_std)

# Das Ergebnis zwecks Debug anzeigen lassen
compare_y = pd.DataFrame(
    {
        "y_test" : y_test.values,
        "y_pred_Low" : y_pred[:,0],
        "y_pred_Average" : y_pred[:,1],
        "y_pred_High" : y_pred[:,2],
    }
)
print(compare_y.head(n=5))

# -----------------------------------------
# Confusion Matrix und ihre Darstellung auf Heatmap
# -----------------------------------------
# Ein Inference Modell zum Konvertieren der Ausgaben des neuronalen Modells in die einzelnen Klassenbezeichnungen erstellen
# Dabei wird ein Lambda Wrapper dem neuronalen Modell eingebaut
inference_model = Sequential([
    model,                                      # Das neuronale Modell mit Eingabe-, Zwischen(versteckt)- und Ausgabeschichten
    Lambda(lambda x: tf.math.argmax(x, axis=1)) # Die Lambdaschicht als Wrapper definieren
])

# y_pred enthaelt nun die wahren Klassenbezeichnungen statt nur die Wahrscheinlickeiten der Klassenzugehoerigkeit
y_pred = inference_model.predict(X_test_std)

# Einen Confusion Matrix und abschließend ein Heatmap erstellen
CreateVisualizeConfusionMatrix(y_test=y_test, y_predicted=y_pred)

# -----------------------------------------------------
# Das Modellfitting analysieren
# -----------------------------------------------------
# Die Leistung des Modells mit den Testdaten bewerten
loss_test, acc_test = model.evaluate(X_test_std, y_test)

# Bestimme das Modellfitting und gebe es auf das Terminal aus
fit, threshold = CheckModelFit(acc_train*100, acc_test*100)
print(f"Optimized Neural Networks: {fit}; underfit threshold = {threshold}%")