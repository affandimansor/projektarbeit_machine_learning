import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Den Trainings- und Testdatensatz einlesen
dataset_dir = os.path.join("..", "..", "Vorbereitung_der_Daten", "export")                          # Den Ordnerpfad durch os.path.join zusammenbauen, damit kein Konflikt 
                                                                                                    # beim Ausfuehren dieses Skriptes in unterschiedlichen Betriebssystemen entsteht.

train_std_df = pd.read_csv(os.path.join(dataset_dir, "standardized_testdata_credit_score.csv"),     # Definiere den Dateipfad fuer den Trainingsdatensatz
                           sep=';')                                                                 # Benutze ; als Spaltentrennzeichen
test_std_df = pd.read_csv(os.path.join(dataset_dir, "standardized_trainingdata_credit_score.csv"),  # Definiere den Dateipfad fuer den Trainingsdatensatz
                          sep=';')                                                                  # Benutze ; als Spaltentrennzeichen

# Die Datensaetze in X- und y-Matrizen fuer weitere Berechnungen aufteilen
# Trainingsdatensatz
X_train_std = train_std_df.drop("credit_score", axis=1)
y_train = train_std_df["credit_score"]

# Testdatensatz
X_test_std = test_std_df.drop("credit_score", axis=1)
y_test = test_std_df["credit_score"]


# Stelle die beste Azahl der Nachbarn k durch eine for-Schleife fest
acc_scores = []                     # Liste zum Abspeichern der Genauigkeit des Modells je k
n_neighbors = np.arange(1, 30)      # Der Wertebereich fuer k

for neighbors in n_neighbors: 
    knn = KNeighborsClassifier(n_neighbors=neighbors,                           # n_neighbors wird variiert
                           p=2,                                                 # euklidischer Abstand
                           metric='minkowski')          
    knn.fit(X_train_std, y_train)                                               # Trainiere das knn Model
    acc_scores.append(accuracy_score(knn.predict(X_test_std), y_test) * 100)    # Berechne und abspeichere die Modellgenauigkeit jedes k

# Visualisiere das Ergebnis
# Kurve der Genauigkeit plotten
plt.plot(n_neighbors,
            acc_scores,
            c='blue',
            lw=2)

# Bestimmen die beste k
best_acc = max(acc_scores)                          # Die maximale Genauigkeit auslesen
best_k = n_neighbors[acc_scores.index(best_acc)]    # Bestimme k fuer das maximale Genauigkeit

# Zeichne den Bestepunkt auf das Diagramm ein
plt.scatter(best_k,
            best_acc,
            marker='*',
            s=50,
            c='red',
            label='Best k')

# Allgemeine Konfigurationen des Diagramms
plt.legend()
plt.title("Accuracy score against number of neighbors k")
plt.grid()
plt.xlabel("n_neighbors [-]")
plt.ylabel("Accuracy score [%]")
plt.show()