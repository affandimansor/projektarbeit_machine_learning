import setup_utils_path                                     # Aktualisiere des sys.path, damit der Ordner utils auf dem root Ordner zugaenglich ist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import heatmap
from utils.src.data_utils import ReadCSVDataset
from utils.src.result_utils import CreateVisualizeConfusionMatrix

# Den Trainings- und Testdatensaetze einlesen
X_train_std, X_test_std, y_train, y_test = ReadCSVDataset(trainds="standardized_trainingdata_credit_score.csv",
                                                          testds="standardized_testdata_credit_score.csv")

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

# ConfusionMatrix mit einem kNN-Modell bei k = 2
knn = KNeighborsClassifier(n_neighbors=best_k,          # Ein kNN-Modell mit n_neighbors = best_k bzw. 2 definiere
                           p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)                           # Trainiere das Modell

# Rufe die entsprechende Funktion aus der utils/results_utils.py zum Erstellen und 
# Darstellen des Confusion Matrixes auf einem HeatMap auf
CreateVisualizeConfusionMatrix(y_test, knn.predict(X_test_std))