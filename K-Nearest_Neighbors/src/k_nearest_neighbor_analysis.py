import setup_utils_path                                     # Aktualisiere des sys.path, damit der Ordner utils auf dem root Ordner zugaenglich ist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils.src.data_utils import ReadCSVDataset
from utils.src.result_utils import CreateVisualizeConfusionMatrix, CheckModelFit

# -----------------------------------------------------
# Auslesen der Datenmenge
# -----------------------------------------------------
X_train_std, X_test_std, y_train, y_test = ReadCSVDataset(trainds="standardized_trainingdata_credit_score.csv",
                                                          testds="standardized_testdata_credit_score.csv")

# -----------------------------------------------------
# Stelle die beste Azahl der Nachbarn k fest
# -----------------------------------------------------
acc_scores = []                     # Liste zum Abspeichern der Genauigkeit des Modells je k
n_neighbors = np.arange(1, 30)      # Der Wertebereich fuer k

for neighbors in n_neighbors: 
    knn = KNeighborsClassifier(n_neighbors=neighbors,                           # n_neighbors wird variiert
                           p=2,                                                 # euklidischer Abstand
                           metric='minkowski')          
    knn.fit(X_train_std, y_train)                                               # Trainiere das knn Model
    acc_scores.append(accuracy_score(knn.predict(X_test_std), y_test) * 100)    # Berechne und abspeichere die Modellgenauigkeit jedes k

# -----------------------------------------------------
# Bestimme die beste k und visualisiere das Ergebnis
# -----------------------------------------------------
# Bestimmen die beste k
best_acc = max(acc_scores)                          # Die maximale Genauigkeit auslesen
best_k = n_neighbors[acc_scores.index(best_acc)]    # Bestimme k fuer das maximale Genauigkeit

# Kurve der Genauigkeiten plotten
plt.plot(n_neighbors,
            acc_scores,
            c='blue',
            lw=2)

# Zeichne den Bestepunkt auf das Diagramm ein
plt.scatter(best_k,
            best_acc,
            marker='*',
            s=50,
            c='red',
            label=f"Best k ({best_k}, {best_acc})")

# Allgemeine Konfigurationen des Diagramms
plt.legend()
plt.title("Accuracy score against number of neighbors k")
plt.grid()
plt.xlabel("n_neighbors [-]")
plt.ylabel("Accuracy score [%]")
plt.show()

# -----------------------------------------------------
# Ein kNN-Modell mit der besten k 
# -----------------------------------------------------
# Ein kNN-Modell mit n_neighbors = best_k definiere
knn = KNeighborsClassifier(n_neighbors=best_k,
                           p=2,
                           metric='minkowski')

# Trainiere das Modell
knn.fit(X_train_std, y_train)

# -----------------------------------------------------
# Confusion Matrix und Heatmap
# -----------------------------------------------------
# Rufe die entsprechende Funktion aus der utils/results_utils.py zum Erstellen und 
# Darstellen des Confusion Matrixes auf einem HeatMap auf
y_pred = knn.predict(X_test_std)
CreateVisualizeConfusionMatrix(y_test, y_pred)

# -----------------------------------------------------
# Das Modellfitting analysieren
# -----------------------------------------------------
# Berechne das accuracy_score der Testdaten
acc_test = accuracy_score(y_test, y_pred) * 100

# Bestimme das Modellfitting und gebe es auf das Terminal aus
fit, threshold = CheckModelFit(best_acc, acc_test)
print(f"k-Nearest-Neighbors: {fit}; gap threshold = {threshold}%")