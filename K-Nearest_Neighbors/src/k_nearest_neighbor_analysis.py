import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Den Trainings- und Testdatensatz einlesen
dataset_dir = os.path.join("..", "..", "Vorbereitung_der_Daten", "export")  # Den Ordnerpfad durch os.path.join zusammenbauen, damit kein Konflikt beim Ausfuehren dieses Skriptes in unterschiedlichen Betriebssystemen entsteht.

train_std_df = pd.read_csv(os.path.join(dataset_dir, "standardized_testdata_credit_score.csv"), sep=';')
test_std_df = pd.read_csv(os.path.join(dataset_dir, "standardized_trainingdata_credit_score.csv"), sep=';')

print(train_std_df.head())

# Ein k-NN Modell initialisieren und trainieren
knn = KNeighborsClassifier(n_neighbors=5,
                           p=2,
                           metric='minkowski')