import os
import pandas as pd
import numpy as np

# Die Funktion liest vorbereiteten Daten aus und geben sie in Form von X- und y-Matrizen zurueck
def ReadCSVDataset(trainds, testds, ds_dir=None):
    if ds_dir is None:
        ds_dir=["..", "..", "Vorbereitung_der_Daten", "export"]

    # Den Trainings- und Testdatensatz einlesen
    ds_dir = os.path.join(*ds_dir)                          # Den Ordnerpfad durch os.path.join zusammenbauen, damit kein Konflikt 
                                                            # beim Ausfuehren dieses Skriptes in unterschiedlichen Betriebssystemen entsteht.
                                                            # Das * wird gebraucht, um auf das einzelne Element der Liste ds_dir zuzugreifen.

    train_df = pd.read_csv(os.path.join(ds_dir, trainds),   # Definiere den Dateipfad fuer den Trainingsdatensatz
                            sep=';')                        # Benutze ; als Spaltentrennzeichen
    test_df = pd.read_csv(os.path.join(ds_dir, testds),     # Definiere den Dateipfad fuer den Trainingsdatensatz
                            sep=';')                        # Benutze ; als Spaltentrennzeichen

    # Die Datensaetze in X- und y-Matrizen fuer weitere Berechnungen aufteilen
    # Trainingsdatensatz
    X_train = train_df.drop("credit_score", axis=1)
    y_train = train_df["credit_score"]

    # Testdatensatz
    X_test = test_df.drop("credit_score", axis=1)
    y_test = test_df["credit_score"]

    # Die Datensaetze zurueckgeben
    return X_train, X_test, y_train, y_test
