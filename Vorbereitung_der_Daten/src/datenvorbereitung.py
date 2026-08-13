import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.plotting import scatterplotmatrix
import matplotlib.pyplot as plt

# Daten aus der csv-Datei einlesen
credit_score = pd.read_csv(os.path.join("..", "..", "Kursmaterial", "Credit_Scores.csv"))   # Den Ordnerpfad durch os.path.join zusammenbauen, damit kein Konflikt beim 
                                                                                            # Ausfuehren dieses Skriptes in unterschiedlichen Betriebssystemen entsteht.

# Alle Spaltennamen umschreiben
credit_score.columns = credit_score.columns.astype(str).str.replace(r'[^A-Za-z0-9]', '_', regex=True)   # Nur Zahlen und Buchstaben erlaubt, alle Sonderzeichen werden mit _ ersetzt
credit_score.columns = [column.lower() for column in credit_score.columns]                              # Nur Kleinbuchstaben


# Annahme: Falls vorhanden, die Zeilen mit fehlenden Daten aus dem Datensatz entfernen
# Annahme: Genuegende Datenpunkte nach dem Entfernen dieser Zeilen sind vorhanden
credit_score.dropna(axis=0, inplace=True)

# Die kategorialen Daten aufteilen
X = credit_score.drop("credit_score", axis=1)                               # X beinhaltet nur die Merkmalspalten
y = credit_score["credit_score"]                                            # y besteht ausschließlich nur aus der Spalte credit_score

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y,
                                                    test_size=0.3,          # Datenmenge in 70% Training und 30% Test aufteilen
                                                    random_state=42,        # Fuer reproduzierbare Ergebnisse ist random_state=42 ausgewaehlt
                                                    stratify=y)             # Damit die Verhaeltnisse des Vorkommens der verschiedenen Klassenbezeichnungen dem Verhältnis
                                                                            # in der Eingabedatenmenge entsprechen

# Die ordinalen Variablen (Credit Score und Education) in Integer-Arrays konvertieren
# Transformiere die Spalte Credit Score
score_categories = [['Low', 'Average', 'High']]                                                 # Definiere eine 2D-Score Liste mit ansteigenden Größen in Reihenfolge
ordinal_encoder = OrdinalEncoder(categories=score_categories)                                   # Definiere ein OrdinalEncoder-Objekt mit vorgegebenen score_categories
credit_score["credit_score"] = ordinal_encoder.fit_transform(credit_score[["credit_score"]])    # Transformiere die kategoriale, ordinale credit scores in Integer 2D-Array
print(np.unique_counts(credit_score["credit_score"]))

# Transformiere die Spalte Education
print(set(credit_score["education"]))                                                   # Unique Kategorien ausgeben
education_categories = [["High School Diploma", "Associate's Degree",                   # Die Education Kategorien steigend in eine 2D-Liste anordnen
                         "Bachelor's Degree", "Master's Degree", 
                         "Doctorate"]]
ordinal_encoder = OrdinalEncoder(categories=education_categories)                       # Definiere ein OrdinalEncoder-Objekt mit vorgegebenen education_categories
credit_score["education"] = ordinal_encoder.fit_transform(credit_score[["education"]])  # Transformiere die kategoriale, ordinale education in Integer 2D-Array

# Die kategorialen, nominalen Merkmale in Integer-Arrays umwandeln
label_encoder = LabelEncoder()
for feature in credit_score[["gender", "marital_status", "home_ownership"]]:
    credit_score[feature] = label_encoder.fit_transform(credit_score[feature].values)


# Die Abhängigkeiten zw. den Merkmalen und der Zielvariable auf einem Streudiagramm visualisieren
scatterplotmatrix(credit_score.values,
                  names=credit_score.columns,
                  figsize=(20,10),
                  alpha=0.5)
plt.tight_layout()
plt.show()

#? Fazit:
#?  1. Da die Merkmale meistens normalverteilt sind, koennten sie standardisiert statt nur normiert werden.
#?  2. Aufgrund bedingter Daten des Credit Scores 0 in der Datenmenge, werden die ML-Modelle nur über ihre Fähigkeit 
#?  zum Vorhersagen des Credit Scores 1 und 2 bewertet.

# Die Daten aufteilen und durch den StandardScaler normieren (Der Parameter random_state beim train_test_split() ist für reproduzierbares Ergebnis wichtig)
X = credit_score.drop("credit_score", axis=1)                           # X beinhaltet nur die Merkmalspalten
y = credit_score["credit_score"]                                        # y besteht ausschließlich nur aus der Spalte credit_score

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,      # Datenmenge in 70% Training und 30% Test aufteilen
                                                    random_state=42,    # Fuer reproduzierbare Ergebnisse ist random_state=42 ausgewaehlt
                                                    stratify=y)         # Damit die Verhaeltnisse des Vorkommens der verschiedenen Klassenbezeichnungen dem Verhältnis
                                                                        # in der Eingabedatenmenge entsprechen

# Standardisiere die Merkmale
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)                                 # Trainiere den Scaler mit dem Datensatz X_train und abschließend transformiert der Scaler den Datensatz
X_test_std = sc.transform(X_test)                                       # Transformiere den Datensatz X_test durch den Scaler mit den gelernten Parametern aus dem Trainingsdatensatz, damit 
                                                                        # keine Datenleckage zw. Trainings- und Testdatensatz auftritt

# Alle relevanten Datensaetze als csv-Datei exportieren
feature_cols = credit_score.columns[:-1]                        # Definiere die Spaltennamen fuer die Merkmale
train_std_df = pd.DataFrame(X_train_std, columns=feature_cols)  # Ein DataFrame Objekt für X_train_std erzeuge
train_std_df["credit_score"] = y_train.values                   # Füge y_train als neue Spalte credit_score hinzu

test_std_df = pd.DataFrame(X_test_std, columns=feature_cols)    # Ein DataFrame Objekt für X_test_std erzeuge
test_std_df["credit_score"] = y_test.values                     # Füge y_test als neue Spalte credit_score hinzu

train_cat_df = pd.DataFrame(X_train_cat, columns=feature_cols)  # Ein DataFrame Objekt für X_train_cat erzeuge
train_cat_df["credit_score"] = y_train_cat.values               # Füge y_cat_train als neue Spalte credit_score hinzu

test_cat_df = pd.DataFrame(X_test_cat, columns=feature_cols)    # Ein DataFrame Objekt für X_test_cat_std erzeuge
test_cat_df["credit_score"] = y_test_cat.values                 # Füge y_cat_test als neue Spalte credit_score hinzu

# Erstelle einen Ordner fuer die Ausgabedateien
output_folder = "../export"
os.makedirs(output_folder, exist_ok=True)

# Definiere die zu exportierenden Dateinamen
filenames = ["transformed_credit_score.csv",
              "standardized_trainingdata_credit_score.csv",
              "standardized_testdata_credit_score.csv",
              "categorial_trainingdata_credit_score.csv",
              "categorial_testdata_credit_score.csv"]

# Packe alle Datensaetze in einer Liste fuer for-Schleife ein
datasets = [credit_score, train_std_df, test_std_df, train_cat_df, test_cat_df]

# Exportiere jeden Datensatz als eigene csv-Datei
for filename, dataset in zip(filenames, datasets):
    filepath = os.path.join(output_folder, filename)                                # Definiere den Dateipfad
    dataset.to_csv(path_or_buf=filepath, sep=';', encoding='utf-8', index=False)    # Exportiere den Datensatz als csv-Datei