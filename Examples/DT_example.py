from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# X und y durch deine eigenen Daten ersetzen
# X = Features / Eingangsvariablen
# y = Zielvariable / Klassen
# --------------------------------------------------

# Beispiel:
# X = df.drop("target", axis=1)
# y = df["target"]

# Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 1. Verschiedene maximale Baumtiefen testen
# --------------------------------------------------

for max_depth in range(1, 11):

    # Decision Tree mit der jeweiligen maximalen Tiefe erstellen
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )

    # Modell mit den Trainingsdaten trainieren
    model.fit(X_train, y_train)

    # Vorhersagen auf den Testdaten
    y_pred = model.predict(X_test)

    # Genauigkeit berechnen
    accuracy = accuracy_score(y_test, y_pred)

    # Tatsächlich erreichte Tiefe des Baumes
    actual_depth = model.tree_.max_depth

    print(
        f"max_depth = {max_depth:2d} | "
        f"tatsächliche Tiefe = {actual_depth:2d} | "
        f"Accuracy = {accuracy:.2f}"
    )

# --------------------------------------------------
# 2. Optimale Parameter mit Cross-Validation suchen
# --------------------------------------------------

param_grid = {
    "max_depth": range(1, 11),
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

# GridSearchCV testet alle Parameterkombinationen
# und verwendet dabei 5-fache Cross-Validation
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Optimierung auf den Trainingsdaten durchführen
grid_search.fit(X_train, y_train)

# Beste Parameter ausgeben
print("\nBeste Parameter:")
print(grid_search.best_params_)

# Beste durchschnittliche Cross-Validation-Accuracy
print("\nBeste Cross-Validation-Accuracy:")
print(f"{grid_search.best_score_:.3f}")

# --------------------------------------------------
# 3. Bestes Modell auf den Testdaten überprüfen
# --------------------------------------------------

best_model = grid_search.best_estimator_

# Vorhersagen mit dem optimierten Modell
y_pred = best_model.predict(X_test)

# Accuracy auf den bisher ungesehenen Testdaten
test_accuracy = accuracy_score(y_test, y_pred)

print("\nTest-Accuracy:")
print(f"{test_accuracy:.3f}")

# Tatsächliche Tiefe des optimierten Baumes
print("\nTatsächliche Tiefe des optimierten Baumes:")
print(best_model.tree_.max_depth)