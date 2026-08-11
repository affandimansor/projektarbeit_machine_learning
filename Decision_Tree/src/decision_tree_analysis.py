import setup_utils_path                                         # Fuege den Ordnerpfad utils dem Systempfad hinzu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from utils.src.data_utils import ReadCSVDataset
from utils.src.result_utils import CreateVisualizeConfusionMatrix

# -----------------------------------------------------
# Auslesen der Datenmenge
# -----------------------------------------------------

#? Annahme: Standardisierte Features sind beim Decision Tree nicht notwendig, aber sie werden aufgrund Ihrer Vorhandensein verwendet.
X_train_std, X_test_std, y_train, y_test = ReadCSVDataset(trainds="standardized_trainingdata_credit_score.csv",
                                                 testds="standardized_testdata_credit_score.csv")

# -----------------------------------------------------
# Bestimme die optimale Tiefe des Entscheidungsbaums
# -----------------------------------------------------
max_depths = np.arange(1,10)    # Array von unterschiedlichen max_depth-Werten
acc_scores = []                 # Die Genauigkeit des Modells bei jedem max_depth
act_depth = []                  # Die tatsaechlichen (bzw. actual) Tiefe des Modells bei jedem max_depth

for max_depth in max_depths:
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=max_depth,
                                  random_state=1)
    tree.fit(X_train_std, y_train)
    acc = accuracy_score(y_test, tree.predict(X_test_std)) * 100
    acc_scores.append(acc)
    act_depth.append(tree.get_depth())
    print(f"Max_depth: {max_depths} | Actual depth built: {tree.get_depth()}")
    print("Acccuracy_score: %.2f" % acc)

# Visualisiere das Ergebnis
fig, ax1 = plt.subplots()

# Definiere die Kurve von accuracy_scores
ax1.plot(max_depths,
         acc_scores,
         c='blue',
         lw=2,
         label='accuracy_score')
ax1.set_xlabel('max_depths [-]')
ax1.set_ylabel('accuracy_score [%]', color='b')
ax1.tick_params('y', colors='b')


# Definiere die Kurve von tatsaechlichen Tiefen
ax2 = ax1.twinx()
ax2.plot(max_depths,
         act_depth,
         c='green',
         lw=2,
         label='actual_depth')
ax2.set_ylabel('actual_depth [-]', color='g')
ax2.tick_params('y', colors='g')

# Die Kurven und ihr Label extrahieren und in die entsprechenden Liste abspeichern
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2

# Allgemeine Konfiguration des Diagramms
plt.legend(lines, labels, loc='upper right')
plt.grid()
plt.show()


# -----------------------------------------------------
# Optimale Parameter durch Cross-Validation suchen
# -----------------------------------------------------
# Die zu untersuchenden Parameter definieren
parameter = {
    "min_samples_leaf" : range(1, 10),
    "max_depth" : range(1, 5),
    "criterion" : ['gini', 'entropy'],
    "ccp_alpha" : tree.cost_complexity_pruning_path(X_train_std, y_train)['ccp_alphas'] # ccp_alphas werden durch diese cost_complexity_pruning_path() festgelegt bzw. berechnet
}

# GridSearchCV durchfuehren
grid_search =  GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=parameter,
    cv=5,   # Benutze das Default 5-fache CrossValidation
    n_jobs=1, # Bedeutet keine Jobs parallel laufen
    scoring='accuracy'
)

# Trainiere ein DecisionTree Modell mit verschiedenen Parameterkombinationen, um die beste Parameterkombination festzustellen
grid_search.fit(X_train_std, y_train)

# Die beste Parameterkombination und ihren Score bzw. ihre Genauigkeit ausgeben
print(f"Best parameter combination:\n {grid_search.best_params_}\nAccuracy score: {np.round(grid_search.best_score_ * 100, 2)}%")

# -----------------------------------------------------
# Das beste Modell mit Testdaten validieren
# -----------------------------------------------------
# Das DecisionTree Modell mit bester Parameterkombination aufrufen
best_DT = grid_search.best_estimator_

# Vorhersagen mit Testdaten
y_pred = best_DT.predict(X_test_std)

# Berechne das accuracy_score
print(f"Accuracy: {np.round(accuracy_score(y_test, y_pred)*100, 2)} %")

# ConfusionMatrix erstellen und ihn auf ein Heatmap darstellen
CreateVisualizeConfusionMatrix(y_test, y_pred)
