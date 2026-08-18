"""
Aufgabe über Wine Qualität von Florian Michel

"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap, scatterplotmatrix

# 1. Load dataset (Red Wine dataset)
url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df_red = pd.read_csv(url_red, sep=';')  # Der Parameter sep ist notwendig, da sonst Pandas alle Spalten in einer Spalte anpacken wird

df_red.columns = df_red.columns.astype(str).str.replace(r'[^A-Za-z0-9]', '_', regex=True)   # Die Sonderzeichen wir Leerzeichen mit '_' ersetzen
print(df_red.head())

# -- TODO: Datenqualität prüfen, z.B. nach NaN prüfen, ob string in Zellen vorhanden sind.
#! -- Geben die Anzahl der Datenpunkte nach quality aus, um zu sehen, ob genügende Datenpunkte für jede Qualitätsklasse vorhanden sind, falls nicht
#! -- muss man dann die Daten anders weiteranalysieren.

# 2. Calculate the correlation matrix
corr_matrix = df_red.corr(numeric_only=True)


# 3. Extract top 5 features with highest absolute correlation to 'quality'
top_5_features = (
    corr_matrix['quality']
    .drop('quality')           # Exclude 'quality' self-correlation (1.0)
    .abs()                     # Use absolute values to capture strong negative correlations too
    .nlargest(5)               # Select top 5
    .index.tolist()            # Convert index names into a Python list
)
# -- TODO: Zu überlegen, ob nur mit top 5 features weiterzuarbeiten oder besser nur manche davon und dann noch weitere Features mit niedrigerer Korrelation in Analyse zu ziehen.

# 4. Plot full heatmap using mlxtend
fig, ax = heatmap(
    matrix=corr_matrix.values,
    row_names=corr_matrix.columns,
    column_names=corr_matrix.columns,
    cmap='coolwarm',
    figsize=(10, 10),
    column_name_rotation=45,
    cell_values=True,
    cell_fmt='.2f',
    cell_font_size=8
)

# 5. Highlight top 5 feature labels in bold red on X and Y axes
# for label in ax.get_xticklabels():
#     if label.get_text() in top_5_features:
#         label.set_color('crimson')
#         label.set_weight('bold')

for label in ax.get_yticklabels():
    if label.get_text() in top_5_features:
        label.set_color('green')
        label.set_weight('bold')

plt.title("Full Heatmap with Top 5 Features Highlighted (Red)", fontsize=13, pad=12)
plt.tight_layout()
plt.show()

df = df_red[top_5_features + ["quality"]]   # Quality is wrapped into a list and then this list is added into the top_5_features list

# Scatterplot von den top_5_features visualisieren
scatterplotmatrix(df.values, figsize=(10,8), names=df.columns, alpha=0.5)
plt.tight_layout()
plt.show()


# ----- Daten aufteilen und skalieren bzw. standardisieren -----
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[top_5_features]
y= df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # random_state ist ein wichtiger Parameter, sonst ist das Ergebnis nicht reproduzierbar, weil Daten beim jedem Durchlauf anders in Trainings- und Testdaten gesplittet werden.
print(X_train[:5])

std_sc = StandardScaler()
X_train_std = std_sc.fit_transform(X_train) # Trainiere den Scaler und transformiere den Trainingsdatensatz
X_test_std = std_sc.transform(X_test)       # Transformiere den Testdatensatz mit den Scaler Parameter aus dem Trainingsdatensatz zum Vermeiden von der Datenleckage


# ----- Modelle auswähle und trainiere
#? ---- TODO: Schleife über mehrere Modelle und dann Boxplot zeichnen lassen für einen Vergleich zwischen denen. Damit kann man den ersten Algorithmus auswählen und ihn dann durch Hyperparameter Tuning optimieren.

# k-NN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50, p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)

# ----- Mit dem trainierten kNN-Modell vorhersagen
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# 1. Generate predictions and calculate accuracy
y_pred = knn.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)

# 2. Get unique classes & compute transposed confusion matrix
# (Transposing places Predicted Label on Y-axis, True Label on X-axis)
labels = sorted(list(set(y_test) | set(y_pred)))
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_transposed = cm.T

# 3. Create masked matrices for diagonal (correct) vs off-diagonal (incorrect)
n = len(labels)
is_diagonal = np.eye(n, dtype=bool)

cm_correct = np.where(is_diagonal, cm_transposed, np.nan)
cm_incorrect = np.where(~is_diagonal, cm_transposed, np.nan)

max_val = cm_transposed.max()

# 4. Plot Green diagonal using mlxtend.plotting.heatmap
# (cell_values=False prevents mlxtend from trying to format NaN cells)
fig, ax = heatmap(
    matrix=cm_correct,
    row_names=labels,
    column_names=labels,
    cmap='Greens',
    figsize=(8, 8),
    column_name_rotation=0,
    cell_values=False
)

# 5. Overlay Red off-diagonal on the same axis
ax.imshow(cm_incorrect, cmap='Reds', vmin=0, vmax=max_val)

# 6. Annotate exact cell values and set text contrast
for i in range(n):
    for j in range(n):
        val = cm_transposed[i, j]
        # Use white text for dark cells, black text for light cells
        text_color = "white" if val > (max_val / 2) else "black"
        ax.text(
            j, i, str(val), 
            ha="center", va="center", 
            color=text_color, fontweight="bold", fontsize=10
        )

# 7. Customize axis titles and display Accuracy
plt.xlabel("True Label", fontsize=12, fontweight="bold", labelpad=10)
plt.ylabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
plt.title(
    f"k-NN Confusion Matrix (Wine Quality)\nAccuracy Score: {accuracy:.4f} ({accuracy * 100:.2f}%)\nn_neighbors:{5} ",
    fontsize=13,
    pad=15,
    fontweight="bold"
)

plt.tight_layout()
plt.show()


# ----- Den Parameter n_neighbors iterativ für besseres Ergebnis untersuchen
# 1. Define range for n_neighbors (5 to 150, step by 10)
k_values = list(range(5, 151, 10))
accuracies = []

# 2. Loop through each k value and calculate test set accuracy
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    
    y_pred = knn.predict(X_test_std)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# 3. Find the best k value and its corresponding highest accuracy
best_acc = max(accuracies)
best_k = k_values[accuracies.index(best_acc)]

print(f"Optimal n_neighbors (k): {best_k}")
print(f"Highest Accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%)")

# 4. Plot accuracy against n_neighbors
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', linewidth=2, label='Test Accuracy')

# Highlight the best k point on the plot
plt.scatter(best_k, best_acc, color='red', s=120, zorder=5, label=f'Best k = {best_k} ({best_acc*100:.2f}%)')

# Formatting plot
plt.title("k-NN Accuracy vs. Number of Neighbors (n_neighbors)", fontsize=14, pad=12, fontweight="bold")
plt.xlabel("n_neighbors (k)", fontsize=12, fontweight="bold")
plt.ylabel("Accuracy Score", fontsize=12, fontweight="bold")
plt.xticks(k_values)  # Show ticks at every tested k value
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()