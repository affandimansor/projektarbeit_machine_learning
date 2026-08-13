import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import heatmap

# Die Funktion erstelle den Confusion Matrix des Ergebnisses und visualisiert ihn auf einem Heatmap
def CreateVisualizeConfusionMatrix(y_test, y_predicted):
    cm = confusion_matrix(y_test, y_predicted)  # Einen Confusion Matrix aus y_test und y_predicted bzw. knn.predict() erstellen
    cols = ["Low", "Average", "High"]           # Definiere die Spaltennamen, die auf das Heatmap-Diagramm einzutragen sind
    hm = heatmap(cm,                            # Das Heatmap definiere
                row_names=cols,
                column_names=cols,
                colorbar=True,
                cmap=plt.cm.YlGn)

    # Allegemeine Konfigurationen fuer das Heatmap
    plt.title('Confusion Matrix of the credit score')
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.show()

# Die Funktion prueft das Fitting ein Modell anhand der accuracy scores (as)
# Alle Eingabeparameter muessen in % uebergeben werden
def CheckModelFit(as_train, as_test, underfit_threshold=None, overfit_threshold=None):
    # Definiere den Defaultwert des underfit_thresholds
    if underfit_threshold is None:
        underfit_threshold = 50

    # Definiere den Defaultwert des overfit_threshold
    if overfit_threshold is None:
        overfit_threshold = 5

    # Geben die beiden Werte aus
    print(f"accuracy_train: {as_train}%; accuracy_test: {as_test}%")

    # Entscheide zw. Under- und Overfit
    if as_train < underfit_threshold and as_test < underfit_threshold:
        fit = f"Model is underfit"
        threshold = underfit_threshold
    else:
        fit = "Model is overfit" if as_train - as_test > overfit_threshold else "Model generalizes well to unseen data"
        threshold = overfit_threshold

    # Geben das fit und den threshold zurueck
    return fit, threshold