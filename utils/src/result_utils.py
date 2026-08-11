import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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