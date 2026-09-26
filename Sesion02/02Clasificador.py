#vamos a desarrollar una aplicación para poder identificar florecitas , a partir de 3 especies, setosa, versicolor virginica

#vamos a comparar una regresión lineal vs Kvecinos 

from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target


#primero es crear el modelo de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=40, stratify=y #garantizar que las clases se puedan representar
)

#tenemos que ajustarlo a parte de StandarScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)

#vamos a desarrollar el modelo para K vecinos

modelos = {
    'Reg. Lineal': LogisticRegression(max_iter=200),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf')
}

for nombre, m in modelos.items():
    m.fit(X_train_s, y_train)
    print(f"\n == {nombre} == ")
    print(classification_report(
        y_test, m.predict(X_test_s),
        target_names=iris.target_names
    ))


#vamos a graficar

fig, axes = plt.subplots(1, 3, figsize=(15,4))

fig.suptitle('Matriz de Confusión por Clasificador de Iris', fontsize=14, fontweight='bold')

for ax, (nombre, m) in zip(axes, modelos.items()):
    cm = confusion_matrix(y_test, m.predict(X_test_s))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(nombre)
    ax.tick_params(axis='x', labelrotation=20)

plt.tight_layout()
plt.show()

