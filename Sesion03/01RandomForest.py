#Un arbol de decision es un modelo de aprendizaje, que a partir de una serie de preguntas (condiciones sobre ciertas caracteristicas), organizadas de forma tal que da un arbol para llegar a una predicción.

#El problema de los arboles de desiciones es el overfitting, (sobreajuste) ya que memoriza los datos de entrenamiento tan bien que pierde la capacidad de generalizar datos nuevos.

#vamos a simular una comparación a partir de arboles de desiciones para nuestro algoritmo claisifficador de cancer

from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier
#este nos va  atraer los diferentes arboles de desiciones 

from sklearn.tree import DecisionTreeClassifier #es para poder comparar un abol de desiciones vs RF

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score   # proporsion de prediccion correcta (calcular los errores)

import numpy as np
import matplotlib.pyplot as plt

#solo para recordar este dataset trae 568 muestras de tumores, con 30 diferentes caracteristicas para identificar si es binigno o maligno

data=load_breast_cancer()

X, y = data.data, data.target

#empezamos con el entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=40)

#vamos hacer el modelo

#arbol de desiciones
dt = DecisionTreeClassifier(random_state=40).fit(X_train, y_train)

#Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=40).fit(X_train, y_train)

#calcular la veracidad o nivel de predicción de los arboles

print(f'Abol de decisiones: {accuracy_score(y_test, dt.predict(X_test)):.4f}')
print(f'Random Forest: {accuracy_score(y_test, rf.predict(X_test)):.4f}')

#tenemos que verificar de las caracteristicas cuales son las mas importantes para el modelo, 

importancia = rf.feature_importances_

#los elementos del ordenamiento de los indices de menor a mayor importancia son
# [ : : -] los invierte (de mayor a menor)
# [ : 10] de tomar solo los 10 primeros elementos

indices = np.argsort(importancia)[::-1][:10]

#vamos a graficar

plt.barh(
    range(10), #posiciones verticales del 0 al 9
    importancia[indices], #valores de importancia
    color='steelblue', edgecolor='white'
)

#vamos a obtener las caracteristicas mas importantes (10)

plt.yticks(range(10), [data.feature_names[i] for i in indices])

plt.xlabel('Importancia (reduccion media del indice: )')

plt.title('Top 10 de caracteristicas mas importantes para el modelo')

#vamos a invertirlo
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

