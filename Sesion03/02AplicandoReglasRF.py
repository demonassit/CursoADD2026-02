#Para este ejercicio vamos a aplicar diferetes reglas para evaluar y optimizar un modelo de clasificiación rigurosa.

# GridSearchCV: que es para la busqueda de hiperparametros, los cuales son configuraciones que no se aprenden del modelo o recorrido, esos los elige el programador, por ejemplo, el numero de aboles, la profundidad, las combinaciones de los posibles y la seleccion del mejor desempeño

# Validación Cruzada: en lugar de evaluar el modelo, con una sola visión de entrenamiento, (train/test) la validacion cruzada divide los datos en k partes (subconjuntos), el modelo se entrena k veces cad vez que usa k-1 de las partes para entrenar el siguiente y asi. El desempeño final es el promedio de las evaluaciones

#Curvas ROC y AUC: sirven para poder evaluar umbrales, un clasificador binario nos ayuda a predecir la probabilidad del umbral (0,1) y con esta curva, nosotros podemos aplicar la matriz de confusion, TPR verdadero positivo, hasta los FPR falso positivo. 

from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier
#este nos va  atraer los diferentes arboles de desiciones 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt

data=load_breast_cancer()

X, y = data.data, data.target

#empezamos con el entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=40)

#escalar las dimensiones
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)

#para la busqueda de los hiperparametros, tenemos que definir esas combinaciones 
#para ello vamos a ocupar a n_estimador, numero de aboles que deseamos para este bosque
#debemos definir la profundidad del arbol arbol -> node -> None (vacio) 
# tenemos que tener en consideración las k (las partes del entrenamiento) para considerar sus combinaciones de los elementos de validación cruzada).

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth' : [None, 5, 10]
}

#para los elementos de busqueda del grid, tenemos un modelo base que debe optimizarse, 
# estimador = modelo que queremos optimizar
# param_grid = combinaciones de hiperparametros que queremos evaluar (hiperparametros)
# cv = numero de particiones de validación cruzada
# tipo de criterio para el calculo de los errores scoring = f1  sirve para maximizar el score de clasificadores binarios 

gs = GridSearchCV(
    RandomForestClassifier(random_state=40),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

#ahora si lo vamos a entrenar

gs.fit(X_train, y_train)

#vamos a mejorar el modelo
best = gs.best_estimator_

#vamos a pasar por el reentrenamiento
mejor = gs.best_estimator_
y_pred = mejor.predict(X_test_s)
y_proba = mejor.predict_proba(X_test_s)[:, 1]


#vamos a reportar los resultados

print('Reporte Clasificador')
print(classification_report(
    y_test, y_pred, target_names=['Maligno', 'Benigno']))

fig, axes = plt.subplots(1,2, figsize=(13,5))
fig.suptitle(f'Evaluación del mejor modelo de Random Forest \n {gs.best_params_}', 
             fontsize=12, fontweight='bold')

#vamos a graficar la matriz de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Maligno', 'Benigno'])
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Matriz de Confusión')

#para verificar los errores
fpr, tpr, _ =roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='steelblue', linewidth=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
axes[1].plot([0,1], [0,1], 'k--', linewidth=2, label='Clasidicador Aleatorio')
axes[1].set_xlabel('FPR Tasa de Falsos Positivos')
axes[1].set_ylabel('TPR Tasa de Verdaderos Positivos')
axes[1].set_title('Curva ROC')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.2)
plt.tight_layout()
plt.show()