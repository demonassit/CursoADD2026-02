#Para este ultimo ejercicio vamos a comprar 3 formas de combinación de modelos, sobre el mismo dataset, 
# Bagging (RF) entrena muchos arboles en paralelo y cada uno con una muestra aleatoria.
# Boosting (Adaboost) Entrrenar varios arboles pero simples en secuencia, cada arbol nuevo le da mas peso a los ejemplos que el anterior que clasifico mal, asi va corrigiendo los errores.
# Stacking combina modelos diferentes (RF, KNN, SVM) sus predicciones se convierten en las entradas de un modelo lineal


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, classification_report

import numpy as np
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=50, stratify=y)

#modelos
arbol = DecisionTreeClassifier(random_state=50)

#bagging 
rf = RandomForestClassifier(n_estimators=200, random_state=50)

ada = AdaBoostClassifier(
    estimator = DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5,
    random_state=50
)

#necesitamos de los 3 modelos crear un metamodelo, para medir distancias para su escalado
modelos_base = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=50)),
    ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
    ('svm', make_pipeline(StandardScaler(), SVC(probability=True, random_state=50)))
]

#necesitamos las predicciones con cv (validacion cruzada)
stacking = StackingClassifier(
    estimators=modelos_base,
    final_estimator=LogisticRegression(),
    cv=5
)

modelos = {
    'Arbol de decision': arbol,
    'Random Forest': rf,
    'AdaBoost': ada,
    'Stacking': stacking
}

#vamos a realizar el entrenamiento
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)

resultados = {}

print(f"{'Modelo':<26}{'CV accuracy':>18}{'Test acc':>10}{'F1':>8}{'AUC':>8}")


for nombre, modelo in modelos.items():
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    resultados[nombre] = {
        'cv_media': cv_scores.mean(),
        'cv_std' : cv_scores.std(),
        'fpr' : fpr,
        'tpr': tpr,
        'auc' : auc(fpr, tpr)
    } 

    print(f"{nombre:<26}{cv_scores.mean():>10.4f} +/- {cv_scores.std():.4f}"
          f"{accuracy_score(y_test, y_pred):>10.4f}{f1_score(y_test, y_pred):>8.4f}{resultados[nombre]['auc']:>8.4f}")

print("\n Reporte Clasificador")
print(classification_report(y_test, stacking.predict(X_test), target_names=['Maligno','Benigno']))

for (nombre, _), peso in zip(modelos_base, stacking.final_estimator_.coef_[0]):
    print(f"   {nombre:<5} {peso:>6.2f}")

#Adaboost permite ver como mejora conforme agrega cada segmento
acc_por_paso = [accuracy_score(y_test, y_p) for y_p in ada.staged_predict(X_test)]

#graficas
colores = ['#2a78d6', '#eb6384', '#1bafa8', '#eda100']

nombres = list(modelos.keys())

fig, axes = plt.subplots(1,3, figsize=(18, 5.5))
fig.suptitle('Ensambles sobre Predictor de Cancer con Bagging VS Boosting VS Stacking', fontsize=12, fontweight='bold')

#primera grafica promedio de la validacion cruzada
medias = [resultados[n]['cv_media'] for n in nombres]
stds = [resultados[n]['cv_std'] for n in nombres]

axes[0].barh(nombres, medias, xerr=stds, color=colores, capsize=4)
for i, m in enumerate(medias):
    axes[0].text(0.855, i, f'{m:3f}', va='center', ha='left', color='white', fontweight='bold')

axes[0].set_xlim(0.85, 1.02)
axes[0].invert_yaxis()
axes[0].set_xlabel('Accuracy (Validación Cruzada, 5 partes)')
axes[0].set_ylabel('Desempeño Promedio y Variación')
axes[0].grid(axis='x', alpha=0.2)

#curvas roc
for nombre, color in zip(nombres, colores):
    r = resultados[nombre]
    axes[1].plot(r['fpr'], r['tpr'], color=color, linewidth=2, label=f"{nombre}(AUC={r['auc']:.3f})")

axes[1].set_xlabel('FPR')
axes[1].set_ylabel('TPR')
axes[1].set_title('Curva ROC')
axes[1].legend(loc='lower right', fontsize=8)
axes[1].grid(alpha=0.2)

#adaboost aprendiendo paso a paso
axes[2].plot(range(1, len(acc_por_paso) + 1), acc_por_paso, color=colores[2], linewidth=2)
axes[2].set_xlabel('Numero de arboles de profundidad')
axes[2].set_ylabel('Accuracy en prueba')
axes[2].set_title('Adaboost de cada arbol que va corrigiendo al anterior')
axes[2].grid(alpha=0.2)

plt.tight_layout()
plt.show()

