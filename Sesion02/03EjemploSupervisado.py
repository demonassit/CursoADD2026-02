#vamos a desarrollar un modelo supersivado entre kvecinos con regresión lineal, la idea es predecir que tanto va a poder avanzar la diabetes en un paciente un años despues, a partir de 10 medidas clinicas, (edad, sexo, indice de masa corporal, presion, etc)

from sklearn.datasets import load_diabetes
#tiene 442 pacientes con 10 caracteristicas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target # y = progresion de la enfermedad, valor entre 25 y 346

print('Caracteristicas: ', diabetes.feature_names)
print('Forma de X ', X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.7, random_state=40
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)

#con k debemos establecer un numero de vecinos y de profundidad que queremos para el arbol, ojo k no debe de ser muy grande
valores_k = range(1,31)

r2_por_k = []

for k in valores_k:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    r2_por_k.append(r2_score(y_test, knn.predict(X_test_s)))

mejor_k = valores_k[int(np.argmax(r2_por_k))]
print(f"\n Mejor k encontrado: {mejor_k}")

#vamos a entrenar
modelo_knn = KNeighborsRegressor(n_neighbors=mejor_k)
modelo_knn.fit(X_train_s, y_train)
y_pred_knn = modelo_knn.predict(X_test_s)

#lineal
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train_s, y_train)
y_pred_lineal = modelo_lineal.predict(X_test_s)

#tenemos que evaluarlo con MSE y R2
resultados = {
    f"KNN (k={mejor_k})" : y_pred_knn,
    "Regresión Lineal" : y_pred_lineal
}

print("\n Modelo MSE y R2")
for nombre, y_pred in resultados.items():
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{nombre:<20}   {mse:>9.2f}   {r2:>6.3f}")

#la regresión lineal nos va a decir cuanto pesa cada variable
print("\n Coeficientes de la regresión lineal")
for nombre, coef in zip(diabetes.feature_names, modelo_lineal.coef_):
    print(f"  {nombre:<5} {coef:>8.2f}")


fig, ejes = plt.subplots(1,3, figsize=(16,5))

ejes[0].plot(valores_k, r2_por_k, marker="o")
ejes[0].axvline(mejor_k, color='red', linestyle='--', label=f"mejor k = {mejor_k}")
ejes[0].set_title("KNN R2 segun el numero de vecinos")
ejes[0].set_xlabel("k")
ejes[0].set_ylabel("R2 de prueba")
ejes[0].legend()

#grafica de los elementos reales vs predicción
for eje, (nombre, y_pred) in zip(ejes[1:], resultados.items()):
    eje.scatter(y_test, y_pred, alpha=0.6)
    eje.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle='--', label='Prediccion Perfecta')
    eje.set_title(f"{nombre} (R2 = {r2_score(y_test, y_pred):.3f})")
    eje.set_xlabel("Valor Real")
    eje.set_ylabel("Valos Predicho")
    eje.legend()

plt.tight_layout()
plt.show()