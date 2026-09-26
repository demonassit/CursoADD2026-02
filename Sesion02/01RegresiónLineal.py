#Supongamos que necesitamos desarrollar una aplicación para poder predecir los precios de una vivienda, derivadoa que tenemos un dataset que nos presenta sus caracteristicas de todo un lote enorme de viviendas


from sklearn.datasets import fetch_california_housing

from sklearn.linear_model import LinearRegression, Ridge, Lasso

#ridge nos sirve para evitar que los coeficientes crezcan demasaido, (overfitting) eso pasa cuando tenemos demasiadas variables

#lasso maneja los valores absolutos, del calculo de los coeficientes que se vayan a 0, esto hace que su seleccion sean automatica para cada variable

#vamos a necesitar secciones definidas para el entrenamiento, y aplicar metricas

from sklearn.model_selection import train_test_split
#para el calculo de los minumos cuadrados MSE
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt

housing = fetch_california_housing()

X, y = housing.data, housing.target

#entrenamiento

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#necesitamos crear una función que se encargue de generar los gafricos de la predicción

def generar_grafico(y_real, y_pred, nombre, rmse, r2, ax_scatter, ax_error):
    #vamos a generar 2 graficos para evaluar el modelo
    # 1.- Dispersión de los valores reales vs predictivos, comprar sus representaciones mas cercanas 
    # 2.- Distribución para los errores, dentro de la prediccion con el fin de evitar algo que este fuera de los parametros

    ax_scatter.scatter(
        y_real, y_pred,
        alpha = 0.3, #esto es para poder transparentar las zonas de mayor densidad o las que mas colisionan
        s = 10 #para no saturar la grafica de dispersión
    )

    #necesitamos los valores de las condiciones de la predición
    minimo = min(y_real.min(), y_pred.min())
    maximo = max(y_real.max(), y_pred.max())

    ax_scatter.plot(
        [minimo, maximo], [minimo, maximo], 'r--', linewidth=1.5, label='Predicción'
    )

    ax_scatter.set_xlabel('Precio Real en 100K dolares')
    ax_scatter.set_ylabel('Precio Predicho a los 100K')
    ax_scatter.set_title('f{nombre}\n RMSE = {rmse: .4f} R2 = {r2: .4f}')
    ax_scatter.legend(fontsize=8)

    #histograma
    residuos = y_real - y_pred

    ax_error.hist(
        residuos,
        bins=50, #tamaño de la grafica de barra
        color = 'salmon',
        edgecolor = 'white',
        linewidth = 0.4
    )

    ax_error.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Error = 0')
    ax_error.set_xlabel('Error real - preditivo')
    ax_error.set_ylabel('Frecuencia')
    ax_error.set_title('Distribución de Errores')
    ax_error.legend(fontsize=0.8)

#ahora vamos a ver la comparativa entre utilizar RL simple, o aplicarlo con Ridge o aplicarlo con Lasso

fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(14,12)
)

fig.suptitle(
    'Comparación de los modelos de regresión para el caso de casitas',
    fontsize=14, fontweight='bold'
)

#empezamos con el modelo a partir de una matriz para cada elemento
for fila, (nombre, modelo) in enumerate([
    ('Lineal', LinearRegression()),
    ('Ridge', Ridge(alpha=0.8)),
    ('Lasso', Lasso(alpha=0.1))
]):
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    #el calculo de la proporcion de la varianza por el modelo de reduccion de coeficientes por el valor absoluto
    r2 = r2_score(y_test, y_pred)

    print(f"{nombre}: RMSE = {rmse: .4f}, R2 = {r2: .4f}")

    generar_grafico(
        y_test, y_pred, nombre, rmse, r2, 
        ax_scatter=axes[fila, 0],
        ax_error=axes[fila, 1]
    )

#todo lo metemos en ajuste para el grafico
plt.tight_layout()
plt.show()
