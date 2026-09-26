import numpy as np
import matplotlib.pyplot as plt

#vamos a utilizar la libreria sklearn para poder tomar elementos para un ejemplo con PCA

from sklearn.preprocessing import StandardScaler
#sirve para establecer las metricas 

from sklearn.datasets import load_breast_cancer
#este es un dataset de imagenes de tumores

#es desarrollar una aplicación la cual nos sirva para poder determinar de una imagen radiografica si es cancer maligno o benigno

#siempre lo primero que debemos de hacer es cargar el dataset
data = load_breast_cancer() #este tiene todas las dimensiones

X = data.data #vamos a traer todos los datos en una matriz con todas sus caracteristicas 30 variables
y = data.target #es la etiqueta para identificar los elementos de la clase

#tenemos que reducirlas para eso ocupamos StandarScaler

scaler = StandardScaler() #define las metricas de estadarización

#definimos el modelo para que aprenda 
X_scaler = scaler.fit_transform(X)

#aplicar algebra lineal por parte cel calculo de SVD
#V = vectores singulares izquierdos de la matriz de 3 variables
#S = valores singulares que se van a obtener referente de la magnitud de cada componente
# D (Vh) = los vectores singulares derechos a partir de la matriz transpuesta

V, S, Vh = np.linalg.svd(X_scaler, full_matrices=False)

#el resultado de esa cosa son PC1 PC2 
pc1 = Vh[0] #nuestro primer componente principal (maligno)
pc2 = Vh[1] #el valor ortogonal respecto de pc1  (benigno)

W = Vh[:2].T #aqui estamos extrayendo los componentes de las raices factorizadas de la matriz transpuesta 

X_nueva = X_scaler.dot(W) #reducir sus dimensiones

#aplicando varianza
varianza_total = (S**2).sum()
varianza_explicada = (S[:2]**2)/varianza_total

print(f"Varianza explicada por cada componente: {varianza_explicada}")
print(f"Varianza total retenida: {varianza_total}")

plt.figure(figsize=(8,6))

plt.scatter(
    X_nueva[:,0], 
    X_nueva[:,1],
    c = y, #es para el color segun su clase [0] maligno [1] benigno
    cmap = 'coolwarm',
    alpha = 0.7
)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA con Numpy para detección de Cancer')
plt.colorbar(label='Clase')
plt.show()



