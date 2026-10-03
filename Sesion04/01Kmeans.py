#k means es un algoritmo que se encarga de identificar patrones a partir de un centroide, para el aprendizaje no supervisado se orienta a partir del uso de clustering (agrupamiento), recordando que a diferencia de las anteriores veces aqui no existen etiquetas (conocimiento previo)

# 1.- elelgir k puntos aleatorios como centroides
# 2.- asignar a cada punto el centroida mas cercano y calcular de forma euclidiana
# 3.- recalculamos el centroide de cad grupo como el promedio de los puntos
# 4.- repetimos los pasos 2 y 3 hasta que los centroides no se muevan

#implementar los metodos del codo y de silueta
#codo (suma de las distancias al cuadrado de cada centroide), la distancia mas corta a medida que k crece la inercia baja. 
#silueta mide que tan bien definido esta cada cluster, compara la distancia media de un puntos a los demas de su propio cluster (cohesión) contra la distancia media al cluster vecino mas cercano

#para este ejercicio vamos a generar muchos datos artificiales, con 4 grupos conocidos para aplicar k means, para los valores de k entre 2 y 9

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

#make_blobs es una dataset de puntos aleatorios, con este vamos a generar 500 muestras, 4 centroides y una semilla

X, _ = make_blobs(n_samples=500, centers=4, random_state=40)

#necesitamos evaluar las distancias 
inercias = []
siluetas = []

#primero tenemos que crear el modelo, segundo tenemos que agruparlo, y apartir de la semilla organizar los grupos, y esto se tiene que ejecutar un numero de veces tal que kmeans se ejecute hacia el calculo de sus centroides

k_range = range(2,10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=40, n_init=10)

    #vamos a entrenar el modelo a partir de devolver las etiquetas encontradas
    labels = km.fit_predict(X)

    inercias.append(km.inertia_)

    siluetas.append(silhouette_score(X, labels))

#graficamos
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))

ax1.plot(k_range, inercias, 'bo-', linewidth=2, markersize=7)
ax1.set_title('Metodo del Codo')
ax1.set_xlabel('Numero de Clusters')
ax1.set_ylabel('Inercia (suma de las distacnias)')

ax2.plot(k_range, siluetas, 'rs-', linewidth=2, markersize=7)

ax2.set_title('Coeficientes de Silueta')
ax2.set_xlabel('Numero de Clusters')
ax2.set_ylabel('Silueta Promedio')

plt.suptitle('Seleccion del numero optimo de clusters: ', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
