#vamos a desarollar una aplicación para poder detectar y reconstruir rostros
#vamos a ocupar un dataset de 400 fotografias, las cuales vamos a reconocer, y vamos a tomar 40 elementos para poder aplicar una matriz de factorización no negativa, y a este aplicarle una tecnica de reducción de variables.


#x = W * H
#datos = pesos * componentes
# los pesos son las caracteristicas que nosotros buscamos, por ejemplo rasgos faciales, caracteristicas unicas del rostro, etc
# Los componentes son los elementos de clasificación, por ejemplo ojos cafes, azules, nariz chata, alargada, etc

import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_olivetti_faces #este es el dataset de rostros


#caracteristicas de este dataset, tiene 400 rostros, de 40 personas diferentes y necesitamos almenos para este modelo 10 fotos de cada persona, por lo tanto necesitamos un vector de tamaño 4096 cada foto 64*64 pixeles 

faces = fetch_olivetti_faces(shuffle=True, random_state=40)

#el valor de X
X = faces.data #es todo el dataset (toda la matriz del dataset)

#vamos a crear nuestro modelo, para identificar 15 rostros

nmf = NMF(n_components=15, random_state=60)

#tenemos que ajustar el modelo para obtener los pesos 

X_nmf = nmf.fit_transform(X)

fig, axes = plt.subplots(3,5, figsize=(12,8))

#para pintarla vamos a recorrer cada subgrafia, e ir dibujando sus componentes
for i, ax in enumerate(axes.ravel()):
    #ravel() lo que hace es convertir en una cuadricula la imagen a dibujar
    #de cada componente i va a tener un vector de tamaño 4096
    #y lo vamos a redimensionar a 64X64
    ax.imshow(nmf.components_[i].reshape(64,64), cmap='gray')
    #ocultamos los ejes para visualizarlo
    ax.axis('off')

plt.suptitle('Componentes de NMF de Rostros')
plt.show()