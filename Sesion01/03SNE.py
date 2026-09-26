#SNE es una tecnica de reduccion de dimensiones especialmente diseñada para visualizar datos de altas dimensiones 2d o 3d
# a diferencia de PCA (buscar direccion maxima de la viarianza) se enfoca en preservar las relaciones vectoriales locales, osea los puntos mas cercanos ortogonales (sus vecinos mas cercanos)

from sklearn.manifold import TSNE
#vamos a ocupar un dataset de 1797 imagenes con digitos 
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt


digitos = load_digits()
X, y = digitos.data, digitos.target

#si son digitos son representaciones en 2d, entonces a partir de ello tenemos que calcularla distancia entre un punto respecto de otro punto si esta entre 5 y 50 

tsne = TSNE(n_components=2, perplexity=80, random_state=40, max_iter=1000)

X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10,8))

scatter = plt.scatter(
    X_tsne[:,0], #coordenas 1 en el espacio TSNE
    X_tsne[:,1], #coordenadas 2 en el mismo espacio
    c = y,
    cmap = 'tab10',
    alpha=0.8
)

plt.colorbar(scatter, label='Digito')
plt.title('Aplicación de TSNE para Clasificar Digitos')
plt.show()