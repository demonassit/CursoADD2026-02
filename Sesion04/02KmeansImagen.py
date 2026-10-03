#Con base al codigo anterior donde se explica codo y silueta, vamos a crear un ejemplo para buscar en una imagen, colores y tonalidades, y apartir de ello hacer que se agrupen pixeles para reconstruir dicha imagen rn RGB

from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt


#todas las imagenes a color es una muestra en 3d segmentadas en patrones de RGB con el uso de matrices para cada color, R, G, B lo que hara kmeans es agrupar los pixeles y debe de identificarlos por su similitud del color, para eso cada cluster va a representar un color promedio, y cada pixel debe de ser reemplazado por su centroide

img = load_sample_image('flower.jpg')

alto, ancho, canal = img.shape

print(f"Froma original de la imagen: {img.shape}")
print(f"Pixeles totales: {alto*ancho}")

#debemos dar un preprocesamienot para la normalizacion de los kmeans, dentro del rango de las 3 matrices, entendiendo que sus combinaciones de colores estan comprendidas entre 0 a 255 

img_normalizada = img/255.0

#tenemos que aplanarla altoXancho para obtener cada matriz de rgb
pixeles = img_normalizada.reshape(-1,3)

print(f"Formar tras aplanar {pixeles.shape}")

#vamos a segmentar con kmeans los 3 valores para k 

valore_k = [8,16, 32]

imagenes_segmentadas = []

for k in valore_k:
    km = KMeans(n_clusters=k, n_init=10, random_state=40)
    km.fit(pixeles)

    #a partir del centroide de su clsuter tenemos que ir reemplazando cada pixel, para que encuentre su valor
    colores_centroides = km.cluster_centers_
    pixeles_segmentados = colores_centroides[km.labels_]

    #ahora reconstruimos la imagen
    img_seg = (pixeles_segmentados.reshape(alto, ancho, 3)*255).astype(np.uint8)
    imagenes_segmentadas.append(img_seg)

    print(f"k = {k:>2} segmentación completa")

fig, axes = plt.subplots(1,4, figsize=(18,5))
fig.suptitle('Segmentacion de imagenes con Kmeans\n cada pixel se reemplaza por el color de su centroide', fontsize=13, fontweight='bold')

#imprimimos la imagen original
axes[0].imshow(img)
axes[0].set_title('Imagen Original')
axes[0].axis('off')

#imprimirmos la imagen segmentada o reconstruida
titulos = [f'k = {k}\n{k} colores unicos' for k in valore_k]

for ax, img_seg, titulo in zip(axes[1:], imagenes_segmentadas, titulos):
    ax.imshow(img_seg)
    ax.set_title(titulo)
    ax.axis('off')

plt.tight_layout()
plt.show()