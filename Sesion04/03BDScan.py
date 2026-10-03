#BDScan define el radio de vecindad de cada punto llamado epsilon, (eps) si dos puntos estan a distancias <= eps con vecino con al menos min_samples, vecinos dentros del eps es el nucleo y los nucleos y sus vecinos forman clusters, los puntos sin ningun nucleo cercano son etiquetados como -1 (ruido)
# 
# Los efectos epsilon pueden variar conforme al dataset
# eps es muy pequeño en el radio y es estrecho, pocos vecinos y muchos muchos pueden ser ruido
# si es eps es adecuado significa que el radio es justo, detecta correctamente las regiones densas, y el numero de clusters es coherente a la estructura
# si eps es grande, radio es amplio, casi todos los puntos son vecinos entre si por lo tanto bdscan lo fusiona
# 
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

#vamos a realizar un ejemplo en el cual vamos a tomar dos medias lunas de forma no convexa para verificar datos que sean realistas

X, _ = make_moons(n_samples=500, noise=0.1, random_state=40)

#tenemos que medir las distancias euclidianas de los parametros que vamos a obtener, para ello tenemos que aplicarlo por medio de Standar

X = StandardScaler().fit_transform(X)

#vamos a configurar los valores de eps, con el mimos de min_samples para aislar los efectos
configuraciones = [
    {'eps':0.05, 'Titulo': 'eps=0.05 (muy pequeño)\n radio estrecho'},
    {'eps':0.20, 'Titulo': 'eps=0.20 (adecuado)\n detecta las 2 lunas de forma correcta'},
    {'eps':0.80, 'Titulo': 'eps=0.80 (muy grande)\n todo lo fusiona'}
]

#vamos a entrenarlo
fig, axes = plt.subplots(1,3, figsize=(16,5))
fig.suptitle('BDSCAN - Efecto del parametro de eps, sobre la densidad de los datos ', fontsize=13, fontweight='bold')

for ax, cfg, in zip(axes, configuraciones):
    db = DBSCAN(eps=cfg['eps'], min_samples=5)
    labels = db.fit_predict(X)

    #las metricas
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = (labels == -1).sum()

    print(f"eps = {cfg['eps']:.2f} => Clusters: {n_clusters} |" f"Ruido: {n_ruido}({n_ruido/len(X)*100:.1f}%)")

    #vamos a colorear los clusters
    mascara_validos = labels >= 0
    ax.scatter(
        X[mascara_validos, 0], X[mascara_validos, 1],
        c=labels[mascara_validos],
        cmap='tab10', alpha=0.7, s=15, label=f'{n_clusters} clusters'
    )

    #vamos a colorear el ruido
    mascara_ruidos = labels == -1
    if mascara_ruidos.any():
        ax.scatter(
            X[mascara_ruidos, 0], X[mascara_ruidos, 1],
            c='black', alpha=0.6, s=25, marker='x', label=f'{n_ruido} ruido total'
        )

    ax.set_title(cfg['Titulo'], fontsize=10)
    ax.set_xlabel('Caracteristicas 1')
    ax.set_ylabel('Caracteristicas 2')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
