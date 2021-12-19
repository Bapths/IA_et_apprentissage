import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

images = ['china.jpg', 'flower.jpg']
#images = ['flower.jpg']

K = 50

# tests sur les 2 images
for image in images:
	img = load_sample_image(image)

	(width,height,channels) = img.shape

	# Création de la matrice des pixels
	X = img.reshape(width*height, channels)

	# création du clustering
	clustering = KMeans(n_clusters=K, init='random', n_init=100)
	clustering.fit(X)
	
	# récupération des étiquettes sous forme matricielle : 
	Y = clustering.labels_.reshape(width, height)
	
	
	# Création de l'image couleur à 3 canaux 
	segmentation = np.zeros((width,height, channels))  # taille identique à img
	
	# à compléter pour remplir le tableau à 3 dimensions 'segmentation'
	# avec les couleurs des centres 
	centers = clustering.cluster_centers_
	segmentation = centers[Y]


	# afficher l'originale, les étiquettes Y et la segmentation avec la couleur moyenne :
	# plt.figure()
	# plt.imshow(img)
	# plt.draw()
	# plt.figure()
	# plt.imshow(Y)
	# plt.draw()
	plt.figure()
	plt.imshow(segmentation / 255)
	# les centres sont à valeurs réelles, 
	# donc imshow s'attend à avoir des valeurs entre 0 et 1 pour segmentation
	# au lieu de 0 et 255 pour les entiers
	plt.draw()
	
plt.show()
		
