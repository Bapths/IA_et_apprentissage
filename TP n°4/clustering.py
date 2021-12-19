import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import KMeans, SpectralClustering


#### programme principal à implémenter dans cette fonction ####
def monprogramme(X, K):
	"""
		X : base d'apprentissage générée avec la souris
		K : paramètre réglé par +/-
	"""
	print("Kmeans clustering lancé avec " + str(len(X)) + " points et K = ", K)

	# création du clustering
	clustering = KMeans(n_clusters=K, init='random', n_init=100)
	clustering.fit(X)
	Y = clustering.labels_
	centers = clustering.cluster_centers_
	
	# ... et tracer le résultat en utilisant une boucle sur les groupes et 
	#	 '.' + couleurs[k]  
	#	pour le type de point du groupe k dans la fonction plot

	couleurs = ['b','r','g','y','m','c','k']

	plt.clf()
	plt.axis([-5, 5, -5, 5])

	for k in range(K) :
		Xk = X[Y == k, :]
		plt.plot(Xk[:, 0], Xk[:, 1], f'.{couleurs[k]}', alpha=0.2)
		plt.plot(centers[k][0], centers[k][1], f'o{couleurs[k]}')
	
	# pour réellement mettre à jour le graphique: 
	fig.canvas.draw()
	

### programme pour Spectral clustering ###
def monprogrammeSpectralClustering(X, K, sigma):
	"""
		X : base d'apprentissage générée avec la souris
		K : paramètre réglé par +/-
		sigma : paramètre réglé par ctrl +/-
	"""
	print("Spectral Clustering lancé avec " + str(len(X)) + " points et sigma = ", sigma)

	# création du clustering
	gamma = 1/(2*sigma**2)

	clustering = SpectralClustering(n_clusters=K, gamma=gamma)
	clustering.fit(X)
	Y = clustering.labels_
	A = clustering.affinity_matrix_
	
	# ... et tracer le résultat 
	plt.figure(1)
	couleurs = ['b','r','g','y','m','c','k']

	plt.clf()
	plt.axis([-5, 5, -5, 5])

	for k in range(K) :
		Xk = X[Y == k, :]
		plt.plot(Xk[:, 0], Xk[:, 1], f'.{couleurs[k]}')
	
	# pour mettre à jour le premier graphique: 
	fig.canvas.draw()
	
	# création de la seconde figure : 
	fig2=plt.figure(2)
	plt.clf()	# pour effacer le résultat précédent
	plt.imshow(A, cmap='gray')
	
	plt.show()	# afficher la figure

def kernel(X1,X2,sigma):
	"""
		Retourne la matrice de noyau K telle que K_ij = K(X1[i], X2[j])
		avec un noyau gaussien K(x,x') = exp(-||x-x'||^2 / 2sigma^2)
	"""	
	m1 = X1.shape[0]
	m2 = X2.shape[0]
	K = np.zeros((m1,m2))
	for i in range(m1):
		for j in range(m2):
			K[i,j] = math.exp(- np.linalg.norm(X1[i] - X2[j])**2 / (2*sigma**2))
	return K

def monprogrammeSpectralClusteringV2(X, K, sigma):
	"""
		X : base d'apprentissage générée avec la souris
		K : paramètre réglé par +/-
		sigma : paramètre réglé par ctrl +/-
	"""
	print("Spectral Clustering lancé avec " + str(len(X)) + " points et sigma = ", sigma)

	# création du clustering
	
	K = kernel(X, X, sigma)
	A = K + np.eye(len(K))
	A = A - np.diag(np.diag(A))
	D = np.sum(A, axis=1)
	D_inv_sqrt = np.sqrt(np.linalg.inv(D))
	print(D)
	# beta = np.linalg.solve(A, Y)
	
	
	# ... et tracer le résultat 
	plt.figure(1)
	couleurs = ['b','r','g','y','m','c','k']

	plt.clf()
	plt.axis([-5, 5, -5, 5])

	# for k in range(K) :
	# 	Xk = X[Y == k, :]
	# 	plt.plot(Xk[:, 0], Xk[:, 1], f'.{couleurs[k]}')
	
	# pour mettre à jour le premier graphique: 
	fig.canvas.draw()
	
	# création de la seconde figure : 
	fig2=plt.figure(2)
	plt.clf()	# pour effacer le résultat précédent
	plt.imshow(A, cmap='gray')
	
	plt.show()	# afficher la figure
	

##### Gestion de l'interface graphique ########


Xplot = np.zeros((0,2))
plotvariance = 0

K = 2
sigma = 1

def onclick(event):
	global Xplot
	
	if plotvariance == 0:
		newX = np.array([[event.xdata,event.ydata]])
	else:
		newX = math.sqrt(plotvariance) * np.random.randn(10, 2) + np.ones((10,1)).dot(np.array([[event.xdata,event.ydata]]))

	print("Ajout de " + str(len(newX)) + " points en (" + str(event.xdata) + ", " + str(event.ydata) + ")")

	Xplot = np.concatenate((Xplot,newX))
	if event.button == 1 and event.key == None:
		plt.plot(newX[:,0], newX[:,1],'.k')
	
	fig.canvas.draw()


def onscroll(event):
	global plotvariance
	if event.button == "up":
		plotvariance = round(plotvariance + 0.2, 1)
	elif event.button == "down" and plotvariance > 0.1:
		plotvariance = round(plotvariance - 0.2, 1)
	print("Variance = ", plotvariance)

def onkeypress(event):
	global K
	global sigma
	global Xplot
	if event.key == " ":
		monprogramme(Xplot, K)
	elif event.key == "c":
		monprogrammeSpectralClustering(Xplot,K,sigma)
	elif event.key == "+" and K < len(Xplot):
		K += 1
		print("K = " , K)
	elif event.key == "-" and K > 1:
		K -= 1
		print("K = " , K)
	elif event.key == "ctrl++":
		sigma *= 2
		print("sigma = " , sigma)
	elif event.key == "ctrl+-":
		sigma /= 2
		print("sigma = " , sigma)
	elif event.key == 'delete':
		Xplot = np.zeros((0,2))	
		plt.clf()
		plt.axis([-5, 5, -5, 5])
		fig.canvas.draw()
	
fig = plt.figure()

plt.axis([-5, 5, -5, 5])

cid = fig.canvas.mpl_connect("button_press_event", onclick)
cid2 = fig.canvas.mpl_connect("scroll_event", onscroll)
cid3 = fig.canvas.mpl_connect("key_press_event", onkeypress)

print("Utilisez la souris pour ajouter des points à la base d'apprentissage :")
print(" clic gauche : points ")
print("\nMolette : +/- variance ")
print("   si variance = 0  => ajout d'un point")
print("   si variance > 0  => ajout de points selon une loi gaussienne")
print("\n ESPACE pour lancer la fonction monprogramme(X,K)")
print("    avec la valeur de K modifiée par +/-\n\n") 
print("\n C pour lancer la fonction monprogrammeSpectralClustering(X,K,sigma)")
print("    avec la valeur de sigma modifiée par Ctrl +/-\n\n") 

plt.show()
