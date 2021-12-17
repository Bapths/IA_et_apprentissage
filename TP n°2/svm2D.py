import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import svm

#### programme principal à implémenter dans cette fonction ####
def monprogramme(Xapp, Yapp, C):
	"""
		Programme pour les SVM linéaires (lancé avec ESPACE)
		
		Xapp, Yapp : base d'apprentissage générée avec la souris
		C	: paramètre réglé par +/-
	"""
	print("Apprentissage lancé avec " + str(len(Xapp)) + " points et C = ", C)

	# à compléter pour apprendre le modèle SVM...

	# création du modèle
	model = svm.LinearSVC(C=C)

	# apprentissage
	model.fit(Xapp, Yapp)

	# création d'une grille de points de test
	r1 = np.arange(-5,5,0.2)
	Xtest = np.zeros((len(r1)*len(r1),2))
	i = 0
	for x1 in r1:
		for x2 in r1:
			Xtest[i,:] = [x1, x2]
			i += 1
	
	# prédire la catégorie pour tous les points de test...
	Ypred = model.predict(Xtest)
		
	# ... et tracer le résultat avec par exemple 

	# XpredBlue = Xtest[Ypred == 1, :]
	# XpredRed = Xtest[Ypred == 2, :]
	# plt.plot(XpredBlue[:,0], XpredBlue[:,1], 'ob', alpha=0.2)
	# plt.plot(XpredRed[:,0], XpredRed[:,1], 'or', alpha=0.2)
	
	# tracer la droite séparation et les marges... 
	w = model.coef_[0]
	b = model.intercept_

	print(model.coef_)
	print(b)

	plt.plot(r1, (-b-w[0]*r1)/w[1], 'k-', lw=2, alpha=0.2) # Séparation

	plt.plot(r1, (-b-1-w[0]*r1)/w[1], 'k--', lw=2, alpha=0.2) # Marge sup
	plt.plot(r1, (-b+1-w[0]*r1)/w[1], 'k--', lw=2, alpha=0.2) # Marge inf

	# calculer et afficher la marge Delta...	
	delta = (1/w[1])/math.sqrt((w[0]/w[1]+1)**2)
	print(f"delta = {delta}")

	
	# pour réellement mettre à jour le graphique: (à garder en fin de fonction)
	fig.canvas.draw()


def monprogrammeNL(Xapp, Yapp, C, sigma):
	"""
		Programme pour les SVM non linéaires (lancé avec N)
		
		Xapp, Yapp : base d'apprentissage générée avec la souris
		C	: paramètre réglé par +/-
		sigma : paramètre réglé par CTRL +/-
	"""
	print("Apprentissage lancé avec " + str(len(Xapp)) + " points, C = ", C, " et sigma = ", sigma )

	# à compléter pour apprendre le modèle SVM non linéaire...
	
	
	# création d'une grille de points de test
	r1 = np.arange(-5,5,0.2)
	Xtest = np.zeros((len(r1)*len(r1),2))
	i = 0
	for x1 in r1:
		for x2 in r1:
			Xtest[i,:] = [x1, x2]
			i += 1
	
	# création du modèle
	model = svm.SVC(C=C, kernel='rbf', gamma=1/(2*sigma**2))

	# apprentissage
	model.fit(Xapp, Yapp)

	# création d'une grille de points de test
	r1 = np.arange(-5,5,0.2)
	Xtest = np.zeros((len(r1)*len(r1),2))
	i = 0
	for x1 in r1:
		for x2 in r1:
			Xtest[i,:] = [x1, x2]
			i += 1
	
	# prédire la catégorie pour tous les points de test...
	Ypred = model.predict(Xtest)

	g = model.decision_function(Xtest)
		
	# et tracer le résultat avec par girlle de points
	# XpredBlue = Xtest[Ypred == 1, :]
	# XpredRed = Xtest[Ypred == 2, :]
	# plt.plot(XpredBlue[:,0], XpredBlue[:,1], 'ob', alpha=0.2)
	# plt.plot(XpredRed[:,0], XpredRed[:,1], 'or', alpha=0.2)
	
	plt.scatter(Xtest[:,0], Xtest[:,1], cmap='rainbow', c=g, alpha=0.5)

	print(f'model support = {model.n_support_}')
	
	# pour réellement mettre à jour le graphique:
	fig.canvas.draw()

	return Xtest,g

##### Gestion de l'interface graphique ########


Xplot = np.zeros((0,2))
Yplot = np.zeros(0)
plotvariance = 0

C = 1
# C = math.inf
sigma = 1

def onclick(event):
	global Xplot
	global Yplot
	
	
	if plotvariance == 0:
		newX = np.array([[event.xdata,event.ydata]])
	else:
		newX = math.sqrt(plotvariance) * np.random.randn(10, 2) + np.ones((10,1)).dot(np.array([[event.xdata,event.ydata]]))

	print("Ajout de " + str(len(newX)) + " points en (" + str(event.xdata) + ", " + str(event.ydata) + ")")

	Xplot = np.concatenate((Xplot,newX))
	if event.button == 1 and event.key == None:
		plt.plot(newX[:,0], newX[:,1],'.b')
		newY = np.ones(len(newX)) * 1
	elif event.button == 3 and event.key == None:
		plt.plot(newX[:,0], newX[:,1],'.r')
		newY = np.ones(len(newX)) * 2
	Yplot = np.concatenate((Yplot,newY))
	
	fig.canvas.draw()


def onscroll(event):
	global plotvariance
	if event.button == "up":
		plotvariance = round(plotvariance + 0.2, 1)
	elif event.button == "down" and plotvariance > 0.1:
		plotvariance = round(plotvariance - 0.2, 1)
	print("Variance = ", plotvariance)

def onkeypress(event):
	global C
	global sigma
	if event.key == " ":
		monprogramme(Xplot, Yplot, C)
	elif event.key == "n":
		
		# monprogrammeNL(Xplot, Yplot, C, sigma)

		f2, axs = plt.subplots(3, 3)
		C_arr = [0.25,1,32]
		sigma_arr = [0.5,1,2]

		Xblue = Xplot[Yplot == 1, :]
		Xred = Xplot[Yplot == 2, :]
		for i in range(3) :
			for j in range(3) :
				Xtest, g = monprogrammeNL(Xplot, Yplot, C_arr[i], sigma_arr[j])
				axs[i, j].scatter(Xtest[:,0], Xtest[:,1], cmap='rainbow', c=g, alpha=0.5)
				axs[i, j].set_title(f'C = {C_arr[i]}; sigma = {sigma_arr[j]}')
				Xblue = Xplot[Yplot == 1, :]
				axs[i, j].plot(Xblue[:,0], Xblue[:,1], '.b')
				axs[i, j].plot(Xred[:,0], Xred[:,1], '.r')

		for ax in axs.flat :
			ax.label_outer()
			ax.set_xlim([-5, 5])
			ax.set_ylim([-5, 5])

		f2.show()

	elif event.key == "+":
		C *= 2
		print("C = " , C)
	elif event.key == "-":
		C /= 2
		print("C = " , C)
	elif event.key == "ctrl++":
		sigma *= 2
		print("sigma = " , sigma)
	elif event.key == "ctrl+-":
		sigma /= 2
		print("sigma = " , sigma)
				
	
fig = plt.figure()

plt.axis([-5, 5, -5, 5])

cid = fig.canvas.mpl_connect("button_press_event", onclick)
cid2 = fig.canvas.mpl_connect("scroll_event", onscroll)
cid3 = fig.canvas.mpl_connect("key_press_event", onkeypress)

print("Utilisez la souris pour ajouter des points à la base d'apprentissage :")
print(" clic gauche : points bleus")
print(" clic droit : points rouges")
print("\nMolette : +/- variance ")
print("   si variance = 0  => ajout d'un point")
print("   si variance > 0  => ajout de points selon une loi gaussienne")
print("\n ESPACE pour lancer la fonction monprogramme(Xapp,Yapp,C)")
print("    avec la valeur de C modifiée par +/-") 
print("\n N pour lancer la fonction monprogrammeNL(Xapp,Yapp,C,sigma)")
print("    avec la valeur de C modifiée par +/-")
print("    et celle de sigma modifiée par CTRL +/-\n\n") 

plt.show()
