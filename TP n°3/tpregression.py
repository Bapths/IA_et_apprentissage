import numpy as np
import matplotlib.pyplot as plt
import math

from numpy.core.numeric import identity
from sklearn.kernel_ridge import KernelRidge
from sklearn import neighbors

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

def krrapp(X,Y,Lambda,sigma):
	"""
		Retourne le vecteur beta du modèle Kernel ridge regression
		à noyau gaussien
		à partir d'une base d'apprentissage X,Y 
	"""

	K = kernel(X, X, sigma)
	A = K + Lambda*np.eye(len(K))
	beta = np.linalg.solve(A, Y)
	
	return beta
	
def krrpred(Xtest,Xapp,beta,sigma):
	""" 
		Retourne le vecteur des prédictions du modèle
		KRR à noyau gaussien de paramètres beta et sigma
	"""
	Ktest = kernel(Xtest, Xapp, sigma)
	ypred = Ktest@beta
	
	return ypred

def skl_krr_app_pred(Xapp, Yapp, Xtest, Lambda, sigma):
	""" 
		Retourne le vecteur des prédictions du modèle
		KRR à noyau gaussien de paramètres beta et sigma
		(bibliothèque sklearn)
	"""
	model = KernelRidge(alpha=Lambda, kernel='rbf', gamma=1/(2*sigma*sigma))
	model.fit(Xapp.reshape(-1, 1), Yapp)
	ypred = model.predict(Xtest.reshape(-1, 1))
	
	return ypred
	

def kppvreg(Xtest, Xapp, Yapp, K):

	n = Xtest.shape[0]  # nb de points de test
	m = Xapp.shape[0]   # nb de points d'apprentissage
	ypred = np.zeros(n)

	for i in range(n) :
		distances_dict = {Xapp[j]:abs(Xtest[i]-Xapp[j]) for j in range(m)}
		kppv_index = []
		for k in range(K) :
			ppv = min(distances_dict, key=distances_dict.get) # key du dict avec la plus petite value
			kppv_index.append(np.where(Xapp == ppv)[0][0])
			distances_dict.pop(ppv, None)
		kppv_pred = Yapp[kppv_index]
		ypred[i] = np.mean(kppv_pred)
			
	return ypred

#################################################
#### Programme principal ########################
#################################################

# 1) générer une base de données de 1000 points X,Y

m = 1000
X = 6 * np.random.rand(m) - 3
Y = np.sinc(X) + 0.2 * np.random.randn(m)


# 2) Créer un base d'apprentissage (Xapp, Yapp) de 30 points parmi ceux de (X,Y) et une base de test(Xtest,Ytest) avec le reste des données

indexes = np.random.permutation(m)  # permutation aléatoire des 1000 indices entre 0 et 1000 
indexes_app = indexes[:30]  # 30 premiers indices
indexes_test = indexes[30:] # le reste

Xapp = X[indexes_app]
Yapp = Y[indexes_app]

Xtest = X[indexes_test]
Ytest = Y[indexes_test]

# ordronner les Xtest pour faciliter le tracé des courbes
idx = np.argsort(Xtest)
Xtest = Xtest[idx]
Ytest = Ytest[idx]

# tracer la figure

plt.figure()
#plt.plot(Xtest,Ytest,'ob', alpha=0.1)
plt.plot(Xapp,Yapp,'*b')
plt.plot(Xtest,np.sinc(Xtest) , '--m')
#plt.legend(['base test', 'base app', 'f_reg(x)'] )


### Tests de la Kernel ridge regression...

Lambda = 0.6 # peut être très grand
sigma = 0.6 # variation entre -3 et 6, eviter donc sigma > 5 ou 6

# =========[ PARTIE KRR ]=========
f2, axs = plt.subplots(5, 5)

Lambda_arr = [0.01, 0.1, 0.5, 1, 10]
sigma_arr = [0.1, 0.2, 0.5, 2, 5]

for i in range(5) :
	for j in range(5) :
		beta = krrapp(Xapp, Yapp, Lambda, sigma)
		Ypred_app = krrpred(Xapp, Xapp, beta, sigma)
		Ypred_test = krrpred(Xtest, Xapp, beta, sigma)

		# Ypred_test_skl = skl_krr_app_pred(Xapp, Yapp, Xtest, Lambda, sigma) # Même chose mais avec le modèle sklearn
		# Ypred_app_skl = skl_krr_app_pred(Xapp, Yapp, Xapp, Lambda, sigma) # Même chose mais avec le modèle sklearn

		axs[i][j].plot(Xapp,Yapp,'*b')
		axs[i][j].plot(Xtest,np.sinc(Xtest) , '--m')
		axs[i][j].plot(Xtest,Ypred_test,'b')

		# axs[i][j].plot(Xtest,Ypred_test_skl,'r') # Affiche la courbe obtenue par modèle sklearn (en rouge)

		err_app = np.mean((Ypred_app - Yapp)**2)
		err_test = np.mean((Ypred_test - Ytest)**2)
		axs[i][j].set_xlabel(f'Sigma={sigma}')
		axs[i][j].set_ylabel(f'Lambda={Lambda}')
		axs[i][j].set_title(f'A={round(err_app*100,2)}%, T={round(err_test*100,2)}%')

for ax in axs.flat :
	ax.label_outer()
	ax.set_xlim([-3, 3])
	ax.set_ylim([-0.5, 1.5])

f2.show()

# =========[ PARTIE KPPV ]=========
f3, axs = plt.subplots(1, 5)

K_arr = [1, 2, 5, 10, 20]

for i in range(5) :
	Ypred_app = kppvreg(Xapp, Xapp, Yapp, K_arr[i])
	Ypred_test = kppvreg(Xtest, Xapp, Yapp, K_arr[i])

	# Avec sklearn
	model = neighbors.KNeighborsRegressor(K_arr[i])
	model.fit(Xapp.reshape(-1,1), Yapp)
	Ypred_test_sklearn = model.predict(Xtest.reshape(-1,1))

	axs[i].plot(Xapp,Yapp,'*b')
	axs[i].plot(Xtest,np.sinc(Xtest) , '--m')

	axs[i].plot(Xtest,Ypred_test,'b')
	axs[i].plot(Xtest,Ypred_test_sklearn,'r')

	err_app = np.mean((Ypred_app - Yapp)**2)
	err_test = np.mean((Ypred_test - Ytest)**2)
	axs[i].set_xlabel(f'K={K_arr[i]}')
	axs[i].set_title(f'A={round(err_app*100,2)}%, T={round(err_test*100,2)}%')

	for ax in axs.flat :
		ax.label_outer()
		ax.set_xlim([-3, 3])
		ax.set_ylim([-0.5, 1.5])

f3.show()

plt.show() # affiche les plots et bloque en attendant la fermeture de la fenêtre

