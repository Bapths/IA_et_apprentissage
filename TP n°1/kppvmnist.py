import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
	
from sklearn import neighbors


#### Fonctions de chargement et affichage de la base mnist ####

def load_mnist(m,mtest):

	X = np.load("mnistX.npy")
	y = np.load("mnisty.npy")

	random_state = check_random_state(0)
	permutation = random_state.permutation(X.shape[0])
	X = X[permutation]
	y = y[permutation]
	X = X.reshape((X.shape[0], -1))

	return train_test_split(X, y, train_size=m, test_size=mtest)


def showimage(x):
	plt.imshow( 255 - np.reshape(x, (28, 28) ), cmap="gray")
	plt.show()
	

#############################
#### Programme principal ####

# chargement de la base mnist:
Xtrain, Xtest, Ytrain, Ytest = load_mnist(11000, 1000)
Xapp = Xtrain[:10000, :]
Yapp = Ytrain[:10000]
Xvalid = Xtrain[10000:, :]
Yvalid = Ytrain[10000:]

print("Taille de la base d'apprentissage : ", Xtrain.shape)

bestErr = 1
bestK = 1

for K in range(1, 11) :

	# création du modèle
	kppv = neighbors.KNeighborsClassifier(K)

	# apprentissage sur X,Y
	kppv.fit(Xapp, Yapp)

	# prédiction de la catégorie pour tous les points de test
	Ypred = kppv.predict(Xvalid)

	err = np.mean(Ypred != Yvalid)
	print(f"K={K}, err={err}")
	if err < bestErr :
		bestErr = err
		bestK = K

print(f"Le K choisi est K={bestK} pour une erreur SUR LA BASE DE VALIDATION err={bestErr}.")

# vrai évaluation de l'erreur pour notre meilleur K
kppv = neighbors.KNeighborsClassifier(bestK)
kppv.fit(Xapp, Yapp)
Ypred = kppv.predict(Xtest)

bestKerr = np.mean(Ypred != Ytest)
print(f"Le meilleur K est K={bestK} pour une erreur RÉELLE de err={bestKerr}.")

commonErr = dict()
for i in range(len(Ypred)) :
	if Ypred[i] != Ytest[i] :
		mini = min(Ypred[i],Ytest[i])
		maxi = max(Ypred[i],Ytest[i])
		key = str(mini) + "-" + str(maxi)
		if key in commonErr.keys() :
			commonErr[key] += 1
		else :
			commonErr[key] = 1

print(commonErr)

# Question 7 :
Xtrain, Xtest, Ytrain, Ytest = load_mnist(60000, 10000)

bestK = 5
kppv = neighbors.KNeighborsClassifier(bestK)
kppv.fit(Xtrain, Ytrain)
Ypred = kppv.predict(Xtest)

bestKerr = np.mean(Ypred != Ytest)
print(f"Pour K={bestK} sur un apprentisage sur 60 000 cas et un test sur 10 000 cas, on trouve une erreur de err={bestKerr}.")

commonErr = dict()
for i in range(len(Ypred)) :
	if Ypred[i] != Ytest[i] :
		mini = min(Ypred[i],Ytest[i])
		maxi = max(Ypred[i],Ytest[i])
		key = str(mini) + "-" + str(maxi)
		if key in commonErr.keys() :
			commonErr[key] += 1
		else :
			commonErr[key] = 1

print("err : quantité")
for key in commonErr.keys() :
	print(f"{key} : {commonErr[key]}")