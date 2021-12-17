import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


def load_mnist(m,mtest):

	# X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
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
	
Xtrain, Xtest, ytrain, ytest = load_mnist(10000, 1000)

for i in range(10000):
	print("X[" + str(i) + "] = ", Xtrain[i])
	print("Image numéro " + str(i) + " étiquettée " + str(ytrain[i]))
	showimage(Xtrain[i])


