import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import svm

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from termcolor import colored
from time import process_time

start = process_time()

# charger les données de défauts de rails
data = np.loadtxt("defautsrails.dat")
X = data[:,:-1]  # tout sauf dernière colonne
Y = data[:,-1]   # uniquement dernière colonne

C = 1
scores = []
print('\n# Création des classifieurs binaires et évalutation de l\'erreur sur la base d\'apprentissage.\n')

for i in range(1, 5) :
    Y_bin = 2*(Y == i)-1

    # création du modèle
    model = svm.LinearSVC(C=C)

    # apprentissage
    model.fit(X, Y_bin)

    # prediction
    Ypred = model.predict(X)
    err_app = np.mean(Ypred != Y_bin)

    print(colored('Cas ', 'magenta') + colored(f'{i}', 'magenta', attrs=['bold']) + colored(' contre tous : erreur d\'apprentissage = ', 'magenta') + colored(f'{err_app}', 'magenta', attrs=['bold']))

    score = model.decision_function(X)
    scores.append(score)

print('\n# Création du classifieurs multi-class général par combinaison des classifieurs binaires.\n')

# Prédiction de tous les cas avec le multiclasse général par combinaison des classifieurs binaires
Ypred = np.argmax(scores, axis=0)+1 # pour une matrice

# Calcul de l'erreur par rapport aux données d'apprentissage
err = np.mean(Ypred != Y)
print(colored('Erreur d\'apprentissage sur le multi-class global = ', 'magenta') + colored(f'{err}\n', 'magenta', attrs=['bold']))

def LOO(index) :

    success = colored('O', 'green', attrs=['bold'])
    fail = colored('X', 'red', attrs=['bold'])

    C = 1
    scores = []
    errs_bin = []

    X_i = np.delete(X, index, axis=0)
    Y_i = np.delete(Y, index)

    x_test = X[index]
    y_test = Y[index]

    errs_bin_c = [0 for i in range(4)]

    for i in range(1, 5) :
        Y_bin = 2*(Y_i == i)-1
        y_test_bin = 2*(y_test == i)-1

        # création du modèle
        model = svm.LinearSVC(C=C)

        # apprentissage
        model.fit(X_i, Y_bin)

        # calcul du score de i
        score = model.decision_function([x_test])
        scores.append(score)

        # prediction
        Ypred = model.predict([x_test])
        if Ypred[0] == y_test_bin :
            errs_bin.append(success)
            errs_bin_c[i-1] = 0
        else :
            errs_bin.append(fail)
            errs_bin_c[i-1] = 1
    
    bin_err_line = ''
    for i in range(4) :
        bin_err_line += f'{errs_bin[i]} '
    bin_err_line = bin_err_line[:-1]

    # Prédiction de tous les cas avec le multiclasse général par combinaison des classifieurs binaires
    Ypred = np.argmax(scores, axis=0)+1 # pour un vecteur

    if index < 10 :
        index_str = colored('00' + str(index), 'blue')
    elif index < 100 :
        index_str = colored('0' + str(index), 'blue')
    else :
        index_str = colored(index, 'blue')

    if Ypred[0] == y_test :
        print(f'[ {index_str} ] {bin_err_line} -> {success}', end=colored('  |  ', 'grey'))
        if (index+1)%4 == 0 :
            print()
        return [0] + errs_bin_c
    print(f'[ {index_str} ] {bin_err_line} -> {fail}', end=colored('  |  ', 'grey'))
    if (index+1)%4 == 0 :
        print()
    return [1] + errs_bin_c

# application de la technique LOO
err_total = 0
err_bin_total = [0 for i in range(4)]

for i in range(140) :
    errs = LOO(i)
    err_total += errs[0]
    for j in range(4) :
        err_bin_total[j] += errs[j+1]

err = err_total/140
print(colored(f'\nL\'erreur de généralisation par méthode LOO est éstimée à err = ', 'magenta') + colored(f'{err}\n', 'magenta', attrs=['bold']))

for i in range(4) :
    print(colored('Le classifieur binaire ', 'magenta') + colored(f'{i+1}', 'magenta', attrs=['bold']) + colored(' a fait ', 'magenta') + colored(f'{err_bin_total[i]}', 'magenta', attrs=['bold']) + colored(' erreur(s) sur 140 soit une erreur err = ', 'magenta') + colored(f'{err_bin_total[i]/140}', 'magenta', attrs=['bold']))

stop = round(process_time()-start, 2)
print(colored('\nTemps de calcul : ', 'green')+colored(f'~{stop}', 'green', attrs=['bold'])+colored(' secondes.\n', 'green'))