import matplotlib.pyplot as plt
import math
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

class EMF():
    #parameters are numpy matrix, k factors, and random weight max
    def __init__(self, npM, k, wMax, wShift):
        sh = npM.shape
        self.P = np.random.rand(sh[0], k)*wMax+wShift
        self.QT = np.random.rand(sh[1], k)*wMax+wShift

    def predict(self, u, i):
        r = np.dot(self.P[u], self.QT[i])

        return r

def constructExplainabilityMatrix(ratings, sh, inds, nbr_size, theta):
    W = np.zeros_like(ratings).astype(float)
    
    for i in tqdm(range(sh[0])):
        for j in range(sh[1]):
            #if ratings[i][j] == 0:
                #list of rating counts for item j in user i's neighborhood
            nbr_rats = [0, 0, 0, 0, 0, 0]
            for ind in inds[i][1:]:
                nbr_rats[ratings[ind][j]] += 1

            #entire neighborhood did not rate the item
            if nbr_rats[0] == nbr_size:
                continue

            #expected value
            exp = 0.0
            for k in range(5):
                exp += (k+1) * float(nbr_rats[k+1]) / (nbr_size)
            
            if exp >= theta:
                W[i][j] = exp

    print("Total non-zero:", np.count_nonzero(W))
    return W

def constructUtilityMatrix(ratings, sh, phi):
    U = np.zeros_like(ratings).astype(float)

    nanrat = np.copy(ratings).astype(float)
    nanrat[nanrat == 0] = np.nan

    U[U == 0]

    item_attr = []
    for i in range(sh[1]):
        if np.isnan(nanrat[:, i]).all():
            item_attr.append([0, 0, 0])
            continue

        nbr = np.sum(~np.isnan(nanrat[:, i]))
        mean = np.nanmean(nanrat[:, i])
        var = np.nanvar(nanrat[:, i])
        item_attr.append([nbr, mean, var])

    item_attr = np.asarray(item_attr)

    for i in tqdm(range(sh[0])):
        x = []
        y = []
        for j in range(sh[1]):
            if ratings[i][j] != 0:
                x.append(j)
                y.append(ratings[i][j])

        if len(set(y)) < 2:
            continue

        lr = LogisticRegression(max_iter=3000, multi_class='multinomial').fit(item_attr[x], np.asarray(y))
        
        for j in range(sh[1]):
            #if ratings[i][j] == 0:
            exp = lr.predict(item_attr[j].reshape(1, -1))
            if exp >= phi:
                U[i][j] = exp
            
    print("Total non-zero:", np.count_nonzero(U))
    return U

def plotExplainability(svd, W, axes, srow, scol, colr):
    #pick a user with regular amount of explainable points
    u = 523
    #u = 49

    #indices for explainable and non explainable points
    exp = []
    not_exp = []

    for i, expval in enumerate(W[u]):
        if expval == 0:
            not_exp.append(i)
        else:
            exp.append(i)

    exp = np.asarray(exp)
    not_exp = np.asarray(not_exp)

    explainable_points = [svd.QT[tuple(exp.T), 0], svd.QT[tuple(exp.T), 1]]

    #not explainable points
    points = [list(svd.QT[tuple(not_exp.T), 0]), list(svd.QT[tuple(not_exp.T), 1])]

    #not explainable, potentially recommended (cos_sim > 0.9)
    pot_points = [[], []]
    b = np.asarray([svd.P[u][0], svd.P[u][1]])
    cnt = 0
    for i in range(len(points[0])):
        g = i - cnt
        a = np.asarray([points[0][g], points[1][g]])
    
        cosSim = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))

        if cosSim > 0.9:
            pot_points[0].append(points[0][g])
            pot_points[1].append(points[1][g])

            points[0].pop(g)
            points[1].pop(g)

            cnt += 1

    #show users
    #users = [svd.P[:, 0], svd.P[:, 1]]
    #axes[m].scatter(users[0], users[1], c='purple')

    axes[srow][scol].scatter(points[0], points[1], c='cyan')
    axes[srow][scol].scatter(pot_points[0], pot_points[1], c='green')
    axes[srow][scol].scatter(explainable_points[0], explainable_points[1], c=colr)
    axes[srow][scol].plot([svd.P[u][0]], [svd.P[u][1]], marker='o', markersize=10, markeredgecolor='black', markerfacecolor='black')

def main():
    random.seed(11)
    np.random.seed(11)

    #Hyperparameters
    beta = 0.01
    theta = 0.1
    phi = 5
    lambdav = 0.005
    alph = 0.001
    nbr_size = 20
    factors = 2

    #Other parameters
    wMax = 2.5
    wShift = 0
    num_epochs = 25
    Jtype = ['regular', 'explainable', 'utility']

    #Reading dataset (MovieLens 1M movie ratings dataset: downloaded from https://grouplens.org/datasets/movielens/1m/)
    data = pd.io.parsers.read_csv('data/ratings.dat', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::', encoding='latin1')

    ratings_mat = np.ndarray(
        shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
        dtype=np.uint8)
    ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

    train_set = ratings_mat[:700]
    sh = train_set.shape

    #TRAIN TEST SPLIT
    nonzero = np.nonzero(train_set)
    indices = []
    for i in range(len(nonzero[0])):
        indices.append([nonzero[0][i], nonzero[1][i]])

    indices = np.asarray(indices)

    test_index = np.random.choice(len(indices), math.floor(len(indices)*0.2), replace=False)
    test_indices = indices[test_index, :]

    test_set = np.zeros_like(train_set)
    test_set[tuple(test_indices.T)] = train_set[tuple(test_indices.T)]
    train_set[tuple(test_indices.T)] = 0

    #UTILITY MATRIX
    print("Constructing Utility Matrix...")
    U = constructUtilityMatrix(train_set, sh, phi)

    #EXPLAINABILITY MATRIX
    nbrs = NearestNeighbors(n_neighbors=nbr_size+1, algorithm='kd_tree').fit(train_set)

    print("\nFinding Nearest Neighbors...")
    dists, inds = nbrs.kneighbors(train_set)

    print("Constructing Explainability Matrix...")
    W = constructExplainabilityMatrix(train_set, sh, inds, nbr_size, theta)

    #PREPARE PLOTS
    fig, expAxes = plt.subplots(nrows=2, ncols=2)

    print("\nBegin Training...")
    for m in range(3):
        print("Training with J = " + str(Jtype[m]) + "...")
        svd = EMF(train_set, factors, wMax, wShift)

        epoch_losses = []
        for epoch in range(num_epochs):
            rloss = 0
            for i in tqdm(range(sh[0])):
                for j in range(sh[1]):
                    if train_set[i][j] != 0:
                        r = svd.predict(i, j)
                        err = (train_set[i][j] - r)

                        if Jtype[m] == 'regular':
                            #regularized
                            svd.P[i] += alph * (svd.QT[j] * err - beta * svd.P[i])
                            svd.QT[j] += alph * (svd.P[i] * err - beta * svd.QT[j])

                        elif Jtype[m] == 'explainable':
                            #regularized with explainable soft constraint
                            svd.P[i] += alph * (svd.QT[j] * err - beta * svd.P[i] - lambdav * (svd.P[i] - svd.QT[j]) * W[i][j])
                            svd.QT[j] += alph * (svd.P[i] * err - beta * svd.QT[j] + lambdav * (svd.P[i] - svd.QT[j]) * W[i][j])

                        elif Jtype[m] == 'utility':
                            #regularized with utility soft constraint
                            svd.P[i] += alph * (svd.QT[j] * err - beta * svd.P[i] - lambdav * (svd.P[i] - svd.QT[j]) * U[i][j])
                            svd.QT[j] += alph * (svd.P[i] * err - beta * svd.QT[j] + lambdav * (svd.P[i] - svd.QT[j]) * U[i][j])

                        rloss += err**2

            print("EPOCH:", epoch+1, "LOSS:", rloss)
            epoch_losses.append(rloss)

        if Jtype[m] == 'regular':
            plotExplainability(svd, W, expAxes, 0, 0, 'red')
            plotExplainability(svd, U, expAxes, 1, 0, 'pink')
        elif Jtype[m] == 'explainable':
            plotExplainability(svd, W, expAxes, 0, 1, 'red')
        elif Jtype[m] == 'utility':
            plotExplainability(svd, U, expAxes, 1, 1, 'pink')

    plt.legend()
    expAxes[0][0].set_title('MF')
    expAxes[0][1].set_title('EMF')

    expAxes[1][0].set_title('MF')
    expAxes[1][1].set_title('UMF')

    plt.show()

if __name__ == "__main__":
    main()