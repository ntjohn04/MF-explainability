import matplotlib.pyplot as plt
import math
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

CONSTRUCT_EXPLAIN = True
CONSTRUCT_UTILITY = True

class EMF():
    #parameters are numpy matrix, k factors, and random weight max
    def __init__(self, npM, k, wMax, wShift):
        sh = npM.shape
        self.P = np.random.rand(sh[0], k)*wMax+wShift
        self.QT = np.random.rand(sh[1], k)*wMax+wShift

    def predict(self, u, i):
        r = np.dot(self.P[u], self.QT[i])

        return r

def constructExplainabilityMatrix(ratings, sh, inds, nbr_size):
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
            
            W[i][j] = exp

    print("Total non-zero:", np.count_nonzero(W))
    return W

#combined utility global and similar
def constructUtilityMatrix(ratings, sh, inds, nbr_size, phi):
    U = np.zeros_like(ratings).astype(float)

    nanrat = np.copy(ratings).astype(float)
    nanrat[nanrat == 0] = np.nan

    #Global Item Attributes
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

    #Similar User Item Attributes
    print("Getting Attributes...")
    sim_attr = [[0]*sh[1] for _ in range(sh[0])]
    for i in tqdm(range(sh[0])):
        for j in range(sh[1]):
            if nanrat[i][j] != np.nan:
                v = nanrat[inds[i][1:], i]
                sim_nbr = np.sum(~np.isnan(v))
                sim_mean = np.nanmean(v)
                sim_var = np.nanvar(v)

                sim_attr[i][j] = [sim_nbr, sim_mean, sim_var]


    print("Filling Matrix...")
    for i in tqdm(range(sh[0])):
        x = []
        for j in range(sh[1]):
            if ratings[i][j] != 0:
                x.append(j)

        y = ratings[i][x]

        if len(set(y)) < 2:
            continue

        full_attr = []
        for j in range(sh[1]):
            full_attr.append(np.concatenate((item_attr[j], sim_attr[i][j])))

        full_attr = np.asarray(full_attr)
        full_attr = np.nan_to_num(full_attr)

        lr = SVR(max_iter=3000).fit(full_attr[x], y)
        
        for j in range(sh[1]):
            exp = lr.predict(full_attr[j].reshape(1, -1))
            U[i][j] = exp
            
    print("Total non-zero:", np.count_nonzero(U))
    return U

def getScores(ratings, svd, sh, test_set, U, W, topn, Jtype, sTheta, sPhi):
    rmse = 0
    for i in range(sh[0]):
        for j in range(sh[1]):
            if test_set[i][j] != 0:
                pred = svd.predict(i, j)
                err = float(test_set[i][j]) - pred
                rmse += err**2
    
    rmse = rmse / np.count_nonzero(test_set)
    rmse = math.sqrt(rmse)

    mep = 0
    mer = 0

    mup = 0
    mur = 0
    for i in tqdm(range(sh[0])):
        ls = []
        for j in range(sh[1]):
            if ratings[i][j] != 0:
                ls.append(0)
            else:
                ls.append(svd.predict(i, j))

        ls = np.asarray(ls)
        inds = ls.argsort()[::-1][:topn]

        exps = 0
        utils = 0

        tot_exps = 0
        for val in W[i]:
            if val >= sTheta:
                tot_exps += 1

        tot_utils = 0
        for val in U[i]:
            if val >= sPhi:
                tot_utils += 1

        for ind in inds:
            if U[i][ind] >= sPhi:
                utils += 1
            if W[i][ind] >= sTheta:
                exps += 1

        mep += exps
        mup += utils

        if tot_exps != 0:
            mer += exps / tot_exps
        
        if tot_utils != 0:
            mur += utils / tot_utils

    mep, mer, mup, mur = mep / sh[0] / topn, mer / sh[0], mup / sh[0] / topn, mur / sh[0]

    return [Jtype, rmse, mep, mer, mup, mur]

def displayScores(scores):
    print("\t\tRMSE\t\tMEP\t\tMER\t\tMUP\t\tMUR")
    for score in scores:
        print(score[0], end="\t\t")
        for i in range(1, 6):
            print(round(score[i], 4), end="\t\t")
        print()

def getNeighbors(ratings, nbr_size):
    inds = []
    for a, r1 in enumerate(tqdm(ratings)):
        sims = []
        for b, r2 in enumerate(ratings):
            if a != b:
                cos_sim = np.dot(r1, r2) / (np.linalg.norm(r1)*np.linalg.norm(r2))
                if np.isnan(cos_sim):
                    cos_sim = 0
                sims.append(cos_sim)
            else:
                sims.append(3)

        inds.append(np.asarray(sims).argsort()[::-1][:nbr_size+1])
    return np.asarray(inds)

def main():
    random.seed(11)
    np.random.seed(11)

    #Hyperparameters
    beta = 0.01
    theta = 0.01
    phi = 0.01
    lambdav = 0.005
    alph = 0.001
    nbr_size = 50
    factors = 10

    #Other parameters
    wMax = 1.22
    wShift = 0
    num_epochs = 30
    topn = 5
    Jtype = ['regular', 'explain', 'utility']
    sTheta = 0.01
    sPhi = 0.01

    #Reading dataset (MovieLens 1M movie ratings dataset: downloaded from https://grouplens.org/datasets/movielens/1m/)
    data = pd.io.parsers.read_csv('data/ratings.dat', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::', encoding='latin1')

    ratings_mat = np.ndarray(
        shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
        dtype=np.uint8)
    ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

    full_set = ratings_mat
    train_set = np.copy(full_set)
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
    if CONSTRUCT_UTILITY:
        print("\nFinding Nearest Neighbors...")
        inds = getNeighbors(full_set, nbr_size)
        print("Constructing Utility Matrix...")
        U = constructUtilityMatrix(full_set, sh, inds, nbr_size, phi)
        #U = constructUtilityMatrix(full_set, sh)
        np.save("u.npy", U)
    else:
        print("Loading Utility Matrix...")
        U = np.load("u.npy")

    #EXPLAINABILITY MATRIX
    if CONSTRUCT_EXPLAIN:
        print("\nFinding Nearest Neighbors...")
        inds = getNeighbors(full_set, nbr_size)

        print("Constructing Explainability Matrix...")
        W = constructExplainabilityMatrix(full_set, sh, inds, nbr_size)
        np.save("w.npy", W)
    else:
        print("Loading Explainability Matrix...")
        W = np.load("w.npy")

    U_train = np.copy(U)
    U_train[tuple(test_indices.T)] = 0

    W_train = np.copy(W)
    W_train[tuple(test_indices.T)] = 0

    #PREPARE PLOTS
    fig, expAxes = plt.subplots(nrows=2, ncols=3)

    print("\nBegin Training...")
    scores = []
    for m in range(len(Jtype)):
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

                        if Jtype[m] == 'regular' or (Jtype[m] == 'explain' and W_train[i][j] < theta) or (Jtype[m] == 'utility' and U_train[i][j] < phi):
                            #regularized
                            svd.P[i] += alph * (2 * svd.QT[j] * err - beta * svd.P[i])
                            svd.QT[j] += alph * (2 * svd.P[i] * err - beta * svd.QT[j])

                        elif Jtype[m] == 'explain':
                            #regularized with explainable soft constraint
                            svd.P[i] += alph * (2 * svd.QT[j] * err - beta * svd.P[i] - lambdav * (svd.P[i] - svd.QT[j]) * W_train[i][j])
                            svd.QT[j] += alph * (2 * svd.P[i] * err - beta * svd.QT[j] + lambdav * (svd.P[i] - svd.QT[j]) * W_train[i][j])

                        elif Jtype[m] == 'utility':
                            #regularized with utility soft constraint
                            svd.P[i] += alph * (2 * svd.QT[j] * err - beta * svd.P[i] - lambdav * (svd.P[i] - svd.QT[j]) * U_train[i][j])
                            svd.QT[j] += alph * (2 * svd.P[i] * err - beta * svd.QT[j] + lambdav * (svd.P[i] - svd.QT[j]) * U_train[i][j])

                        rloss += err**2

            print("EPOCH:", epoch+1, "LOSS:", rloss)
            epoch_losses.append(rloss)

        scores.append(getScores(train_set, svd, sh, test_set, U, W, topn, Jtype[m], sTheta, sPhi))
        np.save(Jtype[m] + "_P.npy", svd.P)
        np.save(Jtype[m] + "_QT.npy", svd.QT)

    displayScores(scores)

if __name__ == "__main__":
    main()