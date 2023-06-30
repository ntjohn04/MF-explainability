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

def plotExplainability(user, svd, W, axes, srow, scol, colr, thresh):
    #pick a user with regular amount of explainable points
    #u = 523
    #u = 49
    #u = 164
    u = user

    #indices for explainable and non explainable points
    exp = []
    not_exp = []

    for i, expval in enumerate(W[u]):
        if expval < thresh:
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
    #axes[srow][scol].scatter(users[0], users[1], c='purple')

    axes[srow][scol].scatter(points[0], points[1], c='cyan')
    axes[srow][scol].scatter(pot_points[0], pot_points[1], c='green')
    axes[srow][scol].scatter(explainable_points[0], explainable_points[1], c=colr)
    axes[srow][scol].plot([svd.P[u][0]], [svd.P[u][1]], marker='o', markersize=10, markeredgecolor='black', markerfacecolor='black')

def getScores(train_set, svd, sh, test_set, U, W, topn, Jtype, sTheta, sPhi):
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
            if train_set[i][j] != 0:
                ls.append(0)
            else:
                ls.append(svd.predict(i, j))

        ls = np.asarray(ls)
        inds = ls.argsort()[::-1][:topn]

        #print()
        #for ind in inds:
        #    print(ls[ind], W[i][ind] > sTheta)

        exps = 0
        utils = 0

        tot_exps = 0
        for j, val in enumerate(W[i]):
            if val >= sTheta and train_set[i][j] == 0:
                tot_exps += 1

        tot_utils = 0
        for j, val in enumerate(U[i]):
            if val >= sPhi and train_set[i][j] == 0:
                tot_utils += 1

        for ind in inds:
            if U[i][ind] >= sPhi:
                utils += 1
            if W[i][ind] >= sTheta:
                exps += 1

        #print()
        #print(i)
        #print(exps, utils)
        #print(tot_exps, tot_utils)

        mep += exps
        mup += utils

        if tot_exps != 0:
            mer += exps / tot_exps
        
        if tot_utils != 0:
            mur += utils / tot_utils

    mep, mer, mup, mur = mep / sh[0] / topn, mer / sh[0], mup / sh[0] / topn, mur / sh[0]

    return [Jtype, rmse, mep, mer, mup, mur]

def displayScores(scores, topn):
    print("@" + str(topn) + "\t\tRMSE\t\tMEP\t\tMER\t\tMUP\t\tMUR")
    for score in scores:
        print(score[0], end="\t\t")
        for i in range(1, 6):
            print(round(score[i], 4), end="\t\t")
        print()

def main():
    random.seed(11)
    np.random.seed(11)

    #Hyperparameters
    factors = 10

    #Other parameters
    wMax = 2.5
    wShift = 0
    topn = 50
    sTheta = [0.01, 0.1, 0.3]
    sPhi = [3, 4, 5]

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

    U = np.load("u.npy")
    W = np.load("W.npy")

    reg_svd = EMF(train_set, factors, wMax, wShift)
    reg_svd.P = np.load("regular_P.npy")
    reg_svd.QT = np.load("regular_QT.npy")

    exp_svd = EMF(train_set, factors, wMax, wShift)
    exp_svd.P = np.load("explain_P.npy")
    exp_svd.QT = np.load("explain_QT.npy")

    util_svd = EMF(train_set, factors, wMax, wShift)
    util_svd.P = np.load("utility_P.npy")
    util_svd.QT = np.load("utility_QT.npy")

    n = int(input("Enter User ID (-1 to get scores): "))
    while (n != -1):
        fig, expAxes = plt.subplots(nrows=2, ncols=3)
        plotExplainability(n, reg_svd, W, expAxes, 0, 0, 'red', 0.1)
        plotExplainability(n, reg_svd, U, expAxes, 1, 0, 'pink', 5)

        plotExplainability(n, exp_svd, W, expAxes, 0, 1, 'red', 0.1)
        plotExplainability(n, exp_svd, U, expAxes, 1, 1, 'pink', 5)
        
        plotExplainability(n, util_svd, W, expAxes, 0, 2, 'red', 0.1)
        plotExplainability(n, util_svd, U, expAxes, 1, 2, 'pink', 5)

        plt.legend()
        expAxes[0][0].set_title('MF')
        expAxes[1][0].set_title('MF')

        expAxes[0][1].set_title('EMF')
        expAxes[1][1].set_title('EMF')

        expAxes[0][2].set_title('UMF')
        expAxes[1][2].set_title('UMF')

        plt.show()
        n = int(input("Enter User ID: "))

    for a in range(len(sTheta)):
        scores = []
        print("PARAMS")
        print("Theta:", str(sTheta[a]), "Phi:", str(sPhi[a]))
        e_cnt = 0
        u_cnt = 0
        for i in range(sh[0]):
            for j in range(sh[1]):
                if W[i][j] >= sTheta[a] and W[i][j] != 0:
                    e_cnt += 1
                if U[i][j] >= sPhi[a] and U[i][j] != 0:
                    u_cnt += 1

        print("Exp Non Zero:", np.count_nonzero(W))
        print("Exp Threshold:", e_cnt)

        print("Util Non Zero:", np.count_nonzero(U))
        print("Util Threshold:", u_cnt)

        scores.append(getScores(train_set, reg_svd, sh, test_set, U, W, topn, "regular", sTheta[a], sPhi[a]))
        displayScores(scores, topn)
        scores.append(getScores(train_set, exp_svd, sh, test_set, U, W, topn, "explain", sTheta[a], sPhi[a]))
        displayScores(scores, topn)
        scores.append(getScores(train_set, util_svd, sh, test_set, U, W, topn, "utility", sTheta[a], sPhi[a]))
        displayScores(scores, topn)
    

        pd.DataFrame(scores).to_csv("scores" + str(a) + ".csv")
    

if __name__ == "__main__":
    main()