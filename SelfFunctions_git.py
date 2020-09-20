# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:42:04 2020

@author: maisuiY
"""
import numpy as np
from sklearn import linear_model
import math

def wavelet_transform(x, J):
    N = len(x)
    C = np.zeros(shape=(J + 1, N))
    # W: wavelet coefficients
    W = np.zeros(shape=(J + 1, N))
    C[0, :] = x.copy()
    for j in range(1, J + 1):
        for k in range(1, N):
            if k >= np.power(2, j - 1):
                C[j, k] = 1 / 2 * (C[j - 1, k] + C[j - 1, k - np.power(2, j - 1)])
            W[j, k] = C[j - 1][k] - C[j, k]
    W[0, :] = C[J, :]
    return W[:, np.power(2, J - 1):]

def data_segmentation(TD, lenBlocks, lenDecomposition):
    innerTD = TD
    innerLenBlocks = lenBlocks
    innerLenDecomposition = lenDecomposition
    allBlocksTrain = []
    for i in range(len(innerTD)):
        sinBlocksTrain = []
        j = 0
        while(j < lenDecomposition):
            if j <= lenDecomposition - innerLenBlocks + 1:
                sinBlocksTrain.append(innerTD[i][:, j: j + innerLenBlocks])
            else:
                sinBlocksTrain.append(innerTD[i][:, j:])
            j = j + innerLenBlocks
        allBlocksTrain.append(sinBlocksTrain)
    return allBlocksTrain
 
def Blocks_SEG(allBlocksTrain, numberTrain, numNodes):
    innerAllBlocksTrain = allBlocksTrain
    allSEGTrain = []
    for i in range(numberTrain):
        sinBlocksSEG = []
        for j in range(len(innerAllBlocksTrain[i])):
            W_learned = SEG_learning(innerAllBlocksTrain[i][j], numNodes)
            sinBlocksSEG.append(W_learned)
        allSEGTrain.append(sinBlocksSEG)
    return allSEGTrain
        
            
def SEG_learning(singleBlockTrain, numNodes):
    innerSingleBlockTrain = singleBlockTrain
    innerNumNodes = numNodes
    tol = 1e-5
    alpha = 1e-15  
    W_learned = np.zeros((innerNumNodes, innerNumNodes))
    # clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, tol=tol, max_iter = 500)
    clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, tol=tol)
    for i in range(innerNumNodes):
        samples = GetData(innerSingleBlockTrain, i)
        clf.fit(samples[:, :-1], samples[:, -1])
        W_learned[i, :] = clf.coef_
    return W_learned

def SEG_calculating(singleBlockTrain, numNodes):
    innerSingleBlockTrain = singleBlockTrain
    innerNumNodes = numNodes
    tol = 1e-5
    alpha = 1e-10  
    W_learned = np.zeros((innerNumNodes, innerNumNodes))
    # clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, tol=tol, max_iter = 500)
    clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, tol=tol, solver= 'cholesky')
    for i in range(innerNumNodes):
        samples = GetData(innerSingleBlockTrain, i)
        clf.fit(samples[:, :-1], samples[:, -1])
        W_learned[i, :] = clf.coef_
    return W_learned

def GetData(L, location):
    len_data = L.shape[1] - 1
    tmp = L.shape[0] + 1
    totalData = np.zeros((len_data, tmp))
    for i in range(len_data):
        totalData[i, :-1] = L[:, i]
        y = L[location, i+1]
        # if y > 0.9999:
        #     y = 0.9999
        # elif y < -0.9999:
        #     y = -0.9999
        totalData[i, -1] =(y)
    return totalData

def New_Representation(allBlocksSEG, numberTrain, flag = '0'):
    innerAllBlocksSEG = allBlocksSEG
# directly transform the 2-dimensional to 1 dimensional
    if flag == '0': 
        BFTSRTrain = []
        for i in range(numberTrain):
            sinBFTSRTrain = []
            for j in range(len(innerAllBlocksSEG[i])):
                tmp = innerAllBlocksSEG[i][j].flatten()
                for k in range(len(tmp)):
                    sinBFTSRTrain.append(tmp[k])
            BFTSRTrain.append(sinBFTSRTrain)
        return BFTSRTrain
# use the max k eigenvalues of the original SEG matrix to represent the time series
    elif flag == '1': 
        BFTSRTrain = []
        for i in range(numberTrain):
            sinBFTSRTrain = []
            for j in range(len(innerAllBlocksSEG[i])):
                tmpa, tmpb = np.linalg.eig(innerAllBlocksSEG[i][j])
                tmpaABS = list(abs(tmpa))
                tmpaABS.sort(reverse = True)
                if len(tmpaABS) < 3:
                    for k in range(len(tmpaABS)):
                        sinBFTSRTrain.append(tmpaABS[k])
                        # SSD = SSD+1
                else:
                    for k in range(3):
                        sinBFTSRTrain.append(tmpaABS[k])
            BFTSRTrain.append(sinBFTSRTrain)
        return BFTSRTrain
# transformed by dct and then use the max k eigenvalues of the low frequency coefficient matrix
    elif flag == '2':
        for i in range(numberTrain):
            for j in range(len(innerAllBlocksSEG[i])):
                innerAllBlocksSEG[i][j] = dct2(innerAllBlocksSEG[i][j])
                innerAllBlocksSEG[i][j] = np.around(innerAllBlocksSEG[i][j], 4)
        Compression = []
        for i in range(numberTrain):
            localCompression = []
            for j in range(len(innerAllBlocksSEG[i])):
                tmpCom = innerAllBlocksSEG[i][j][0:4, 0:4]
                localCompression.append(tmpCom)
            Compression.append(localCompression)
        BFTSRTrain = []
        for i in range(numberTrain):
            sinBFTSRTrain = []
            for j in range(len(Compression[i])):
                tmpa, tmpb = np.linalg.eig(Compression[i][j])
                tmpaABS = list(abs(tmpa))
                tmpaABS.sort(reverse = True)
                if len(tmpaABS) < 3:
                    for k in range(len(tmpaABS)):
                        sinBFTSRTrain.append(tmpaABS[k])
                        # SSD = SSD+1
                else:
                    for k in range(3):
                        sinBFTSRTrain.append(tmpaABS[k])
            BFTSRTrain.append(sinBFTSRTrain)
        return BFTSRTrain               
                
def dct2(M):
    row = M.shape[0]
    U = np.zeros((row, row),dtype = float)
    for i in range(row):
        if i == 0:
            c = math.sqrt(1/8)
        else:
            c = math.sqrt(1/4)
        for j in range(row):
            U[i,j] = c * math.cos(math.pi * i * (j + 0.5) / row)
    D = np.dot(U, M)
    D = np.dot(D, U.transpose())
    return D
    
        
    
    