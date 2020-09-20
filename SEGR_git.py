# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:08:35 2020

@author: maisuiY
"""

# Import the package of Python
import numpy as np
import pandas as pd
import operator
from SelfFunctions_git import wavelet_transform, data_segmentation, Blocks_SEG, New_Representation
from functools import reduce
from sklearn import preprocessing 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import os
from pandas import DataFrame

#######################
# SEGR
#######################

flag = '0' # '0','1' or '2'

for numNodes in range(6, 7):
    accu_STGR_svc = []
    accu_STGR_rf = []
    silhoutteCoefficient_STGR_HC = []
    silhoutteCoefficient_STGR_kmeans = []
    dataset_name = []
    STGRRepresentationLength = []
    # Preparing the data
    root_dir = r"F:\code\python\BFTS_master\data"
    for file in os.listdir(root_dir):
        file_name = root_dir + "\\" + file
        for file1 in os.listdir(file_name):
            if file1 == file + "_TEST.tsv":
                file_name1 = file_name + "\\" + file1
                test = pd.read_csv(file_name1,delimiter='\t',header=None)
            elif file1 == file + "_TRAIN.tsv":
                file_name1 = file_name + "\\" + file1
                train = pd.read_csv(file_name1,delimiter='\t',header=None)
        
        # keep the original splitting
        trainNew = train
        testNew = test
        
        ###### train data
        numberTrain = trainNew.shape[0]
        lenTimeTrain = trainNew.shape[1]
        lenDecompositionTrain = lenTimeTrain - np.power(2, numNodes - 2)
        labelTrain = []
        for i in range(numberTrain):
            labelTrain.append(trainNew.iat[i, 0])
            
        ## Pre_Preparing1. wavelet decomposition
        trainDecomposition = []
        for i in range(numberTrain):
            maxLevel = numNodes - 1
            tmpArray = trainNew[i:i+1].values
            tmpArray = reduce(operator.add, tmpArray)
            # normalTS = preprocessing.minmax_scale(tmpArray[1:].copy(), axis = 0,  feature_range=(-1, 1))
            normalTS = tmpArray[1:].copy()
            # remove the label columns
            coffis = wavelet_transform(normalTS, maxLevel)
            trainDecomposition.append(coffis)
            
            
        ## 1. Data segmentation
        lenBlocks = 50000
        allBlocksTrain = data_segmentation(trainDecomposition, lenBlocks, lenDecompositionTrain)
        

        ## 2.2 without Normalization  
        normalAllBlocksTrain = allBlocksTrain.copy()
        
        ## 3. SEG Learning
        
        allBlocksFCMs = Blocks_SEG(normalAllBlocksTrain, numberTrain, numNodes)
        
        ## 4. SEGR time series 
        
        
        STGRTrain = New_Representation(allBlocksFCMs, numberTrain, flag)
        
        ###### test data
        numberTest = testNew.shape[0]
        lenTimeTest = testNew.shape[1]
        lenDecompositionTest = lenTimeTest - np.power(2, numNodes - 2)
        labelTest = []
        for i in range(numberTest):
            labelTest.append(testNew.iat[i, 0])
             
        ## Pre_Preparing1. wavelet decomposition
        testDecomposition = []
        for i in range(numberTest):
            maxLevel = numNodes - 1
            tmpArray2 = testNew[i:i+1].values
            tmpArray2 = reduce(operator.add, tmpArray2)
            # normalTE = preprocessing.minmax_scale(tmpArray2[1:], axis = 0, feature_range=(-1, 1))
            normalTE = tmpArray2[1:]
            coffis2 = wavelet_transform(normalTE, maxLevel)
            testDecomposition.append(coffis2)
            
            
        ## 1. Data segmentation
        allBlocksTest = data_segmentation(testDecomposition, lenBlocks, lenDecompositionTest)        
            
        ## 2.2 without Normalization    
        normalAllBlocksTest = allBlocksTest.copy()
        
        ## 3. SEG Learning
        
        allBlocksFCMsTest = Blocks_SEG(normalAllBlocksTest, numberTest, numNodes)
        
        ## 4. SEGR time series 
        
        STGRTest = New_Representation(allBlocksFCMsTest, numberTest, flag)
        
        
        ##################################################################
        ## Application1: classification
        ##################################################################
        
        NormalSTGRTrain = preprocessing.scale(STGRTrain)
        NormalSTGRTest = preprocessing.scale(STGRTest)
        svclassifier = SVC(kernel='rbf')
        clf = svclassifier.fit(NormalSTGRTrain, labelTrain)
        accu = clf.score(NormalSTGRTest, labelTest)     
        accu_STGR_svc.append(accu) 
        print('accu_SVC = ', accu)
        forest = RandomForestClassifier(n_estimators=500)
        clf = forest.fit(NormalSTGRTrain, labelTrain)
        accu = clf.score(NormalSTGRTest, labelTest)
        accu_STGR_rf.append(accu)
        print('accu_RF = ', accu)
        
        dataset_name.append(file)
        STGRRepresentationLength.append(len(STGRTrain[0]))
        
                    
        
        # ##################################################################
        # ## Application2: clustering
        # ##################################################################        
        
        # NormalSTGRTrain = preprocessing.scale(STGRTrain)
        # NormalSTGRTest = preprocessing.scale(STGRTest)
        # Totaldata = np.concatenate((NormalSTGRTrain, NormalSTGRTest) ,axis = 0)
        
        # statis = sorted(labelTrain, reverse = True)
        # numClusters = 1
        # for i in range(1, len(statis)):
        #     if abs(statis[i] - statis[i - 1])>0.01:
        #         numClusters = numClusters + 1
        # print("numClusters = ", numClusters)        
        # Hclustering = cluster.AgglomerativeClustering(n_clusters = numClusters, affinity = "cosine", linkage = "complete")
        # Result = Hclustering.fit(Totaldata)
        # Sl = metrics.silhouette_score(Totaldata, Result.labels_, metric='cosine')
        # silhoutteCoefficient_STGR_HC.append(Sl)
        # print('HC_silhoutteCoefficient = ', Sl)
        
        # kmeansclustering = KMeans(n_clusters = numClusters)
        # Result2 = kmeansclustering.fit(Totaldata)
        # Sl2 = metrics.silhouette_score(Totaldata, Result2.labels_, metric='cosine')
        # silhoutteCoefficient_STGR_kmeans.append(Sl2)
        # print('kmeans_silhoutteCoefficient = ', Sl2)
        
        # dataset_name.append(file)
        # STGRRepresentationLength.append(len(STGRTrain[0]))
        
    ##################################################################
    ## Application1: saving classification result
    ##################################################################    
    columnIndex1 = "STGR_SVC_with_" + str(numNodes) + "_Nodes"
    columnIndex2 =  "STGR_RF_with_" + str(numNodes) + "_Nodes"
    columnIndex3 = "STGR_Length"
    ResultDataFrame = {columnIndex1: accu_STGR_svc, columnIndex2: accu_STGR_rf, columnIndex3: STGRRepresentationLength }    
    frame = DataFrame(ResultDataFrame, columns = [columnIndex1, columnIndex2, columnIndex3], index = dataset_name)
    savePath = r"F:\code\python\BFTS_master\Results\ClassificationResults_STGR_" + str(numNodes) + ".csv"
    if not os.path.exists(r"F:\code\python\BFTS_master\Results"):
        os.makedirs(r"F:\code\python\BFTS_master\Results")
    # frame.to_csv(savePath)
    
    # ##################################################################
    # ## Application2: saving clustering result
    # ##################################################################    
    # columnIndex1 = "STGR_HC_with_" + str(numNodes) + "_Nodes"
    # columnIndex2 =  "STGR_kmeans_with_" + str(numNodes) + "_Nodes"
    # columnIndex3 = "STGR_Length"
    # ResultDataFrame = {columnIndex1: silhoutteCoefficient_STGR_HC, columnIndex2: silhoutteCoefficient_STGR_kmeans,columnIndex3: STGRRepresentationLength }    
    # frame = DataFrame(ResultDataFrame, columns = [columnIndex1, columnIndex2, columnIndex3], index = dataset_name)
    # savePath = r"F:\code\python\BFTS_master\Results\ClusteringResults_STGR_" + str(numNodes) + ".csv"
    # if not os.path.exists(r"F:\code\python\BFTS_master\Results"):
    #     os.makedirs(r"F:\code\python\BFTS_master\Results")
    # frame.to_csv(savePath)
     
    


