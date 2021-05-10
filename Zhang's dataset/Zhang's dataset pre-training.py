import numpy as np
import tensorflow as tf
import gc
import random
import math
import random
from datetime import datetime
import requests
import pandas as pd
import csv
import sqlite3
import random
import os
import sys
import networkx as nx
from numpy import linalg as LA
from numpy.linalg import inv
import copy
import csv
import array
import random
import numpy
import keras
from keras import regularizers
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from keras.layers import normalization
from time import time
import multiprocessing as mp
import sys
import math


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]




if __name__ == "__main__":

    with open('data//entities.txt') as f:
        entity2id = {}
        for line in f:
            eid, entity1 = line.strip().split('\t')
            entity2id[eid] = int(entity1)

    csv_keyword = csv.reader(open('data//'  + 'drug-disease1.csv', 'r'))
    keywords = []
    for row in csv_keyword:
        keywords.append(row)

    csv_keyword2 = csv.reader(open('data//' + 'otherlink.csv', 'r'))
    keywords2 = []
    for row in csv_keyword2:
        keywords2.append(row)

    interaction = []
    for i in range(0, len(keywords)):
        a = []
        a.append(keywords[i][0])
        a.append(str(keywords[i][2]))
        interaction.append(a)
    lengleng = len(interaction)
    entitya = 267
    entityb = 570
    lgcnallresult = []
    kgallresult = []
    pinqilai = []
    for xxx in range(1):
        np.random.seed(xxx)
        np.random.shuffle(interaction)

        k = 0
        print('this is the ' + str(xxx) + 'fold')

        testone = interaction[k * (len(interaction) // 5):(k + 1) * (len(interaction) // 5)]
        trainone = interaction[(k + 1) * (len(interaction) // 5):len(interaction)]
        testone = np.array(testone)
        print(testone.shape)
        length = (k + 1) * (len(interaction) // 5) - k * len(interaction) // 5






        # with open('data//'  + 'test.txt', 'w') as f:
        #     for i in range(k * (len(interaction) // 5), (k + 1) * (len(interaction) // 5)):
        #         f.write(
        #             interaction[i][0] + '\t' + str(
        #                 interaction[i][1].replace(",", '')) + '\t' + 'drug-disease' + '\n')  #
        # with open('data//'  + 'valid.txt', 'w') as f:
        #     for i in range(k * (len(interaction) // 5), (k + 1) * (len(interaction) // 5)):
        #
        #         f.write(
        #             interaction[i][0] + '\t' + str(interaction[i][1].replace(",", '')) + '\t' + 'drug-disease' + '\n')
        #     length = (k + 1) * (len(interaction) // 5) - k * len(interaction) // 5
        with open('data//'+ 'train.txt', 'w') as f:

            for i in range(len(keywords2)):
                f.write(keywords2[i][0] + '\t' + keywords2[i][2].replace(",", '') + '\t' + str(keywords2[i][1]) + '\n')




        os.system('python main.py ' + '--data_dir ' + './data/' + '/ ' + '--method ' + 'complex')

