import numpy as np
import scipy.sparse as sp
import scipy.io as spio
import tensorflow as tf
import gc
import random
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

import random
from datetime import datetime
import os
import requests
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score
import numpy as np
import csv
import sqlite3
import csv
import numpy as np
import random
import os
import sys
import networkx as nx
from numpy import linalg as LA
import math
from numpy.linalg import inv
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import copy
from numpy import linalg as LA
import csv
import array
import random
import numpy

from scipy.linalg import pinv as pinv
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


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 128, 'Number of units in hidden layer 3.')
def get_Jaccard2_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    E = np.ones_like(X.T)
    denominator=X * E + E.T * X.T - X * X.T
    denominator_zero_index=np.where(denominator==0)
    denominator[denominator_zero_index]=1
    result = X * X.T / denominator
    result[denominator_zero_index]=0
    result = result - np.diag(np.diag(result))
    return result

def matrix_normalize(similarity_matrix):
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        similarity_matrix = similarity_matrix - np.diag(np.diag(similarity_matrix))
        for i in range(200):
            D = np.diag(np.array(np.sum(similarity_matrix, axis=1)).flatten())
            xxx=np.sqrt(D)
            xxx[np.isnan(xxx)]=0
            D = np.linalg.pinv(xxx)
            similarity_matrix = D * similarity_matrix * D
    else:
        for i in range(similarity_matrix.shape[0]):
            if np.sum(similarity_matrix[i], axis=1) == 0:
                similarity_matrix[i] = similarity_matrix[i]
            else:
                similarity_matrix[i] = similarity_matrix[i] / np.sum(similarity_matrix[i], axis=1)
    return similarity_matrix
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


def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
    test_index = np.where(train_matrix == 0)
    real_score = interaction_matrix[test_index]
    predict_score = predict_matrix[test_index]
    return get_metrics(real_score, predict_score)


def constructAdjNet(drug_dis_matrix):
    drug_matrix = np.matrix(np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    # adj =  adj + sp.eye(adj.shape[0])
    return adj


def weight_variable_glorot(input_dim, output_dim, name=""):
    # 初始化
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )

    return tf.Variable(initial, name=name)


def weight_variable_glorot2(input_dim, name=""):
    # 初始化
    init_range = np.sqrt(3.0 / (input_dim * 2))
    initial = tf.random_uniform(
        [input_dim, input_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    ) + tf.eye(input_dim)
    # initial = tf.eye(input_dim)
    # + tf.eye(input_dim)
    return tf.Variable(initial, name=name)


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    # keep_prob设置神经元被选中的概率
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    # tf.cast 数据类型转换
    # tf.floor向下取整,ceil向上取整
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    # np.vstack 垂直堆叠数组123，456
    # 堆叠成123
    # 456
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    # 图的预处理，拉普拉斯正则化
    # coo 是一种矩阵格式
    adj_ = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # adj_nomalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
    adj_nomalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_nomalized)


class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')

            # tf.summary.histogram(self.name + '/weights', self.vars['weights'])
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1 - self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)

            outputs = self.act(x)
            # tf.add_to_collection(self.name+'w1',self.vars['weights'])
        return outputs


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
            # tf.summary.histogram(self.name + '/weights', self.vars['weights'])
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
            # tf.add_to_collection('w3',self.vars['weights'])
        return outputs


class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot2(input_dim, name='weights')
            # tf.summary.histogram(self.name + '/weights', self.vars['weights'])

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1 - self.dropout)

            U = inputs[0:267, :]
            V = inputs[267:, :]
            U = tf.matmul(U, self.vars['weights'])
            V = tf.transpose(V)
            x = tf.matmul(U, V)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
            # tf.add_to_collection('w2',self.vars['weights'])
        return outputs,self.vars['weights']


class GCNModel():

    def __init__(self, placeholders, num_features, features_nonzero, adj_nonzero, name, act=tf.nn.elu):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        # atten = tf.random_uniform(
        #     [3,1],
        #     minval=0,
        #     maxval=1,
        #     dtype=tf.float32
        # )
        # self.att=tf.Variable(atten,'attention')
        self.att = tf.Variable(tf.constant([0.5, 0.33, 0.25]))
        with tf.variable_scope(self.name):
            self.build()

    def build(self):
        self.adj = dropout_sparse(self.adj, 1 - self.adjdp, self.adj_nonzero)
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden2 = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden1)

        self.emb = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=FLAGS.hidden2,
            output_dim=FLAGS.hidden3,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden2)

        self.embeddings = self.hidden1 * self.att[0] + self.hidden2 * self.att[1] + self.emb * self.att[2]

        self.reconstructions,self.relationemb = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=FLAGS.hidden3, act=tf.nn.sigmoid)(self.embeddings)


class Optimizer():
    def __init__(self, model, preds, labels, w, lr, num):


        global_step = tf.Variable(0, trainable=False)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=
        #     cyclic_learning_rate(global_step=global_step,learning_rate=lr*0.1,
        #                  max_lr=lr, mode='exp_range',gamma=.995))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        alpha = 0.25
        gamma = 2

        alpha_t = labels * alpha + (tf.ones_like(labels) - labels) * (1 - alpha)

        p_t = labels * preds + (tf.ones_like(labels) - labels) * (tf.ones_like(labels) - preds) + 1e-7
        focal_loss = - alpha_t * tf.pow((tf.ones_like(labels) - p_t), gamma) * tf.log(p_t)
        self.cost = tf.reduce_mean(focal_loss)

        # self.cost = norm * tf.reduce_mean(
        #     tf.nn.weighted_cross_entropy_with_logits(
        #         logits=preds_sub, targets=labels_sub, pos_weight=1))

        self.opt_op = self.optimizer.minimize(self.cost, global_step=global_step, )
        self.grads_vars = self.optimizer.compute_gradients(self.cost)


def constructXNet(drug_dis_matrix, drug_matrix, dis_matrix):
    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))


def Get_embedding_Matrix(train_drug_dis_matrix, seed, epochs, dp, w, lr, drug_dis_matrix, adjdp, num):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    adj = constructAdjNet(train_drug_dis_matrix)  # 没有sim就用这个吧
    # adj=constructXNet(train_drug_dis_matrix,drug_matrix,dis_matrix)
    adj = sp.csr_matrix(adj)
    num_nodes = adj.shape[0]
    num_edges = adj.sum()

    X = constructAdjNet(train_drug_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, features_nonzero, adj_nonzero, name='yeast_gcn')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            w=w, lr=lr, num=num)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  #
    sess.run(tf.global_variables_initializer())

    for epoch in range(10000):  ####
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)  #
            metric_tmp = roc_auc_score(drug_dis_matrix.flatten(), res)  #
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost),
                  "score=")
            print(metric_tmp)
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    relationemb=sess.run(model.relationemb,feed_dict=feed_dict)
    print(111111111111111)
    print(relationemb.shape)
    np.save('lgcn_relation.npy',relationemb)
    embeddingss = sess.run(model.embeddings, feed_dict=feed_dict)
    print(embeddingss.shape)
    embeddingsss = embeddingss
    np.save('lgcn_em.npy', embeddingsss)
    print(sess.run(model.att, feed_dict=feed_dict))
    sess.close()
    print(res.shape)
    return res


def cross_validation_experiment(drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, dp, w, lr, adjdp, g):
    index_matrix = np.mat(np.where(np.abs(drug_dis_matrix) == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam % k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating drug-disease...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k + 1))
        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        drug_len = drug_dis_matrix.shape[0]
        dis_len = drug_dis_matrix.shape[1]

        drug_disease_res = Get_embedding_Matrix(train_matrix, drug_matrix, dis_matrix, seed, epochs, dp, w, lr,
                                                drug_dis_matrix, adjdp)
        predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)

        metric_tmp = cv_model_evaluate(drug_dis_matrix, predict_y_proba, train_matrix)
        print(metric_tmp)

        metric += metric_tmp

        del train_matrix

        gc.collect()

    print(metric / k_folds)

    metric = np.array(metric / k_folds)

    return metric



if __name__ == "__main__":

    with open('data/entities.txt') as f:
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
        if keywords[i][3]=='1':   # is used to identify therapeutic associations
            a = []
            a.append(keywords[i][0])
            a.append(str(keywords[i][2]))
            interaction.append(a)
    lengleng = len(interaction)
    entitya = 267
    entityb = 570
    lgcnallresult = []#
    kgallresult = []
    pinqilai = []
    for xxx in range(50):
        np.random.seed(xxx)
        np.random.shuffle(interaction)
        k = xxx%4
        print('this is the ' + str(xxx) + 'fold')

        testone = interaction[k * (len(interaction) // 5):(k + 1) * (len(interaction) // 5)]
        trainone = interaction[(k + 1) * (len(interaction) // 5):len(interaction)]
        testone = np.array(testone)
        print(testone.shape)
        length = (k + 1) * (len(interaction) // 5) - k * len(interaction) // 5
        cx = sqlite3.connect("data//alldatamin.db")
        cur = cx.cursor()
        cur.execute("select distinct hid from alldata where interactiontype='drug-disease'")
        drug = cur.fetchall()
        for i in range(len(drug)):
            drug[i] = str(drug[i]).replace("'", '').replace("(", "").replace(")", '').replace(",", '')
        with open('data//druglist.txt', 'w') as f:
            for i in range(len(drug)):
                f.write(str(drug[i]) + '\n')

        cur.execute("select distinct tid from alldata where interactiontype='drug-disease'")
        disease = cur.fetchall()
        with open('data//diseaselist.txt', 'w') as f:
            for i in range(len(disease)):
                f.write(str(disease[i]) + '\n')
        for i in range(len(disease)):
            disease[i] = str(disease[i]).replace("'", '').replace("(", "").replace(")", '').replace(",", '')



        import csv

        csv_keyword = csv.reader(open('data//drug-disease1.csv', 'r'))
        keywords = []
        for row in csv_keyword:
            keywords.append(row)
        matys = np.zeros((267, 570))
        for i in range(0, k * (len(interaction) // 5)):
            matys[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1
        for i in range((k + 1) * (len(interaction) // 5), len(interaction)):
            matys[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1
        for i in range((k) * (len(interaction) // 5), (k + 1) * len(interaction) // 5):
            matys[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1

        zeroindex = np.where(matys == 0)
        zeroindex = np.array(zeroindex)
        zeroindex = zeroindex.T
        np.random.seed(20 + xxx)
        np.random.shuffle(zeroindex)
        zeroindex = zeroindex.T
        neg1 = zeroindex

        mat = np.zeros((267, 570))
        for i in range(0, k * (len(interaction) // 5)):
            mat[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1
        for i in range((k + 1) * (len(interaction) // 5), len(interaction)):
            mat[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1



        zeroindex1 = np.where(mat == 0)
        zeroindex1 = np.array(zeroindex1)
        zeroindex1 = zeroindex1.T
        np.random.seed(20 + xxx)
        np.random.shuffle(zeroindex1)
        zeroindex1 = zeroindex1.T
        neg2 = zeroindex1




        # entity = np.load('rescal_entity_emb.npy')
        # relation = np.load('rescal_relation_emb.npy')
        # entity = np.array(entity)
        # relation = np.array(relation)

        num = 2 * (len(keywords2) + len(interaction) - (len(interaction) // 5))
        epochs = [[100]]
        ws = [1.4]
        adjdps = [0.6]
        dps = [0.3]
        lrs = [0.001]
        gs = [2]
        simws = [1]
        yuzhi = [0.08, 0.05, 0.02, 0.01]
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        for w in ws:
            for epoch in epochs:
                for dp in dps:
                    for adjdp in adjdps:
                        for lr in lrs:
                            result = np.zeros((1, 7), float)
                            average_result = np.zeros((1, 7), float)
                            circle_time = 1
                            for i in range(circle_time):
                                drug_disease_res = Get_embedding_Matrix(mat, xxx, epochs, dp, w, lr, matys, adjdp,
                                                                        num)
                                drug_len = 267
                                dis_len = 570
                                predict = drug_disease_res.reshape(drug_len, dis_len)
        predict = np.array(predict)
        print(predict.shape)
        np.save('predict.npy', predict)

        scoretrue = []
        for i in range((k) * (len(interaction) // 5), (k + 1) * len(interaction) // 5):
            scoretrue.append(predict[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))])
        scorefalse = []
        #aa=random.randint(0,neg1.shape[1]-length)
        aa=0
        for i in range(aa, aa+length):
            scorefalse.append(predict[int(zeroindex[0][i])][int(zeroindex[1][i])])#
            # print(ysbigmat[int(zeroindex[0][i])][int(zeroindex[1][i])])
        scoretrue = np.array(scoretrue)
        scorefalse = np.array(scorefalse)
        print(scoretrue[0:100])
        print(scorefalse[0:100])
        #
        #
        label = np.zeros((scoretrue.shape[0] + scorefalse.shape[0]))
        label[0:scoretrue.shape[0]] = 1
        label = np.array(label)
        allscore = np.concatenate((scoretrue, scorefalse))
        print(allscore.shape)
        result = get_metrics(np.mat(label), np.mat(allscore))
        result = np.array(result)
        lgcnallresult.append(result)
        print(result)

        #aaaa = np.load('rescal_entity_emb.npy')
        aaaa = np.load('TransD_entity_emb.npy')
        # scaler = MinMaxScaler()
        # scaler.fit(aaaa)
        # # scaler.data_max_
        # aaaa = scaler.transform(aaaa)
        aaaa = np.array(aaaa)
        bbbb = np.load('lgcn_em.npy')
        bbbb = np.array(bbbb)
        allembedding = aaaa
        #allembedding1 = bbbb
        drug_embedding = np.zeros((267, 128))
        disease_embedding = np.zeros((570, 128))
        drug_embedding1 = np.zeros((267, 128))
        disease_embedding1 = np.zeros((570, 128))
        for i in range(len(drug)):
            drug_embedding[i] = allembedding[entity2id[drug[i]]]
        for i in range(len(disease)):
            disease_embedding[i] = allembedding[entity2id[disease[i]]]
        print(drug_embedding.shape)
        print(disease_embedding.shape)
        drug_embedding=drug_embedding
        disease_embedding=disease_embedding
        drug_embedding1 = bbbb[0:267]
        disease_embedding1 = bbbb[267:]
        # drug_embedding = np.concatenate((drug_embedding , bbbb[0:267]),axis=1)
        # disease_embedding = np.concatenate((disease_embedding,bbbb[267:]),axis=1)

        mat = np.zeros((267, 570))
        for i in range(0, k * (len(interaction) // 5)):
            mat[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1
        for i in range((k + 1) * (len(interaction) // 5), len(interaction)):
            mat[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1

        # allembedding=aaaa
        # allembedding1=bbbb



        disease_sim = np.load('data//disease_sim.npy')
        #disease_sim=np.ones((570,570))
        print(disease_sim)
        disease_sim=disease_sim-np.eye(570)
        #disease_sim=matrix_normalize(disease_sim)
        #disease_sim = matrix_normalize(disease_sim)
        drug_smiles=np.load('data//drug_smiles.npy')
        drug_smiles = drug_smiles.astype(np.float)
        #drug_smiles=matrix_normalize(drug_smiles)
        #drug_smiles=get_Jaccard2_Similarity(drug_smiles)
        print(11111111111111111111)
        #drug_sim=get_Jaccard2_Similarity(drug_smiles)

        print(drug_smiles.shape)


        # scaler = MinMaxScaler()
        # scaler.fit(disease_sim)
        # # scaler.data_max_
        # disease_sim = scaler.transform(disease_sim)
        disease_sim=np.array(disease_sim)
        #disease_sim=np.ones((570,570))

        # scaler = MinMaxScaler()
        # scaler.fit(drug_smiles)
        # # scaler.data_max_
        # drug_smiles = scaler.transform(drug_smiles)#
        drug_smiles=np.array(drug_smiles)
        # drug_smiles=np.ones((267,128))

        def get_model(num_users, num_items, k, latent_dim, regs=[0, 0], activation_function='hard_sigmoid'):
            # get_custom_objects().update({'vallina_relu': Activation(vallina_relu)})
            # Input variables
            user_input = Input(shape=(1,), dtype='int32', name='user_input')
            user_fea = Input(shape=(k,), dtype='float32', name='user_fea')

            item_input = Input(shape=(1,), dtype='int32', name='item_input')
            item_fea = Input(shape=(570,), dtype='float32', name='item_fea')

            a = drug_embedding
            b = disease_embedding

            a1 = drug_embedding1
            b1 = disease_embedding1
            # all=a+b
            # a=K.constant(a)#
            # b=K.constant(b)##
            item_fea_trans = Dense(128, kernel_initializer='glorot_uniform', activation='relu',use_bias=True)(item_fea)
            #item_fea_trans = keras.layers.core.Dropout(0.4)(item_fea_trans)#####
            item_fea_trans = normalization.BatchNormalization(input_shape=[128], epsilon=1e-6, weights=None)(item_fea)
            #user_fea_trans = Dense(128, kernel_initializer='glorot_uniform', activation='relu',use_bias=True)(user_fea)
            #user_fea_trans = keras.layers.core.Dropout(0.4)(user_fea_trans)
            user_fea_trans = normalization.BatchNormalization(input_shape=[128], epsilon=1e-6, weights=None)(user_fea)
            print(1111111111)
            print(user_fea_trans.shape)

            # Crucial to flatten an embedding vector!
            MF_Embedding_User = Embedding(weights=[a], name='user_embedding', output_dim=128, input_dim=num_users,
                                          input_length=1, trainable=False)
            MF_Embedding_Item = Embedding(weights=[b], name='item_embedding', output_dim=128, input_dim=num_items,
                                          input_length=1, trainable=False)
            MF_Embedding_User1 = Embedding(weights=[a1], name='user_embedding1', output_dim=128, input_dim=num_users,
                                          input_length=1, trainable=False)
            MF_Embedding_Item1 = Embedding(weights=[b1], name='item_embedding1', output_dim=128, input_dim=num_items,
                                          input_length=1, trainable=False)
            # MF_Embedding_User = normalization.BatchNormalization(input_shape=[128], epsilon=1e-6, weights=None)(MF_Embedding_User)
            # MF_Embedding_Item = normalization.BatchNormalization(input_shape=[128], epsilon=1e-6, weights=None)(MF_Embedding_Item)
            user_latent = Flatten()(MF_Embedding_User(user_input))
            item_latent = Flatten()(MF_Embedding_Item(item_input))
            user_latent1 = Flatten()(MF_Embedding_User1(user_input))
            item_latent1 = Flatten()(MF_Embedding_Item1(item_input))#

            user_latent = Dense(128, kernel_initializer='glorot_uniform', activation='relu', use_bias=True)(user_latent)
            user_latent = keras.layers.core.Dropout(0.4)(user_latent)
            item_latent = Dense(128, kernel_initializer='glorot_uniform', activation='relu', use_bias=True)(item_latent)
            item_latent = keras.layers.core.Dropout(0.4)(item_latent)  #

            user_latent1 = Dense(128, kernel_initializer='glorot_uniform', activation='relu', use_bias=True)(
                user_latent1)
            user_latent1 = keras.layers.core.Dropout(0.4)(user_latent1)
            item_latent1 = Dense(128, kernel_initializer='glorot_uniform', activation='relu', use_bias=True)(
                item_latent1)
            item_latent1 = keras.layers.core.Dropout(0.4)(item_latent1)

            # user_fea_trans = Dense(128, kernel_initializer='glorot_uniform', activation='relu', use_bias=True)(user_fea_trans)
            # user_fea_trans = keras.layers.core.Dropout(0.4)(user_latent)
            # user_fea_trans = Dense(128, kernel_initializer='glorot_uniform', activation='relu', use_bias=True)(user_fea_trans)
            # user_fea_trans = keras.layers.core.Dropout(0.4)(user_fea_trans)
            #
            # item_fea_trans = Dense(128, kernel_initializer='glorot_uniform', activation='relu', use_bias=True)(item_fea_trans)
            # item_fea_trans = keras.layers.core.Dropout(0.4)(item_fea_trans)
            # item_fea_trans = Dense(128, kernel_initializer='glorot_uniform', activation='relu', use_bias=True)(item_fea_trans)
            # item_fea_trans = keras.layers.core.Dropout(0.4)(item_fea_trans)
            # user_fea=Flatten()(user_fea)
            # item_fea=Flatten()(item_fea)
            # user_latent = keras.layers.Add()(
            #     [user_fea_trans, user_latent])  # user_fea+user_latent##user_fea+user_latent#
            # item_latent = keras.layers.Add()(
            #     [item_fea_trans, item_latent])  # item_fea+item_latent##item_fea+item_latent#
            # user_latent = Dense(128, kernel_initializer='glorot_normal', activation='relu')(user_latent)
            # user_latent = keras.layers.core.Dropout(0.5)(user_latent)
            # item_latent = Dense(128, kernel_initializer='glorot_normal', activation='relu')(item_latent)
            # item_latent = keras.layers.core.Dropout(0.5)(item_latent)  #

            # user_latent1 = Dense(256, kernel_initializer='glorot_normal', activation='relu')(user_latent1)
            # user_latent1 = keras.layers.core.Dropout(0.4)(user_latent1)
            # item_latent1 = Dense(256, kernel_initializer='glorot_normal', activation='relu')(item_latent1)
            # item_latent1 = keras.layers.core.Dropout(0.4)(item_latenmat1)

            user_latentf=keras.layers.Concatenate()([user_latent,user_latent1])
            item_latentf=keras.layers.Concatenate()([item_latent,item_latent1])
            #user_latentf = user_latent1
            #item_latentf = item_latent1

            # user_latentf = keras.layers.Multiply()([user_latent, user_latent1])
            # item_latentf = keras.layers.Multiply()([item_latent, item_latent1])


            user_item_concat = keras.layers.Concatenate()([user_fea_trans, item_fea_trans, user_latentf, item_latentf])
            #user_item_concat = keras.layers.Concatenate()([user_latentf, item_latentf])
            # att = Dense(latent_dim, kernel_initializer='random_uniform', activation='softmax')(user_item_concat)
            #
            # vec = keras.layers.Multiply()([user_latent, item_latent])
            #
            # predict_vec = keras.layers.Multiply()([vec, att])#

            prediction = Dense(256, kernel_initializer='glorot_uniform', activation='relu',use_bias=True)(user_item_concat)
            prediction = keras.layers.core.Dropout(0.4)(prediction)
            prediction = Dense(128, kernel_initializer='glorot_uniform', activation='relu',use_bias=True)(prediction)
            prediction = keras.layers.core.Dropout(0.4)(prediction)
            prediction = Dense(64, kernel_initializer='glorot_uniform', activation='relu',use_bias=True)(prediction)
            prediction = keras.layers.core.Dropout(0.4)(prediction)
            prediction = Dense(1, kernel_initializer='glorot_uniform', name='prediction', activation='sigmoid',use_bias=True)(prediction)
##
            model = Model(inputs=[user_input, user_fea, item_input, item_fea], outputs=prediction)

            return model


        # trainindex = np.where(mat == 1)
        # trainindex = np.array(trainindex)
        #
        # allembedding = np.concatenate((aaaa, bbbb), axis=1)
        # print(allembedding.shape)

        model = get_model(267, 570, 128, 128, [0, 0], 'hard_sigmoid')
        model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])


        def get_train_instances(train, user_review_fea, item_review_fea):
            user_input, user_fea, item_input, item_fea, labels = [], [], [], [], []
            num_users = train.shape[0]
            # for (u, i) in train.keys():
            #     # positive instance
            #     user_input.append(u)
            #     user_fea.append(user_review_fea[u])
            #     item_input.append(i)
            #     item_fea.append(item_review_fea[i])
            #     label = train[u, i]
            #     labels.append(label)
            for u in range(train.shape[0]):
                for i in range(train.shape[1]):
                    user_input.append(u)
                    user_fea.append(user_review_fea[u])
                    item_input.append(i)
                    item_fea.append(item_review_fea[i])
                    label = train[u, i]
                    labels.append(label)

            return user_input, np.array(user_fea, dtype='float32'), item_input, np.array(item_fea,
                                                                                         dtype='float32'), labels


        def get_test_instances(mat, user_review_fea, item_review_fea, neg1):
            user_input, user_fea, item_input, item_fea, labels = [], [], [], [], []
            mat = np.zeros((267, 570))

            for i in range((k) * (len(interaction) // 5), (k + 1) * len(interaction) // 5):
                mat[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1
            oneindex = np.where(mat == 1)
            oneindex = np.array(oneindex)

            mat = np.zeros((267, 570))
            for i in range(0, k * (len(interaction) // 5)):
                mat[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1
            for i in range((k + 1) * (len(interaction) // 5), len(interaction)):
                mat[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1
            for i in range((k) * (len(interaction) // 5), (k + 1) * len(interaction) // 5):
                mat[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1

            # zeroindex = np.where(mat == 0)
            # zeroindex = np.array(zeroindex)
            # zeroindex = zeroindex.T
            # np.random.seed(20 + xxx)
            # np.random.shuffle(zeroindex)
            # zeroindex = zeroindex.T
            # neg1 = zeroindex
            #aa=random.randint(0, neg1.shape[1]-length)
            neg11 = neg1[:, aa:aa+length]
            # print(oneindex.shape)
            # print(neg1.shape)
            allindex = np.concatenate((oneindex, neg11), axis=1)
            print(allindex.shape)
            for i in range(allindex.shape[1]):
                user_input.append(allindex[0][i])
                user_fea.append(user_review_fea[allindex[0][i]])
                item_input.append(allindex[1][i])
                item_fea.append(item_review_fea[allindex[1][i]])
                label = mat[allindex[0][i], allindex[1][i]]
                labels.append(label)

            return user_input, np.array(user_fea, dtype='float32'), item_input, np.array(item_fea,
                                                                                         dtype='float32'), labels


        user_input, user_fea, item_input, item_fea, labels = get_train_instances(mat, drug_smiles, disease_sim)

        user_input_1, user_fea_1, item_input_1, item_fea_1, labels_1 = get_test_instances(mat, drug_smiles, disease_sim,
                                                                                          neg1)

        # Training
        mat = np.zeros((267, 570))
        for i in range(0, k * (len(interaction) // 5)):
            mat[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1  #
        for i in range((k + 1) * (len(interaction) // 5), len(interaction)):
            mat[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1
        for i in range((k) * (len(interaction) // 5), (k + 1) * len(interaction) // 5):
            mat[drug.index(interaction[i][0])][disease.index(interaction[i][1].replace(",", ''))] = 1
        resultmax = np.zeros((1, 7))
        for epoch in range(300):
            hist = model.fit([np.array(user_input), user_fea, np.array(item_input), item_fea],  # input
                             np.array(labels),  # labels
                             batch_size=6000, epochs=1, verbose=0, shuffle=True, validation_data=(
                [np.array(user_input_1), user_fea_1, np.array(item_input_1), item_fea_1], np.array(labels_1)))
            loss = hist.history
            print(loss)

            if epoch % 1 == 0:
                score = model.evaluate([np.array(user_input_1), user_fea_1, np.array(item_input_1), item_fea_1],
                                       np.array(labels_1))
                # print(score)
                print('loss:%.3f,Accuracy:%.3f' % (score[0], score[1]))
                # dense1_layer_model = Model(inputs=[np.array(user_input_1), user_fea_1, np.array(item_input_1), item_fea_1],
                #                            outputs=model.get_layer('prediction').output)

                dense1_output = model.predict([np.array(user_input_1), user_fea_1, np.array(item_input_1), item_fea_1])
                dense1_output = dense1_output
                dense1_output = np.array(dense1_output)
                dense1_output = np.reshape(dense1_output, (dense1_output.shape[0]))
                # print(dense1_output[0:100])
                # print(dense1_output[-100:])
                # label = np.zeros(dense1_output.shape[0])
                # label[0:testone.shape[0]] = 1
                label=np.array(labels_1)
                result = get_metrics(np.mat(label), np.mat(dense1_output))
                result = np.array(result)
                if epoch == 0:
                    resultmax = result
                else:
                    if result[0] > resultmax[0]:
                        resultmax = result

                print(resultmax)
        pinqilai.append(resultmax)

        xxx = np.zeros((1, 7))
        for i in range(len(lgcnallresult)):
            xxx = xxx + lgcnallresult[i]
        print(xxx / len(lgcnallresult))
        with open('lgcnresult.txt', 'w') as f:
            for i in range(len(lgcnallresult)):
                f.write(str(lgcnallresult[i]) + '\n')
            f.write('all' + str(xxx / len(lgcnallresult)))



        xxx = np.zeros((1, 7))
        for i in range(len(pinqilai)):
            xxx = xxx + pinqilai[i]
        print(xxx / len(pinqilai))
        with open('pinqilai.txt', 'w') as f:
            for i in range(len(pinqilai)):
                f.write(str(pinqilai[i]) + '\n')
            f.write('all' + str(xxx / len(pinqilai)))  #