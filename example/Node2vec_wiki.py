import math
import os
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tqdm import trange
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from model.RandomWalker import RandomWalker
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error

num_walks = 80    #序列数量
walk_length = 10  #序列长度
workers = 4

def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()

# embedding相似度计算
# def embedding_similarity(embedding1, embedding2):
    # embedding1 = tf.nn.l2_normalize(embedding1, axis=1)
    # embedding2 = tf.nn.l2_normalize(embedding2, axis=1)
    # plt.scatter(embedding1[:,0], embedding1[:,1],c='b')
    # plt.scatter(embedding2[:,0],embedding2[:,1], c='g')
    # plt.show()
    # sim_matrix = tf.nn.sigmoid(tf.matmul(embedding1, tf.transpose(embedding2)));
    # loss_simi = tf.reduce_mean(tf.square(sim_matrix))
    # return loss_simi
    ##########################################
    # model_TSNE = TSNE(n_components=2)
    # node_pos1 = model_TSNE.fit_transform(embedding1)
    # node_pos2 = model_TSNE.fit_transform(embedding2)
    # plt.scatter(node_pos1[:,0],node_pos1[:,1])
    # plt.scatter(node_pos2[:,0],node_pos2[:,1])
    # plt.show()
    # norm1 = np.linalg.norm(embedding1, axis=-1, keepdims=True)
    # norm2 = np.linalg.norm(embedding2, axis=-1, keepdims=True)
    # dot1 = np.dot(embedding1, embedding2.T)
    # dot2 = np.dot(norm1, norm2.T)
    # cos = dot1/dot2
    # # # embedding1_norm = embedding1/norm1
    # # # embedding2_norm = embedding2/norm2
    # # # cos = np.dot(embedding1_norm, embedding2_norm.T)
    # # cos[cos < 0] = -1
    # # cos[cos > 0] = 1
    # sim = tf.reduce_mean(tf.square(cos))
    # if sim>0.9:
    #     return sim
    # else:
    #     return sim+0.2
    ###########################################

graphList = ["community_1935594.txt","community_1944993.txt","community_1782816.txt","community_2199413.txt","community_1767325.txt","community_1935607.txt","community_2199319.txt"]
embeddingList = []


for graphName in graphList:
    G = nx.read_edgelist("../data/"+graphName, nodetype=None, create_using=nx.DiGraph(),
                         data=(('weight', int)))
    # 根据RandomWalker产生序列
    rw = RandomWalker(G, p=1, q=1, use_rejection_sampling=0)
    rw.preprocess_transition_probs()
    sentences = rw.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    model = Word2Vec(sentences=sentences,
                     vector_size=128,
                     min_count=5,
                     sg=1,
                     hs=0,
                     workers=workers,
                     window=5,
                     epochs=3)

    # 获取embedding dict
    # embedding_dict = {}
    embedding_list = []
    # model_TSNE = TSNE(n_components=2)
    for word in G.nodes():
        # embedding_dict[word] = model.wv[word]
        embedding_list.append(model.wv[word])
    embedding_list = np.array(embedding_list).flatten().tolist()
    embedding_list_pad = np.pad(embedding_list,(0,55936-len(embedding_list)),'constant')
    embeddingList.append(embedding_list_pad.tolist())
    print(graphName+" embedding finish")

neigh = NearestNeighbors(n_neighbors=7).fit(np.array(embeddingList))
distance, indices = neigh.kneighbors(np.array([embeddingList[6]]))
print(distance)
print(indices)
# print(embedding_similarity(embeddingList[1], embeddingList[1]))

# model = TSNE(n_components=2)
# node_pos0 = model.fit_transform(embeddingList[0])
# node_pos1 = model.fit_transform(embeddingList[1])
# node_pos2 = model.fit_transform(embeddingList[2])
# node_pos3 = model.fit_transform(embeddingList[3])
# plt.scatter(node_pos0[:,0],node_pos0[:,1])
# plt.scatter(node_pos1[:,0],node_pos1[:,1])
# plt.scatter(node_pos2[:,0],node_pos2[:,1])
# plt.scatter(node_pos3[:,0],node_pos3[:,1])
# plt.show()

# with open("../embeddings/community1998099.txt", 'w', encoding='utf-8') as f:
#     for key, value in embedding_dict.items():
#         f.write(key)
#         f.write(':')
#         f.write(str(value))
#         f.write('\n')

# plot_embeddings(embedding_dict)