


import copy
import torch
from torch import nn
import os
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


from pydoc import locate

CACHE_SIZE=100
LR = 0.001  
BATCH_SIZE=32

def lruPredict(C,LRUQ,Y_OPT):
    global lruCorrect, lruIncorrect
    Y_current = []
    KV = defaultdict(int)
    for i in range(len(LRUQ)):
        KV[LRUQ[i]] = len(LRUQ) - i
    KV_sorted = Counter(KV)
    evict_dict = dict(KV_sorted.most_common(eviction))
    for e in C:
        if e in evict_dict:
            Y_current.append(1)
        else:
            Y_current.append(0)
    for i in range(len(Y_current)):
        if Y_current[i] is Y_OPT[i]:
            lruCorrect+=1
        else:
            lruIncorrect+=1
    return Y_current

# returns sequence of blocks in prioirty order

def Y_getBlockSeq(Y_pred_prob):
    x = []
    for i in range(len(Y_pred_prob)):
        x.append(Y_pred_prob[i][0])
    x = np.array(x)
    idx = np.argsort(x)
    idx = idx[:eviction]
    return idx


def Y_getMinPredict(Y_pred_prob):
    x = []
    for i in range(len(Y_pred_prob)):
        x.append(Y_pred_prob[i][0])
    x = np.array(x)
    idx = np.argpartition(x, eviction)
    
    Y_pred = np.zeros(len(Y_pred_prob), dtype=int)
    for i in range(eviction):
        Y_pred[idx[i]] = 1
    assert(Counter(Y_pred)[1] == eviction)
    return Y_pred


def lfuPredict(C,LFUDict,Y_OPT):
    global lfuCorrect, lfuIncorrect
    Y_current = []
    KV = defaultdict()
    for e in C:
        KV[e] = LFUDict[e]
    KV_sorted = Counter(KV)
    evict_dict = dict(KV_sorted.most_common(eviction))
    for e in C:
        if e in evict_dict:
            Y_current.append(1)
        else:
            Y_current.append(0)
    for i in range(len(Y_current)):
        if Y_current[i] is Y_OPT[i]:
            lfuCorrect+=1
        else:
            lfuIncorrect+=1
    return Y_current

# return "eviction" blocks that are being accessed furthest
# from the cache that was sent to us.

def getY(C,D):
    assert(len(C) == len(D))
    Y_current = []
    KV_sorted = Counter(D)
    evict_dict = dict(KV_sorted.most_common(eviction))
    assert(len(evict_dict) == eviction)
    all_vals = evict_dict.values()
    for e in C:
        if e in evict_dict.values():
            Y_current.append(1)
        else:
            Y_current.append(0)
    #print (Y_current.count(1))
    assert(Y_current.count(1) == eviction)
    assert((set(all_vals)).issubset(set(C)))
    return Y_current

def getLFURow(LFUDict, C):
    x_lfurow = []
    for e in C:
        x_lfurow.append(LFUDict[e])
    norm = x_lfurow / np.linalg.norm(x_lfurow)
    return norm
    
def getLRURow(LRUQ, C):
    x_lrurow = []
    KV = defaultdict(int)
    for i in range(len(LRUQ)):
        KV[LRUQ[i]] = i
    for e in C:
        x_lrurow.append(KV[e])
    norm = x_lrurow / np.linalg.norm(x_lrurow)
    return norm

def normalize(feature, blocks):
    x_feature = []
    for i in range(len(blocks)):
        x_feature.append(feature[blocks[i]])
    return x_feature / np.linalg.norm(x_feature)

def getX(LRUQ, LFUDict, C):
#def getX(LRUQ, LFUDict, C, CacheTS, CachePID):   
    X_lfurow = getLFURow(LFUDict, C)
    X_lrurow = getLRURow(LRUQ, C)
    X_bno    = C / np.linalg.norm(C)
#     X_ts     = normalize(CacheTS, C)
#     X_pid    = normalize(CachePID, C)
    return (np.column_stack((X_lfurow, X_lrurow, X_bno)))
    
    
def populateData(LFUDict, LRUQ, C, D):
#def populateData(LFUDict, LRUQ, C, D, CacheTS, CachePID):
    global X,Y
    C = list(C)
    Y_current = getY(C, D)
    #X_current = getX(LRUQ, LFUDict, C, CacheTS, CachePID)
    X_current = getX(LRUQ, LFUDict, C)
    Y = np.append(Y, Y_current)
    X = np.concatenate((X,X_current))
    assert(Y_current.count(1) == eviction)
    return Y_current

def hitRate(blocktrace, frame, model):
    LFUDict = defaultdict(int)
    LRUQ = []
#     CacheTS = defaultdict(int)
#     CachePID = defaultdict(int)

    hit, miss = 0, 0

    C = []
    evictCacheIndex = np.array([])
    #count=0
    #seq_number = 0
    for seq_number, block in enumerate(tqdm(blocktrace, desc="OPT")):
        #print(len(evictCacheIndex))
        LFUDict[block] +=1
        #CacheTS[blocktrace[seq_number]] = timestamp[seq_number]
        #CachePID[blocktrace[seq_number]] = pid[seq_number]
        if block in C:
            hit+=1
#             if C.index(block) in evictCacheIndex:
#                 np.delete(evictCacheIndex, C.index(block))
                
            LRUQ.remove(block)
            LRUQ.append(block)
        else:
            evictPos = -1
            miss+=1
            if len(C) == frame:
                if len(evictCacheIndex) == 0: # call eviction candidates
                    X_test = getX(LRUQ, LFUDict, C)
                    #X_test = getX(LRUQ, LFUDict, C, CacheTS, CachePID)
                    Y_pred_prob = model(torch.FloatTensor(X_test))
                    # index of cache blocks that should be removed
                    evictCacheIndex = Y_getBlockSeq(Y_pred_prob)
                    #return Y_pred_prob, evictCacheIndex
                # evict from cache
                evictPos = evictCacheIndex[0]
                evictBlock = C[evictPos]
                LRUQ.remove(evictBlock)
                #del CacheTS [evictBlock]
                #del CachePID [evictBlock]
            if evictPos is -1:
                C.append(block)
            else:
                C[evictPos] = block
                evictCacheIndex = np.delete(evictCacheIndex, 0)
            LRUQ.append(block)
            #CacheTS [blocktrace[seq_number]] = timestamp[seq_number]
            #CachePID [blocktrace[seq_number]] = pid[seq_number]
        #seq_number += 1

    hitrate = hit / (hit + miss)
    print(hitrate)
    return hitrate

def hitRate2(blocktrace, frame, model):
    LFUDict = defaultdict(int)
    LRUQ = []
#     CacheTS = defaultdict(int)
#     CachePID = defaultdict(int)

    hit, miss = 0, 0

    C = []
    evictCacheIndex = np.array([])
    #count=0
    #seq_number = 0
    for seq_number, block in enumerate(tqdm(blocktrace, desc="OPT")):
        #print(len(evictCacheIndex))
        LFUDict[block] +=1
        #CacheTS[blocktrace[seq_number]] = timestamp[seq_number]
        #CachePID[blocktrace[seq_number]] = pid[seq_number]
        if block in C:
            hit+=1
#             if C.index(block) in evictCacheIndex:
#                 np.delete(evictCacheIndex, C.index(block))
                
            LRUQ.remove(block)
            LRUQ.append(block)
        else:
            evictPos = -1
            miss+=1
            if len(C) == frame:
                if len(evictCacheIndex) == 0: # call eviction candidates
                    #X_test = getX(LRUQ, LFUDict, C)
                    #X_test = getX(LRUQ, LFUDict, C, CacheTS, CachePID)
                    blockNo = C / np.linalg.norm(C)
                    recency_ = np.array([LRUQ.index(i) for i in C])
                    recency_ = recency_ / np.linalg.norm(recency_)
                    frequency_ = np.array([LFUDict[i] for i in C])
                    frequency_ = frequency_ / np.linalg.norm(frequency_)
#                     stack = np.column_stack((blockNo, recency_, frequency_)).reshape(1,frame*3)
                    stack = np.column_stack((blockNo, recency_, frequency_))
                    #X_current = model.predict(stack)[0]
                    Y_pred_prob = model(torch.FloatTensor(stack).unsqueeze(0))
                    evictCacheIndex = np.argsort(Y_pred_prob[0].detach().numpy())[::-1][:eviction]
                    # index of cache blocks that should be removed
                    #return Y_pred_prob, evictCacheIndex
                # evict from cache
                evictPos = evictCacheIndex[0]
                evictBlock = C[evictPos]
                LRUQ.remove(evictBlock)
                #del CacheTS [evictBlock]
                #del CachePID [evictBlock]
            if evictPos is -1:
                C.append(block)
            else:
                C[evictPos] = block
                evictCacheIndex = np.delete(evictCacheIndex, 0)
            LRUQ.append(block)
            #CacheTS [blocktrace[seq_number]] = timestamp[seq_number]
            #CachePID [blocktrace[seq_number]] = pid[seq_number]
        #seq_number += 1

    hitrate = hit / (hit + miss)
    print(hitrate)
    return hitrate

import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm 
from collections import Counter, deque, defaultdict
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier


test = "cheetah.cs.fiu.edu-110108-113008.4.blkparse"

df = pd.read_csv(test, sep=' ',header = None)
df.columns = ['timestamp','pid','pname','blockNo', \
              'blockSize', 'readOrWrite', 'bdMajor', 'bdMinor', 'hash']

testBlockTrace = df['blockNo'].tolist()

testBlockTrace = testBlockTrace[:int(len(testBlockTrace)*0.1)]

len(testBlockTrace)

sampling_freq = CACHE_SIZE # number of samples skipped
eviction = int(0.7 * CACHE_SIZE)  



CNN = 1
NN = 2

exps = [
# ['exp_cnn',CNN],
# ['exp_cnn2',CNN],
# ['exp_cnn3',CNN],
# ['exp_cnn4',CNN],
# ['exp_cnn5',CNN],
# ['exp_cnn6',CNN],
['exp_nn',NN],
['exp_nn2',NN],
]


with open('result','a') as f:
	for exp in exps:
		pth = exp[0]+'/model.pth'
		model_class = exp[0]+'.model.CNN'
		my_class = locate(model_class)
		model = my_class()
		model.load_state_dict(torch.load(pth,map_location='cpu'))
		model.eval()
		if exp[1] == NN:
			for i in [10,100,200,400,600,800,1000]:
				Hitrate = hitRate(testBlockTrace, i, model)
				f.write('Fully connected NN, %f with cache size = %d\n' % (Hitrate, i))
			f.flush()
		else:
			Hitrate = hitRate2(testBlockTrace, 100, model)
			f.write('Convolutional NN, %f with cache size = %d\n' % (Hitrate, 100))
			f.flush()




