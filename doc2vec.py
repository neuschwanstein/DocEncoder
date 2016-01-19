import math
import collections
import random

import nltk
import tensorflow as tf
import numpy as np
from itertools import groupby
from more_itertools import unique_everseen

import articles

def doc2vec(q_w, q_p, batch_size=200, steps=10000, k=12, db_limit=100):
    global ps,ts,T_w,T_p,n
    k = 5

    ps,ts = initialize(k,db_limit)
    T_w = len(ts)
    T_p = len(ps)

    feed_bow = stochastic_batch_bow(k,n=1)

    # Distributed Memory parameters
    W = tf.Variable(tf.random_uniform([T_w, q_w], -1.0, 1.0))
    D = tf.Variable(tf.random_uniform([T_p, q_p], -1.0, 1.0))

    # Index placeholders
    t_cs = tf.placeholder(tf.int32, shape=[n])
    t_ps = tf.placeholder(tf.int32, shape=[n])
    t_ws = tf.placeholder(tf.int32, shape=[n,None])

    # Context vector (from surrounding ws and p)
    h_ps = tf.gather(D, t_ps)
    h_ws = tf.gather(W, t_ws)
    h_ws = tf.reduce_mean(h_ws,1)   # `mix' context words (average in this case)
    h = tf.concat(1, [h_ps,h_ws])

    # Softmax parameters for PVDM movdel
    U = tf.Variable(tf.zeros([q_w+q_p,T_w]))
    b = tf.Variable(tf.zeros([T_w]))

    mask = tf.diag(tf.ones([n]))
    y = logsoftmax(tf.matmul(h,U) + b)
    y = tf.gather(tf.transpose(y), t_cs)  # Evaluate the probabilities of target t_cs of each example
    y = tf.mul(mask,y)                    # Mask elements off diagonal

    cost_dmm = -tf.reduce_sum(y)
    train_dmm = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    # Distributed bag of words model
    D_bow = tf.Variable(tf.random_uniform([T_p,q_p], -1.0, 1.0))

    # Index placeholders
    t_ps_bow = tf.placeholder(tf.int32, shape=[n])
    t_cs_bow = tf.placeholder(tf.int32, shape=[n,None]) # None bcz len(par) might be shorter
    mask_1_bow = tf.placeholder(tf.int32, shape=[None]) # One big hack: array of one of undefined length
    
    # PVBOW vector
    h_bow = tf.gather(D_bow, t_ps)

    # Softmax paramaters for PVDBOW model
    U_bow = tf.Variable(tf.zeros([q_p,T_w]))
    b_bow = tf.Variable(tf.zeros([T_w]))

    mask_bow = tf.SparseTensor(values=mask_1_bow, indices=t_cs_bow, shape=tf.shape(y_bow))
    y_bow = logsoftmax(tf.matmul(h_bow,U_bow) + b_bow)
    y_bow = tf.mul(mask_bow,y_bow)

    cost_bow = -tf.reduce_sum(y_bow)
    train_bow = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        print("Beginning SGD operation...")
        for i in range(steps):
            # PVDM training
            feed_dmm = dict(zip(
                [t_ps,t_ws,t_cs,t_cs_bow], stochastic_batch_dmm(k,n)))
            new_cost_dmm,_ = sess.run([cost_dmm,train_dmm],feed_dmm)
            
            # BOW Training
            feed_bow = dict(zip(
                [t_ps_bow,mask_1_bow,t_cs_bow], stochastic_batch_bow(k,n)))
            new_cost_bow,_ = sess.run([cost_bow,train_bow],feed_bow)
            
            # print("Average probability %d: %f (%f)" % (i,np.exp(-cost_val/n),cost_val))
            print("New costs %d: %f,%f" % (i, new_cost_dmm, new_cost_bow))

            if math.isnan(new_cost_dmm) or math.isnan(new_cost_bow):
                raise ValueError
        
    print("DONE")

def initialize(k, db_limit):
    stops = { '.',';',',' }        # Improve with NLTK
    bof = ['__BOF__'] * k          # Padding tokens
    eof = ['__EOF__'] * k
    
    ps = [nltk.word_tokenize(a.content) for a in articles.fetch(db_limit)]
    ps = [bof + [w.lower() for w in p if w not in stops] + eof
          for p in ps if len(p)]
    ws = [w for p in ps for w in p]
    ts = {w: i for i,w in enumerate(collections.Counter(ws))} # word -> id
    return ps,ts

def stochastic_batch_dmm(k, n=200):
    t_ps = [random.randrange(T_p) for _ in range(n)]
    cs = [random.randrange(k, len(ps[t_p])-k) for t_p in t_ps]
    t_cs = [ts[ps[t_p][c]] for c,t_p in zip(cs,t_ps)]
    t_ws = [[ts[w] for w in ps[t_p][c-k:c]+ps[t_p][c+1:c+k+1]] for c,t_p in zip(cs,t_ps)]
    return t_ps, t_ws, t_cs

def stochastic_batch_bow(k, n=200):
    '''Returns 2k examples (possibly with replacement) for each of the n paragraphs.  Each of
    these 2k results are then sorted and `merged'. The number of merged values are then
    kept in the mask value. t_css assign for each paragraph of t_ps a list of target
    words.

    '''
    t_ps = [random.randrange(T_p) for _ in range(n)]
    max_ls = [len(ps[t_p])-2*k if len(ps[t_p]) >= 4*k else 2*k for t_p in t_ps]
    css = [[random.randrange(k,l+k) for _ in range(2*k)] for l in max_ls]
    t_css = [[ts[ps[t_p][c]] for c in cs] for cs,t_p in zip(css,t_ps)]
    counters = [collections.Counter(sorted(t_cs)) for t_cs in t_css]
    mask = [list(counter.keys()) for counter in counters]
    t_css = [list(counter.values()) for counter in counters]
    return t_ps,mask,t_css

def logsoftmax(M):
    '''LSE(v) = log(exp(v1) + ... + exp(vn))
              = v_s + log(exp(v1-vs) + ... + exp(vn - vs))
    where v_s = max(v1,...,vn)
    Do this for each column (hence the reshape to correctly broadcast)

    '''
    x_star = tf.reshape(tf.reduce_max(M,1), [n,1])
    logsumexp = tf.reshape(tf.log(tf.reduce_sum(tf.exp(M-x_star),1)),[n,1]) + x_star
    return M - logsumexp    
