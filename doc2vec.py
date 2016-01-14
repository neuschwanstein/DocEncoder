import collections
import nltk
import article_db
import tensorflow as tf
import random
import numpy as np

global ps,ws,ts,T_w,T_p

def stochastic_batch(k, n=200):
    t_ps = [random.randrange(T_p) for _ in range(n)]
    cs = [random.randrange(k, len(ps[t_p])-k) for t_p in t_ps]
    t_cs = [ts[ps[t_p][c]] for c,t_p in zip(cs,t_ps)]
    t_ws = [[ts[w] for w in ps[t_p][c-k:c]+ps[t_p][c+1:c+k+1]] for c,t_p in zip(cs,t_ps)]

    return t_ps, t_ws, t_cs


def doc2vec(q_w, q_p, batch_size=200, db_limit=100):
    # Initialization
    stops = { '.',';',',' }        # Improve with NLTK
    ps = [nltk.word_tokenize(a.content) for a in article_db.fetch(10)]
    ps = [[w.lower() for w in p if w not in stops] for p in ps]
    ws = [w for p in ps for w in p]
    ts = {w: i for i,w in enumerate(collections.Counter(ws))} # word -> id

    T_w = len(ts)
    T_p = len(ps)

    # NPLM parameters
    W = tf.Variable(tf.random_uniform([T_w, q_w], -1.0, 1.0))
    D = tf.Variable(tf.random_uniform([T_p, q_p], -1.0, 1.0))
    D_bow = tf.Variable(tf.random_uniform([T_p,q_p], -1.0, 1.0))

    t_cs = tf.placeholder(tf.int32, shape=[n])
    t_ps = tf.placeholder(tf.int32, shape=[n])
    t_ws = tf.placeholder(tf.int32, shape=[n,None])

    # Context vector (from surrounding ws and p)
    h_ps = tf.gather(D, t_ps)
    h_ws = tf.gather(W, t_ws)
    h_ws = tf.reduce_mean(h_ws,1)   # `mix' context words (average in this case)
    h = tf.concat(1, [h_ps,h_ws])

    # PVBOW vector
    h_bow = tf.gather(D_bow, t_ps)

    # Softmax parameters for PVDM movdel
    U = tf.Variable(tf.zeros([q_w+q_p,T_w]))
    b = tf.Variable(tf.zeros([T_w]))

    # Softmax paramaters for PVDBOW model
    U_bow = tf.Variable(tf.zeros([q_p,T_w]))
    b_bow = tf.Variable(tf.zeros([T_w]))

    y = tf.nn.softmax(tf.matmul(h,U) + b) # Perform softmax with params (U,b) from `context' h
    y = tf.gather(tf.transpose(y), t_cs)  # Evaluate the probabilities of target t_cs of each example
    y = tf.pack([y[i,i] for i in range(n)]) # n. This last line is a big hack... Todo find better sol.

    y_bow = tf.nn.softmax(tf.matmul(h_bow,U_bow) + b_bow)
    y_bow = tf.gather(tf.transpose(y_bow), t_cs)
    y_bow = tf.pack([y_bow[i,i] for i in range(n)])

    cost = -tf.reduce_sum(y)
    cost_bow = -tf.reduce_sum(y_bow)

    train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    train_bow = tf.train.GradientDescentOptimizer(0.1).minimize(cost_bow)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(100):
        feed = dict(zip([t_ps, t_ws, t_cs], stochastic_batch(k=4,n=n)))
        sess.run([train,train_bow], feed)

    print("DONE")
