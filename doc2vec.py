import math
import collections
import nltk
import article_db
import tensorflow as tf
import random
import numpy as np

def stochastic_batch(k, n=200):
    t_ps = [random.randrange(T_p) for _ in range(n)]
    cs = [random.randrange(k, len(ps[t_p])-k) for t_p in t_ps]
    t_cs = [ts[ps[t_p][c]] for c,t_p in zip(cs,t_ps)]
    t_ws = [[ts[w] for w in ps[t_p][c-k:c]+ps[t_p][c+1:c+k+1]] for c,t_p in zip(cs,t_ps)]

    return t_ps, t_ws, t_cs

def logsoftmax(M):
    # See https://en.wikipedia.org/wiki/LogSumExp
    # LSE(v) = log(exp(v1) + ... + exp(vn))
    # LSE(v) = v_s + log(exp(v1-vs) + ... + exp(vn - vs))
    # where v_s = max(v1,...,vn)
    # Do this for each column (hence the reshape to correctly broadcast)
    x_star = tf.reshape(tf.reduce_max(M,1), [n,1])
    logsumexp = tf.reshape(tf.log(tf.reduce_sum(tf.exp(M-x_star),1)),[n,1]) + x_star
    return M - logsumexp

def doc2vec(q_w, q_p, batch_size=200, steps=10000, db_limit=100):
    global ps,ws,ts,T_w,T_p,n

    # Initialization
    print("Initializing computation variables...")
    n = batch_size
    stops = { '.',';',',' }        # Improve with NLTK
    ps = [nltk.word_tokenize(a.content) for a in article_db.fetch(db_limit)]
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

    mask = tf.diag(tf.ones([n]))

    # Softmax paramaters for PVDBOW model
    U_bow = tf.Variable(tf.zeros([q_p,T_w]))
    b_bow = tf.Variable(tf.zeros([T_w]))

    # y = tf.nn.softmax(tf.matmul(h,U) + b) # Perform softmax with params (U,b) from `context' h
    y = logsoftmax(tf.matmul(h,U) + b)
    y = tf.gather(tf.transpose(y), t_cs)  # Evaluate the probabilities of target t_cs of each example
    # y = tf.log(y)
    y = tf.mul(mask,y)                    # Mask elements off diagonal

    # y_bow = tf.nn.softmax(tf.matmul(h_bow,U_bow) + b_bow)
    y_bow = logsoftmax(tf.matmul(h_bow,U_bow) + b_bow)
    y_bow = tf.gather(tf.transpose(y_bow), t_cs)
    # y_bow = tf.log(y_bow)
    y_bow = tf.mul(mask,y_bow)

    cost = -tf.reduce_sum(y)
    cost_bow = -tf.reduce_sum(y_bow)

    train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    train_bow = tf.train.GradientDescentOptimizer(0.1).minimize(cost_bow)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        print("Beginning SGD operation...")
        avg_cost = np.array([0.,0.])
        for i in range(steps):
            feed = dict(zip([t_ps, t_ws, t_cs], stochastic_batch(k=4,n=n)))
            sess.run([train,train_bow],feed)
            cost_val,cost_bow_val = sess.run([cost,cost_bow], feed)

            print("New cost %d: (%f,%f)" % (i,cost_val,cost_bow_val))
            if math.isnan(cost_val) or math.isnan(cost_bow_val):
                raise ValueError
        
    print("DONE")
