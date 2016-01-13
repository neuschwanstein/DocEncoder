import collections
import nltk
import article_db
import tensorflow as tf
import random
import numpy as np

'''Returns a matrix of the encoding for each word in our vocabulary.'''
stops = { '.',';',',' }        # Improve with NLTK
ps = [nltk.word_tokenize(a.content) for a in article_db.fetch(10)]
ps = [[w.lower() for w in p if w not in stops] for p in ps]
words = [w for p in ps for w in p]
counter = collections.Counter(words) # Other data structure?

# Assign an id to each word
ts = {w: i for i,w in enumerate(counter)}

T_w = len(ts)
T_p = len(ps)

def stochastic_batch(k, n=200):
    t_ps = [random.randrange(T_p) for _ in range(n)]
    cs = [random.randrange(k, len(ps[t_p])-k) for t_p in t_ps]
    t_cs = [ts[ps[t_p][c]] for c,t_p in zip(cs,t_ps)]
    t_ws = [[ts[w] for w in ps[t_p][c-k:c]+ps[t_p][c+1:c+k+1]] for c,t_p in zip(cs,t_ps)]

    return t_ps, t_ws, t_cs

q_w = 5
q_p = 6
n = 200

# NPLM parameters
W = tf.Variable(tf.random_uniform([T_w, q_w], -1.0, 1.0))
D = tf.Variable(tf.random_uniform([T_p, q_p], -1.0, 1.0))

t_cs = tf.placeholder(tf.int32, shape=[n])
t_ps = tf.placeholder(tf.int32, shape=[n])
t_ws = tf.placeholder(tf.int32, shape=[n,None])

# Context vector (from surrounding ws and p)
h_ps = tf.gather(D, t_ps)
h_ws = tf.gather(W, t_ws)
h_ws = tf.reduce_mean(h_ws,1)   # `mix' context words (average in this case)
h = tf.concat(1, [h_ps,h_ws])

# Softmax parameters
U = tf.Variable(tf.zeros([q_w+q_p,T_w]))
b = tf.Variable(tf.zeros([T_w]))

y = tf.nn.softmax(tf.matmul(h,U) + b)
y = tf.gather(tf.transpose(y), t_cs)
y = tf.pack([y[i,i] for i in range(n)]) # *BIG* hack.

cost = -tf.reduce_sum(y)
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

keys = [t_ps, t_ws, t_cs]
for i in range(10):
    feed = { k: val for k,val in zip(keys,stochastic_batch(k=2,n=n)) }
    sess.run(train, feed)


print("Hello")
