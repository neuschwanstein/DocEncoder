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

def stochastic_batch(k, batch_size=200):
    n = batch_size
    t_ps = [random.randrange(T_p) for _ in range(n)]
    cs = [random.randrange(k, len(ps[t_p])-k) for t_p in t_ps]
    t_cs = [ts[ps[t_p][c]] for c,t_p in zip(cs,t_ps)]
    t_ws = [[ts[w] for w in ps[t_p][c-k:c]+ps[t_p][c+1:c+k+1]] for c,t_p in zip(cs,t_ps)]

    return t_ps, t_ws, t_cs

stochastic_batch(2,5)

q_w = 5
q_p = 6

T_w = 12
T_p = 10

n = 3

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

feed_dict = { t_ps: [0,1,2], t_ws: [[1,2,3,4],[2,3,4,5],[3,4,5,6]], t_cs: [11,10,9] }
# print("h_ps=\n",sess.run(h_ps, feed_dict))
# print("W=",sess.run(W))
# print("h_ws=\n",sess.run(h_ws, feed_dict))
print('h=\n',sess.run(h, feed_dict))
print('y=\n',sess.run(y, feed_dict))


print("Hello")





# def old_stochastic_batch(k_around, batch_size=1):
#     # Paragraph 1-hot vector
#     p = random.randrange(T_p)
#     t_p = [0] * T_p
#     t_p[p] = 1
#     t_p = np.asarray(t_p).reshape(T_p,1)

#     par = pars[p]
#     l = len(par)

#     # Center 1-hot vector
#     c_pos = random.randrange(k_around, l-k_around)
#     c = par[c_pos]
#     t_c = [0] * T_w
#     t_c[word_dict[c]] = 1
#     t_c = np.asarray(t_c).reshape(T_w,1)

#     # Surrounding words 1-hot matrix
#     ws = par[c_pos-k_around:c_pos] + par[c_pos+1:c_pos+k_around+1]
#     t_ws = [[0] * (2*k_around) for _ in range(T_w)]
#     for j,i in enumerate([word_dict[w] for w in ws]):
#         t_ws[i][j] = 1
        
#     return t_par, t_c, t_ws

# def full_softmax_classifier(q_w,q_p):

#     t_c = tf.placeholder(tf.float32, shape=[T_w,1], name='t_c')       # 1-hot vector for central word
#     t_ws = tf.placeholder(tf.float32, shape=[T_w,None], name='t_ws') # 1-hot matrix for words
#     t_par = tf.placeholder(tf.float32, shape=[T_p,1], name='t_par')      # 1-hot vector for pars
    
#     # NPLM parameters
#     W = tf.Variable(tf.random_uniform([q_w,T_w], -1.0, 1.0))
#     D = tf.Variable(tf.random_uniform([q_p,T_p], -1.0, 1.0))
    
#     # Context vectors 
#     h_word = tf.matmul(W,t_ws)
#     h_word = tf.reduce_mean(h_word,1) # Take the average of context words.
#     h_par = tf.matmul(D,t_par)
#     h_par = tf.reshape(h_par,[q_p])
#     h = tf.concat(0,[h_word,h_par])
    
#     # Softmax sizes
#     p = q_w + q_p
#     T = T_w
    
#     h = tf.reshape(h,[p,1])
    
#     # Softmax parameters
#     U = tf.Variable(tf.zeros([T, p]))
#     b = tf.Variable(tf.zeros([T,1]))
    
#     y = tf.nn.softmax(b + tf.matmul(U,h))
#     cost = -tf.reduce_sum(t_c * tf.log(y))
    
#     train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

#     init = tf.initialize_all_variables()
#     sess = tf.Session()
#     sess.run(init)

#     for i in range(100):
#         batch_t_par, batch_t_c, batch_t_ws = stochastic_batch(3)
#         feed_dict = { t_par: batch_t_par, t_c: batch_t_c, t_ws: batch_t_ws }
#         # sess.run(train, feed_dict)
#         sess.run(h, feed_dict)

#     print("Done.")
#     return sess.run(D)

# # D = full_softmax_classifier(q_w=50, q_p=150)
