import collections
import nltk
import article_db
import tensorflow as tf

'''Returns a matrix of the encoding for each word in our vocabulary.'''
stops = { '.',';',',' }        # Improve with NLTK
pars = [nltk.word_tokenize(a.content) for a in article_db.fetch(10)]
corpus = [[w.lower() for w in p if w not in stops] for p in pars]
words = [w for p in corpus for w in p]
counter = collections.Counter(words) # Other data structure.

p_word = 50
p_par = 150
T_word = len(counter)
T_par = len(corpus)

t_c = tf.placeholder(tf.float32, shape=[T_word,1])       # 1-hot vector for central word
t_word = tf.placeholder(tf.float32, shape=[T_word,None]) # 1-hot matrix for words
t_par = tf.placeholder(tf.float32, shape=[T_par,1])      # 1-hot vector for pars

# NPLM parameters
W = tf.Variable(tf.random_uniform([p_word,T_word], -1.0, 1.0))
D = tf.Variable(tf.random_uniform([p_par,T_par], -1.0, 1.0))

# Context vectors 
h_word = tf.matmul(W,t_word)
h_word = tf.reduce_mean(h_word,1) # Take the average of context words.
h_par = tf.matmul(D,t_par)
h_par = h_par[:,1]
h = tf.concat(0,[h_word,h_par])

# Softmax sizes
p = p_word + p_par
T = T_word

h = tf.reshape(h,[p,1])

# Softmax parameters
U = tf.Variable(tf.zeros([T, p]))
b = tf.Variable(tf.zeros([T,1]))

y = tf.nn.softmax(b + tf.matmul(U,h))
cost = -tf.reduce_sum(t_c * tf.log(y))

# train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

# init = tf.initialize_all_variables()

# sess = tf.Session()
# sess.run(init)
print("hello")
