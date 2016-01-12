import collections
import nltk
import article_db
import tensorflow as tf

'''Returns a matrix of the encoding for each word in our vocabulary.'''
stops = { '.',';',',' }        # Improve with NLTK
corpus_pars = [nltk.word_tokenize(a.content) for a in article_db.fetch(10)]
corpus = [w.lower() for p in corpus_pars for w in p if w not in stops]
counter = collections.Counter(corpus)

p = 100
T = len(counter)

h = tf.placeholder(tf.float32, shape=[p,None])
y_hot = tf.placeholder(tf.float32, shape=[T,None])

U = tf.Variable(tf.zeros([T,p]))
b = tf.Variable(tf.zeros([T]))

y = tf.nn.softmax(tf.matmul(U,h))
cost = -tf.reduce_sum(y_hot * tf.log(y))

train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
