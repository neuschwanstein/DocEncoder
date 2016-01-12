import collections
import nltk
import article_db
import tensorflow as tf
import random

'''Returns a matrix of the encoding for each word in our vocabulary.'''
stops = { '.',';',',' }        # Improve with NLTK
pars = [nltk.word_tokenize(a.content) for a in article_db.fetch(10)]
pars = [[w.lower() for w in p if w not in stops] for p in pars]
words = [w for p in pars for w in p]
counter = collections.Counter(words) # Other data structure?

# Assign an id to each word
word_dict = {w: i for i,w in enumerate(counter)}

T_word = len(word_dict)
T_par = len(pars)

def stochastic_batch(k_around, batch_size=1):
    # Paragraph 1-hot vector
    p = random.randrange(T_par)
    t_par = [0] * T_par
    t_par[p] = 1

    par = pars[p]
    l = len(par)

    # Center 1-hot vector
    c_pos = random.randrange(k_around, l-k_around)
    c = par[c_pos]
    t_c = [0] * T_word
    t_c[word_dict[c]] = 1

    # Surrounding words 1-hot matrix
    ws = par[c_pos-k_around:c_pos] + par[c_pos+1:c_pos+k_around+1]
    t_ws = [[0] * (2*k_around) for _ in range(T_word)]
    for j,i in enumerate([word_dict[w] for w in ws]):
        t_ws[i][j] = 1

    return t_par, t_c, t_ws        
    

q_word = 50
q_par = 150

t_c = tf.placeholder(tf.float32, shape=[T_word,1])       # 1-hot vector for central word
t_word = tf.placeholder(tf.float32, shape=[T_word,None]) # 1-hot matrix for words
t_par = tf.placeholder(tf.float32, shape=[T_par,1])      # 1-hot vector for pars

# NPLM parameters
W = tf.Variable(tf.random_uniform([q_word,T_word], -1.0, 1.0))
D = tf.Variable(tf.random_uniform([q_par,T_par], -1.0, 1.0))

# Context vectors 
h_word = tf.matmul(W,t_word)
h_word = tf.reduce_mean(h_word,1) # Take the average of context words.
h_par = tf.matmul(D,t_par)
h_par = h_par[:,1]
h = tf.concat(0,[h_word,h_par])

# Softmax sizes
p = q_word + q_par
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
