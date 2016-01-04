import math
import collections
import random
import numpy as np
import tensorflow.python.platform
import tensorflow as tf
from ArticleReader import ArticleReader

words = []
dictionary = dict()
for record in ArticleReader():
    words.extend(record.article.lower().split())

c = collections.Counter(words)
# for word in words:
#     if word in dictionary:
        

def build_dataset(words):
  # count = [['UNK', -1]]
  # count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    count = collections.Counter(words).items()
    dictionary = dict()

    # Assign an id (0 to W) for each word in order of occurence
    for word,_ in count:
        dictionary[word] = len(dictionary)
        
    data = list()
    for word in words:
        index = dictionary[word]
        data.append(index)
        
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
vocabulary_size = len(count)


data_index = 0


# Step 4: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
                
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
            
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

# batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# for i in range(10):
#   print(batch[i], '->', labels[i, 0])
#   print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])


# Step 5: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

valid_size = 16
valid_window = 100
# valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
valid_examples = np.random.choice(valid_window, valid_size)
num_sampled = 64

graph = tf.Graph()

with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):

        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev= 1.0/math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                       num_sampled, vocabulary_size))
    
    # Construct the SGD optimizer using a learing rate of 1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)


# Step6: Begin training

num_steps = 100001

with tf.Session(graph=graph) as session:

    # Initialize variables before using them
    tf.initialize_all_variables().run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = { train_inputs: batch_inputs, train_labels: batch_labels }

        # Perform one update step by evaluating the optimizaer
        _,loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # EXPENSIVE STEP!
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8 # Nearest neigbors
                nearest = (-sim[i,:]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()
