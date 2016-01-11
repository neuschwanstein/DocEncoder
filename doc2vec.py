import collections
import nltk
import article_db

'''Returns a matrix of the encoding for each word in our vocabulary.'''
stops = { '.',';',',' }        # Improve with NLTK
corpus_pars = [nltk.word_tokenize(a.content) for a in article_db.fetch(10)]
corpus = [w.lower() for p in corpus_pars for w in p if w not in stops]
counter = collections.Counter(corpus)

p = 100
