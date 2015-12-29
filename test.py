import collections
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
    for word,_ in count:
        dictionary[word] = len(dictionary) # Assign an id (0 to W) for each word in order of occurence
        
    data = list()
    for word in words:
        index = dictionary[word]
        data.append(index)
        
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print("Hello")
