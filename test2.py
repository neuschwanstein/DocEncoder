import nltk
from ArticleReader import ArticleReader

corpus = []
# for record in ArticleReader(100):
#     corpus.extend(nltk.word_tokenize(record.article))
# corpus = [word for nltk.word_tokenize(r.article) in ArticleReader(100) for word in ]
for record in ArticleReader(100):
    corpus.extend(nltk.word_tokenize(record.article))
tagged_corpus = nltk.pos_tag(corpus)

num_words = []
for i,tag in enumerate(tagged_corpus):
    if tag[1] != 'CD':          # Numeral tag
        continue
    print(*corpus[i-5 if i-5>= 0 else 0 : i+2])
    
        
        
    
    
# tagged_corpus = nltk.pos_tag(corpus)
# nums = [tag[0] for tag in tagged_corpus if tag[1] == 'CD']
