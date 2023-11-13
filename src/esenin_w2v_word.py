import sys

from gensim.models import Word2Vec

WORD = sys.argv[1]

model = Word2Vec.load("word2vec.model")

###

most_similar = model.wv.most_similar(positive=[WORD])

print("Most similar for", WORD)
print(most_similar)
