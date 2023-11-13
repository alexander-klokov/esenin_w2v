import sys

from gensim.models import Word2Vec
from esenin_w2v_utils import tsnescatterplot

WORD = sys.argv[1]

model = Word2Vec.load("word2vec.model")

tsnescatterplot(model, WORD, [i[0] for i in model.wv.most_similar(negative=[WORD])])
