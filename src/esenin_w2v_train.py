import multiprocessing
import random
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

from esenin_w2v_utils import reduce_dimensions

df_clean = pd.read_csv('df_clean.csv')

sent = [row.split() for row in df_clean['sentence_clean']]

phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

#  training
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
model = Word2Vec(min_count=1,
                     window=2,
                     vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)

model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)

model.save("word2vec.model")

# visuzlize the model
x_vals, y_vals, labels = reduce_dimensions(model)

plt.figure(figsize=(12, 12))
plt.scatter(x_vals, y_vals)

indices = list(range(len(labels)))
selected_indices = random.sample(indices, 50)
for i in selected_indices:
    plt.annotate(labels[i], (x_vals[i], y_vals[i]))

plt.show()
