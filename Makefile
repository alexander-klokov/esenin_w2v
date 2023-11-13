SRC='src'

data:
	python3 ${SRC}/esenin_w2v_data.py

train:
	python3 ${SRC}/esenin_w2v_train.py

word:
	python3 ${SRC}/esenin_w2v_word.py $(word)

tsne:
	python3 ${SRC}/esenin_w2v_tsne.py $(word)
