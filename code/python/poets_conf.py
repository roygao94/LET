from os import path

from nltk.corpus import stopwords

W2V_DIR = 'word2vec'
CORPUS_DIR = path.join(W2V_DIR, 'poems')
MODEL_DIR = path.join(W2V_DIR, 'poets_model')

text = ['THE SONG OF THE HAPPY SHEPHERD']
DIMENSION = 120
MIN_FREQ = 5

stop_words = stopwords.words('english')

poets = ['browning',
         'hardy',
         'tagore',
         'wilde',
         'yeats',
         ]
