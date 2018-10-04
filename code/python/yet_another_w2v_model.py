# import logging
import os
import re

from gensim.models import word2vec
from nltk import word_tokenize

from poets_conf import *


# word_count = {}
#
#
# def count_word(_s):
#     for t in _s:
#         for w in word_tokenize(t.lower()):
#             if w not in stop_words:
#                 if w in word_count:
#                     word_count[w] += 1
#                 else:
#                     word_count[w] = 1


def remove_stop_words(_s):
    # return [[w for w in word_tokenize(t.lower()) if w not in stop_words and word_count[w] >= MIN_FREQ] for t in _s]
    return [[w for w in word_tokenize(t.lower()) if w not in stop_words] for t in _s]


def prepare_sentence(s):
    # count_word(s)
    return remove_stop_words(s)


def get_sent_from_corpus(_corpus_dir):
    sent = [re.sub(r'[^a-zA-Z ]+', ' ', line) for line in open(_corpus_dir, 'r', encoding='utf-8')]
    return remove_stop_words(sent)


def train_model(corpus_dir):
    sentences = get_sent_from_corpus(corpus_dir)
    return word2vec.Word2Vec(sentences, size=DIMENSION, min_count=MIN_FREQ)


def dump_model(model_dir, _poet, _model):
    if not path.exists(model_dir):
        os.makedirs(model_dir)

    # 保存模型，以便重用
    _model.save(path.join(model_dir, '%s.model' % _poet))
    # 对应的加载方式
    # _model = word2vec.Word2Vec.load(path.join(model_dir, '%s.model' % _poet))


def test_model(_model, _text):
    _text = remove_stop_words(_text)
    # print(" ".join(w + " " + str(word_count[w]) for w in _text[0]))
    print(" ".join(w + " " + str(_model.wv.vocab[w].count) for w in _text[0] if w in _model.wv.vocab))
    for w in _text[0]:
        if w in _model.wv.vocab:
            print(w, ' '.join((format(x, ".5f") for x in _model[w].tolist())))


if __name__ == '__main__':
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    most_freq = {}

    # train model for different poets
    for poet in poets:
        print("[INFO] training model of %s ..." % poet)
        corpus_at = path.join(CORPUS_DIR, '%s.txt' % poet)
        model_at = path.join(MODEL_DIR, '%s.model' % poet)
        model = train_model(corpus_at)
        dump_model(MODEL_DIR, poet, model)
        test_model(model, text)
        top20 = sorted(model.wv.vocab.items(), key=lambda x: x[1].count, reverse=True)[:20]
        for word, vocab_obj in top20:
            if word not in most_freq:
                most_freq[word] = vocab_obj.count
            else:
                most_freq[word] += vocab_obj.count

    most_freq20 = sorted(most_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    print(' '.join([x[0] for x in most_freq20]))
    for word, count in most_freq20:
        print('%s: %s' % (word, count))
