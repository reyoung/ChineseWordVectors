import os
import cPickle
import random

__all__ = ['reader_creator']


def reader(filename, window_size, word_limit):
    s = word_limit
    e = word_limit + 1
    with open(filename, 'rb') as f:
        sentences = cPickle.load(f)
        random.shuffle(sentences)
        for sentence in sentences:
            sentence = filter(lambda x: x < word_limit, sentence)
            sentence = [s] + sentence + [e]

            for i in xrange(len(sentence) - window_size):
                yield sentence[i: i + window_size]


def reader_creator(window_size, word_limit, path):
    def __impl__():
        for dirpath, dirnames, filenames in os.walk(path):
            if len(filenames) != 0:
                random.shuffle(filenames)
                for filename in filenames:
                    for item in reader(filename=os.path.join(dirpath, filename),
                                       window_size=window_size,
                                       word_limit=word_limit):
                        yield item

    return __impl__
