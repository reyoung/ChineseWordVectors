import cPickle
import sys

WORD_DICT = 'word_dict.pkl'
IN_FILE = sys.argv[1]
OUT_FILE = sys.argv[2]

with open(WORD_DICT, 'rb') as f:
    word_dict = cPickle.load(f)


def mapping_words():
    with open(IN_FILE, 'r') as fin:
        for line in fin:
            line = line.decode('utf-8')
            yield [word_dict[elem] for w in line.strip().split() for elem in w]


cPickle.dump(list(mapping_words()), open(OUT_FILE, 'wb'), -1)
