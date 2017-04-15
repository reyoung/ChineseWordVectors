import cPickle
import sys

WORD_DICT = sys.argv[3]
IN_FILE = sys.argv[1]
OUT_FILE = sys.argv[2]

with open(WORD_DICT, 'rb') as f:
    word_dict = cPickle.load(f)


def mapping_words():
    with open(IN_FILE, 'r') as fin:
        for line in fin:
            line = line.decode('utf-8')
            line = filter(lambda x: x is not None,
                          [word_dict.get(w, None) for w in
                           line.strip().split()])
            if len(line) != 0:
                yield line


cPickle.dump(list(mapping_words()), open(OUT_FILE, 'wb'), -1)
