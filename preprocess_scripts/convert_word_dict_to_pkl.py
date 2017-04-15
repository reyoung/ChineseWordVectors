import cPickle
import sys

wd = dict()
idx = 0
for line in sys.stdin:
    line = line.decode('utf-8')
    try:
        wc, w = line.split()
        wd[w] = idx
        idx += 1
    except:
        pass

cPickle.dump(wd, sys.stdout, -1)
