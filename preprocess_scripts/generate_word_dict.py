import collections
import sys

word_dict = collections.defaultdict(int)
for line in sys.stdin:
    line = line.decode('utf-8')
    line = line.strip()
    for w in line.split():
        for elem in w:
            word_dict[elem] += 1

word_list = [None] * len(word_dict)

for i, key in enumerate(word_dict):
    word_list[i] = (key, word_dict[key])

word_list.sort(key=lambda x: x[1], reverse=True)

for w, cnt in word_list:
    print cnt, w.encode('utf-8')
