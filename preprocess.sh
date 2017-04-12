#!/bin/bash
# using pv, tr, sort, uniq.
cd `dirname $0`
echo "Generating Word Dict"

pv data/data/* | python generate_word_dict.py > word_dict

echo "Convert Word Dict to Python Pickle Format"
pv word_dict | python convert_word_dict_to_pkl.py >word_dict.pkl

echo "Convert Text to Pickle Format"
mkdir -p preprocessed/data/data

sz=`find ./data/data/ -name 'data_*'| wc -c`

find ./data/data/ -name 'data_*' | xargs -Iop -P3 python word_to_wordids.py op ./preprocessed/op
