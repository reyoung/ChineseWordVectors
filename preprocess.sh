#!/bin/bash
# using pv, tr, sort, uniq.
cd `dirname $0`
echo "Generating Word Dict"
pv data/data/* | python preprocess_scripts/generate_word_dict.py > word_dict_wiki
pv weibo_data/* | python preprocess_scripts/generate_word_dict.py > word_dict_weibo 

echo "Convert Word Dict to Python Pickle Format"
pv word_dict_wiki | python preprocess_scripts/convert_word_dict_to_pkl.py >word_dict_wiki.pkl
pv word_dict_weibo | python preprocess_scripts/convert_word_dict_to_pkl.py >word_dict_weibo.pkl

echo "Convert Text to Pickle Format"
mkdir -p preprocessed_wiki/data/data
mkdir -p preprocessed_weibo/weibo_data

find ./data/data/ -name 'data_*' | xargs -Iop -P3 python preprocess_scripts/word_to_wordids.py op ./preprocessed_wiki/op word_dict_wiki.pkl

find ./weibo_data/ -name 'stc_*' | xargs -Iop -P3 python preprocess_scripts/word_to_wordids.py op ./preprocessed_weibo/op word_dict_weibo.pkl
