import os
import pdb
import json
import nltk
# data
# ├── test
# │   ├── ham
# │   └── spam
# └── train
#     ├── ham
#     └── spam

with open('pos_tags_to_index.json') as f:
    pos_tags_to_index = json.load(f)
with open ('./words.txt') as f:
    words = f.read()
words = set(words.splitlines())

def load_data():
    data = {
        'train':{
            'ham':[],
            'spam':[]
            },
        'test':{
            'ham':[],
            'spam':[]
            }
    }
    data['train']['ham'] = read_dir('./data/train/ham')
    data['train']['spam'] = read_dir('./data/train/spam')
    data['test']['ham'] = read_dir('./data/test/ham')
    data['test']['spam'] = read_dir('./data/test/spam')
    return data

def read_dir(path_to_dir):
    filenames = os.listdir(path_to_dir)
    vec_list = []
    for filename in filenames:
        filepath = os.path.join(path_to_dir, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath) as f:
                    text = f.read()
                tag_count = count_pos_tags(text)
                vec = make_vec(tag_count)
                vec_list.append(vec)
            except UnicodeDecodeError:
                pass
    return vec_list

def count_pos_tags(text):
    """Count the times each part of speech tag is used in text."""
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    tag_count = {}
    for tagged_token_tuple in tagged_tokens:
        tag = tagged_token_tuple[1]
        tag_count[tag] = tag_count.get(tag,0) + 1
    return tag_count

def make_vec(tag_count):
    """Convert tag_count into a vector."""
    vec = [0] * len(pos_tags_to_index)
    tags = tag_count.keys()
    for tag in tags:
        n = tag_count[tag]
        try:
            i = pos_tags_to_index[tag]
        except KeyError:
            pdb.set_trace() 
        vec[i] += n
    return vec


if __name__ == "__main__":  
    data = load_data()
    vec = data['train']['spam'][0]
