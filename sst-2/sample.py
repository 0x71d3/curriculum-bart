import csv
import os
import re
import sys

in_dir = sys.argv[1]

with open(os.path.join(in_dir, 'test.tsv')) as f:
    test_size = len(f.readlines())

dict_of_pairs = {}

for split in ['train', 'dev']:
    pairs = []
    with open(os.path.join(in_dir, split + '.tsv')) as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            pairs.append((row['sentence'], int(row['label'])))

    if split == 'train':
        dict_of_pairs['train'] = pairs[:-test_size]
        dict_of_pairs['test'] = pairs[-test_size:]
    else:
        dict_of_pairs['val'] = pairs

for split, pairs in dict_of_pairs.items():
    with open(split + '.tsv', 'w') as f:
        for sentence, label in pairs:
            f.write(sentence + '\t' + str(label) + '\n')
