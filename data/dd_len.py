import csv
import os
import shutil
import sys

from nltk.tokenize import word_tokenize

orig_dir, output_dir = sys.argv[1:]

for split in ['val', 'test']:
    shutil.copyfile(
        os.path.join(orig_dir, split + '.tsv'),
        os.path.join(output_dir, split + '.tsv')
    )

pairs = []
lens = []
with open(os.path.join(orig_dir, 'train.tsv')) as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        pairs.append(tuple(row))
        lens.append(len(word_tokenize(row[0])))

len_map = dict(zip(pairs, lens))
pairs.sort(key=len_map.get)

with open(os.path.join(output_dir, 'train.tsv'), 'w') as f:
    for pair in pairs:
        f.write('\t'.join(pair) + '\n')

with open(os.path.join(output_dir, 'diffic.txt'), 'w') as f:
    for pair in pairs:
        f.write(str(len_map[pair]) + '\n')
