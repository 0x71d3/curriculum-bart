import csv
import math
import os
import shutil
import sys
from collections import Counter

from nltk.tokenize import word_tokenize

orig_dir, output_dir = sys.argv[1:]

for split in ['val', 'test']:
    shutil.copyfile(
        os.path.join(orig_dir, split + '.tsv'),
        os.path.join(output_dir, split + '.tsv')
    )

pairs = []
with open(os.path.join(orig_dir, 'train.tsv')) as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        pairs.append(tuple(row))

counter = Counter()
for sentence, _ in pairs:
    for word in word_tokenize(sentence):
        counter[word] += 1
N_total = sum(counter.values())

rarity_map = {}
for pair in pairs:
    rarity = 0.0
    for word in word_tokenize(pair[0]):
        rarity += -math.log(counter[word] / N_total)
    rarity_map[pair] = rarity
    
pairs.sort(key=rarity_map.get)

with open(os.path.join(output_dir, 'train.tsv'), 'w') as f:
    for pair in pairs:
        f.write('\t'.join(pair) + '\n')

with open(os.path.join(output_dir, 'diffic.txt'), 'w') as f:
    for pair in pairs:
        f.write(str(rarity_map[pair]) + '\n')
