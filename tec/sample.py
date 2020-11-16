import csv
import os
import re
import sys

labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

mention = re.compile(r'^@\w+\s+')
hashtag = re.compile(r'\s+#\w+$')

in_path = sys.argv[1]

pairs = []
with open(in_path) as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        text = '\t'.join(row[1:-1])
        label = labels.index(row[-1][3:])

        # clean text
        while re.search(mention, text):
            text = re.sub(mention, '', text)
        while re.search(hashtag, text):
            text = re.sub(hashtag, '', text)

        pairs.append((text, label))

val_size = len(pairs) // 10
train_size = len(pairs) - 2 * val_size

dict_of_pairs = {
    'train': pairs[:train_size],
    'val': pairs[train_size:train_size+val_size],
    'test': pairs[train_size+val_size:]
}

for split, pairs in dict_of_pairs.items():
    with open(split + '.tsv', 'w') as f:
        for text, label in pairs:
            f.write(text + '\t' + str(label) + '\n')
