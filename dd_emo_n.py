import csv
import os
import sys
from collections import Counter

input_dir, output_dir = sys.argv[1:]

for split in ['train', 'validation', 'test']:
    pairs = []
    with open(os.path.join(input_dir, split, 'dialogues_' + split + '.txt')) as f:
        for line in f:
            texts = line.split('__eou__')[:-1]
            for i in range(len(texts) - 1):
                pairs.append((texts[i].strip(), texts[i+1].strip()))

    if split == 'train':
        emotions = []
        with open(os.path.join(input_dir, split, 'dialogues_emotion_' + split + '.txt')) as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            for row in reader:
                for label in row[:-2]:
                    emotions.append(int(label))

        assert len(pairs) == len(emotions)

        emotion_map = dict(zip(pairs, emotions))
        emotion_order = list(Counter(emotions))

        pairs.sort(key=lambda pair: emotion_order[emotion_map[pair]])

        with open(os.path.join(output_dir, 'diffic.txt'), 'w') as f:
            for pair in pairs:
                f.write(str(emotion_order[emotion_map[pair]]) + '\n')
        
    elif split == 'validation':
        split = 'val'

    with open(os.path.join(output_dir, split + '.tsv'), 'w') as f:
        for utterance, response in pairs:
            f.write(utterance + '\t' + response + '\n')
