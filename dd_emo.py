import csv
import os
import sys

input_dir, output_dir = sys.argv[1:]

for split in ['train', 'validation', 'test']:
    utterances = []
    with open(os.path.join(input_dir, split, 'dialogues_' + split + '.txt')) as f:
        for line in f:
            texts = line.split('__eou__')
            for text in texts[:-2]:
                utterances.append(text.strip())

    emotions = []
    with open(os.path.join(input_dir, split, 'dialogues_emotion_' + split + '.txt')) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        for row in reader:
            for label in row[:-2]:
                emotions.append(label)

    assert len(utterances) == len(emotions)
    pairs = list(zip(utterances, emotions))
    
    if split == 'validation':
        split = 'val'

    with open(os.path.join(output_dir, split + '.tsv'), 'w') as f:
        for utterance, emotion in pairs:
            f.write(utterance + '\t' + emotion + '\n')
