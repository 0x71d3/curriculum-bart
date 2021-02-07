import csv
import os
import shutil

original_dir = 'dd'

labeled_dir = 'dd_emo+_nll'
unlabeled_dir = 'dd_emo-'

output_dir = 'dd_emo_nll_woinf'

if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

for split in ['val', 'test']:
    shutil.copyfile(
        os.path.join(original_dir, split + '.tsv'),
        os.path.join(output_dir, split + '.tsv')
    )

response_map = {}
with open(os.path.join(original_dir, 'train.tsv')) as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        response_map[row[0]] = row[1]

pairs = []
for input_dir in [unlabeled_dir, labeled_dir]:
    with open(os.path.join(input_dir, 'train.tsv')) as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            pairs.append((row[0], response_map[row[0]]))

diffics = []
with open(os.path.join(labeled_dir, 'diffic.txt')) as f:
    for line in f:
        diffics.append(float(line))

diffics = [0.0] * (len(pairs) - len(diffics)) + diffics

with open(os.path.join(output_dir, 'train.tsv'), 'w') as f:
    for pair in pairs:
        f.write('\t'.join(pair) + '\n')

with open(os.path.join(output_dir, 'diffic.txt'), 'w') as f:
    for diffic in diffics:
        f.write(str(diffic) + '\n')
