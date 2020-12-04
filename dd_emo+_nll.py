import argparse
import csv
import glob
import os
import shutil
from tqdm.auto import tqdm

import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from model_roberta import RobertaFinetuner
from dataset import EmotionDataset

parser = argparse.ArgumentParser()

parser.add_argument('--orig_dir')
parser.add_argument('--output_dir')
parser.add_argument('--model_dir')

args = parser.parse_args()

for split in ['val', 'test']:
    shutil.copyfile(
        os.path.join(args.orig_dir, split + '.tsv'),
        os.path.join(args.output_dir, split + '.tsv')
    )

pairs = []
with open(os.path.join(args.orig_dir, 'train.tsv')) as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        pairs.append(tuple(row))

checkpoint_path = glob.glob(args.model_dir + '/checkpointepoch=*.ckpt')[0]
model = RobertaFinetuner.load_from_checkpoint(
    checkpoint_path,
    get_dataset=None
).model.cuda()

model.eval()

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
dataset = EmotionDataset(tokenizer, args.orig_dir, 'train', 512)

loader = DataLoader(dataset, batch_size=4)

errors = []
for batch in tqdm(loader):
    outs = model(
        input_ids=batch['source_ids'].cuda(),
        attention_mask=batch['source_mask'].cuda()
    )
    lprobs = F.log_softmax(outs[0], dim=-1)

    error = -lprobs.gather(-1, batch['target_ids'].cuda().unsqueeze(-1))
    errors.extend(error.squeeze().tolist())

error_map = dict(zip(pairs, errors))
pairs.sort(key=error_map.get)

with open(os.path.join(args.output_dir, 'train.tsv'), 'w') as f:
    for pair in pairs:
        f.write('\t'.join(pair) + '\n')

with open(os.path.join(args.output_dir, 'diffic.txt'), 'w') as f:
    for pair in pairs:
        f.write(str(error_map[pair]) + '\n')
