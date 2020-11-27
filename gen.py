import glob
import os
import sys
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

from model import BartFinetuner
from dataset import ResponseDataset

data_dir, output_dir = sys.argv[1:]

# model = BartForConditionalGeneration.from_pretrained(output_dir).cuda()
checkpoint_path = glob.glob(output_dir + '/checkpointepoch=*.ckpt')[0]
model = BartFinetuner.load_from_checkpoint(
    checkpoint_path,
    get_dataset=None
).model.cuda()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

dataset = ResponseDataset(tokenizer, data_dir, 'test', 128)
loader = DataLoader(dataset, batch_size=16)

model.eval()

outputs = []
for batch in tqdm(loader):
    outs = model.generate(
        input_ids=batch['source_ids'].cuda(),
        attention_mask=batch['source_mask'].cuda(), 
        max_length=128,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    outputs.extend(dec)

with open(os.path.join(output_dir, 'pred.txt'), 'w') as f:
    for dec in outputs:
        f.write(dec + '\n')
