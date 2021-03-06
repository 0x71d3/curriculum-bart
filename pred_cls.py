import glob
import os
import sys
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from transformers import BartTokenizer

from model import BartFinetuner
from dataset import EmotionDataset

data_dir, output_dir = sys.argv[1:]

checkpoint_path = glob.glob(output_dir + '/checkpointepoch=*.ckpt')[0]
model = BartFinetuner.load_from_checkpoint(
    checkpoint_path,
    get_dataset=None
).model.cuda()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

dataset = EmotionDataset(tokenizer, data_dir, 'test', max_len=512)
loader = DataLoader(dataset, batch_size=2, num_workers=4)

model.eval()

outputs = []
for batch in tqdm(loader):
    outs = model(
        input_ids=batch['source_ids'].cuda(),
        attention_mask=batch['source_mask'].cuda(),
        # labels=batch['target_ids'].cuda()
    )

    dec = outs[0].argmax(-1).tolist()
    outputs.extend(dec)

with open(os.path.join(output_dir, 'pred.txt'), 'w') as f:
    for dec in outputs:
        f.write(str(dec) + '\n')
