import glob
import os
import sys
import textwrap
from tqdm.auto import tqdm

from sklearn import metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from dataset import EmotionDataset
from model import BartFinetuner

data_dir, output_dir = sys.argv[1:]

checkpoint_path = glob.glob(output_dir + '/checkpointepoch=*.ckpt')[0]
model = BartFinetuner.load_from_checkpoint(
    checkpoint_path,
    get_dataset=None
).model.cuda()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

model.eval()

dataset = EmotionDataset(tokenizer, data_dir, 'test', max_len=512)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

it = iter(loader)

batch = next(it)
print(batch["source_ids"].shape)

for i in range(8):
    outs = model(
        input_ids=batch["source_ids"].cuda(),
        attention_mask=batch["source_mask"].cuda(),
        # labels=batch['target_ids'].cuda()
    )

    dec = outs[0].argmax(-1).tolist()

    texts = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in batch['source_ids']
    ]
    targets = [ids.item() for ids in batch['target_ids']]

    for i in range(2):
        lines = textwrap.wrap(f"Text: {texts[i]}")
        print("\n".join(lines))
        print(f"\nActual emotion: {targets[i]}")
        print(f"Predicted emotion: {dec[i]}")
        print("=" * 70 + "\n")

loader = DataLoader(dataset, batch_size=2, num_workers=4)

outputs = []
targets = []
for batch in tqdm(loader):
    outs = model(
        input_ids=batch["source_ids"].cuda(),
        attention_mask=batch["source_mask"].cuda(),
        # labels=batch['target_ids'].cuda()
    )

    dec = outs[0].argmax(-1).tolist()
    target = [ids.item() for ids in batch["target_ids"]]

    outputs.extend(dec)
    targets.extend(target)

print(metrics.accuracy_score(targets, outputs))

print(metrics.classification_report(targets, outputs, digits=4))

cm = metrics.confusion_matrix(targets, outputs)

labels = (
    ["positive", "negative"] if data_dir == 'sst-2'
    else ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize = (10, 7))
sn.heatmap(df_cm, annot=True, cmap='Purples', fmt='g')

# save results
with open(os.path.join(output_dir, "outputs.txt"), 'w') as f:
    for dec in outputs:
        f.write(str(dec) + '\n')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
