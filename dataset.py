import csv
import os

import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.path = os.path.join(data_dir, type_path + '.tsv')

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze(0)  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze(0)  # might need to squeeze

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

    def _build(self):
        with open(self.path, newline='') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                input_, target = '\t'.join(row[:-1]), int(row[-1])

                # tokenize inputs
                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input_],
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    return_tensors="pt"
                )
                # tokenize targets
                tokenized_targets = {
                    'input_ids': torch.tensor([[target]], dtype=torch.long),
                    'attention_mask': torch.tensor([[1]], dtype=torch.long)
                }

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
