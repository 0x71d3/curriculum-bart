import argparse
import os
import shutil

import pytorch_lightning as pl

from model import (
    BartFinetuner,
    CurriculumBartFinetuner,
    LoggingCallback,
    args_dict
)
from dataset import ResponseDataset, EmotionDataset

parser = argparse.ArgumentParser()

for name, default in args_dict.items():
    parser.add_argument('--' + name, default=default, type=type(default))
parser.add_argument('--num_labels', default=2, type=int)

parser.add_argument('--task', choices=['generation', 'classification'])

parser.add_argument('--curriculum', action='store_true')
parser.add_argument('--curriculum_epochs', default=10, type=int)
parser.add_argument('--init_competence', default=0.01, type=float)

parser.add_argument('--save_pretrained', action='store_true')

args = parser.parse_args()

if os.path.isdir(args.output_dir):
    shutil.rmtree(args.output_dir)
os.mkdir(args.output_dir)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir,
    prefix='checkpoint',
    monitor='val_loss',
    mode='min',
    save_top_k=1
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision=16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
    reload_dataloaders_every_epoch=args.curriculum
)


def get_dataset(tokenizer, type_path, args):
    if args.task == 'generation':
        return ResponseDataset(
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            type_path=type_path,
            max_len=args.max_seq_length
        )
    else:
        return EmotionDataset(
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            type_path=type_path,
            max_len=args.max_seq_length
        )


# initialize model
model = (
    CurriculumBartFinetuner(args, get_dataset) if args.curriculum
    else BartFinetuner(args, get_dataset)
)

# initialize trainer
trainer = pl.Trainer(**train_params)

# start fine-tuning
trainer.fit(model)

## save the model this way so next time you can load it using T5ForConditionalGeneration.from_pretrained
if args.save_pretrained:
    model.model.save_pretrained(args.output_dir)
