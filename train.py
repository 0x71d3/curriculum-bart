import argparse
import os

import pytorch_lightning as pl

from model import BartFinetuner, LoggingCallback, args_dict
from dataset import EmotionDataset

parser = argparse.ArgumentParser()
for name, default in args_dict.items():
    parser.add_argument('--' + name, default=default, type=type(default))
args = parser.parse_args()

os.mkdir(args.output_dir)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir,
    prefix="checkpoint",
    monitor="val_loss",
    mode="min",
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
)


def get_dataset(tokenizer, type_path, args):
    return EmotionDataset(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        type_path=type_path,
        max_len=args.max_seq_length
    )


model = BartFinetuner(args, get_dataset)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

os.mkdir('bart_large_emotion')

## save the model this way so next time you can load it using T5ForConditionalGeneration.from_pretrained
model.model.save_pretrained('bart_large_emotion')
