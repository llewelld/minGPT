import os
from argparse import ArgumentParser

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Dataset, DataLoader
import math
from collections import OrderedDict

import torch
from mingpt.callback import CUDACallback
from mingpt.lr_decay import LearningRateDecayCallback
from mingpt.utils import sample
from mingpt.model import GPT
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.callbacks import DeviceStatsMonitor


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        # Use an OrderedDict to ensure deterministic behaviour
        chars = list(OrderedDict.fromkeys(data))
        data_size, vocab_size = len(data), len(chars)
        rank_zero_info('data has %d characters, %d unique.' % (data_size, vocab_size))

        assert vocab_size < 256, 'The vocabulary exceeds byte-size storage'

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = bytes([self.stoi[s] for s in data])

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        dix = list(self.data[i:i + self.block_size + 1])
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

if __name__ == '__main__':
    seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--n_layer', default=22, type=int)
    parser.add_argument('--n_head', default=16, type=int)
    parser.add_argument('--n_embd', default=3072, type=int)
    parser.add_argument('--learning_rate', default=6e-4, type=float)
    parser.add_argument('--block_size', default=128, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--precision', default=16, type=int)
    parser.add_argument('--strategy', default='auto')
    parser.add_argument('--enable_progress_bar', default=True, type=bool)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--accelerator', default="gpu")
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--gpus', default=None, type=int)

    args = parser.parse_args()

    if not os.path.exists("input.txt"):
        os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

    text = open('input.txt', 'r').read()  # don't worry we won't run out of file handles
    train_dataset = CharDataset(text, args.block_size)  # one line of poem is roughly 50 characters
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    gpu_support = args.accelerator == 'gpu' and args.devices > 0

    if args.gpus != None:
        if args.gpus > 0:
            args.accelerator = 'gpu'
        args.devices = args.gpus

    model = GPT(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.learning_rate,
        gpu_support=gpu_support,
        itos=train_dataset.itos,
    )

    lr_decay = LearningRateDecayCallback(
        learning_rate=6e-4,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * args.block_size
    )

    profiler = SimpleProfiler(dirpath='.', filename='perf_logs')
    trainer = Trainer(
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,
        callbacks=[lr_decay, DeviceStatsMonitor()] + ([CUDACallback()] if gpu_support else []),
        precision=args.precision,
        enable_progress_bar=args.enable_progress_bar,
        strategy=args.strategy,
        accelerator=args.accelerator,
        devices=args.devices,
        profiler=profiler,
    )
    trainer.fit(model, train_loader)

    trainer.save_checkpoint("trained.ckpt")

    context = "By artificial intelligence, or wrong surmise,"
    x = torch.tensor([model.stoi[s] for s in context], dtype=torch.long)[None,...].to(model.device)
    y = sample(model, x, 1000, temperature=0.9, sample=True, top_k=5)[0]
    completion = ''.join([model.itos[int(i)] for i in y])
    print(completion)

