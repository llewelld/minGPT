from pytorch_lightning.strategies import DeepSpeedStrategy
import deepspeed
import pytorch_lightning as pl
import torch
import torch.nn as nn
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.nn import functional as F
from mingpt.block import Block
from torch.optim import AdamW

class GPT(pl.LightningModule):
    def __init__(self,
                 vocab_size,
                 weight_decay=0.1,
                 betas=(0.9, 0.95),
                 learning_rate=3e-4,
                 n_embd=768,
                 block_size=128,
                 embd_pdrop=0.1,
                 n_layer=12,
                 n_head=4,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 gpu_support=False,
                 itos=[],
                 ):
        super().__init__()
        self.save_hyperparameters()
        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # decoder head
        self.ln_f = nn.LayerNorm(self.hparams.n_embd)
        self.head = nn.Linear(self.hparams.n_embd, self.hparams.vocab_size, bias=False)

        self.block_size = self.hparams.block_size

        # transformer
        self.blocks = nn.ModuleList([Block(self.hparams) for _ in range(self.hparams.n_layer)])

        self.gpu_support = gpu_support

        # Retain a copy of the vocabulary conversion map for checkpointing purposes
        self.itos = itos
        self.init_stoi()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        # todo: need to enable deepspeed cpu adam only if offloading

        if self.gpu_support:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)
            return FusedAdam(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)
        else:
            return AdamW(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config['zero_optimization']
            return config.get('offload_optimizer') or config.get('offload_param')
        return False

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        if self.gpu_support:
            for block in self.blocks:
                x = deepspeed.checkpointing.checkpoint(block, x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log('train_loss', loss)
        return loss

    def get_block_size(self):
        return self.block_size

    def on_save_checkpoint(self, checkpoint):
        # Save out the vocab conversion dictionary to the checkpoint
        # This avoids us having to load in the data to use the trained model
        checkpoint['itos'] = self.itos

    def on_load_checkpoint(self, checkpoint):
        # Load in the vocab conversion dictionary from the checkpoint
        # This avoids us having to load in the data to use the trained model
        self.itos = checkpoint['itos']
        self.init_stoi()

    def init_stoi(self):
        # Create a reverse lookup, strings to integers
        self.stoi = {value: key for key, value in self.itos.items()}
