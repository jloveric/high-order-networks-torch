# Copied from the tutorial
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
# and converted to lightning.

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import torch
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
import torch.utils.data as data


def collate_fn():
    pass


class SequenceDataset(data.Dataset):
    def __init__(self, data, bptt=35):
        self.data = data
        self.bptt = bptt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_len = min(self.bptt, len(self.data) - 1 - idx)
        #print('self.data.shape', self.data.shape)
        data = self.data[idx:idx+seq_len]
        target = self.data[idx+1:idx+1+seq_len].reshape(-1)
        return data, target


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print('x.shape', x.shape,'pe.shape', self.pe.shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(pl.LightningModule):

    def __init__(self, ntokens, ninp, nhead, nhid, nlayers, dropout=0.5, num_workers=5):
        super(TransformerModel, self).__init__()

        self.num_workers = num_workers
        self.ntokens = ntokens  # the size of vocabulary
        self.emsize = 200  # embedding dimension
        self.nhid = nhid  # the dimension of the feedforward network model in nn.TransformerEncoder
        # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.nlayers = nlayers
        self.nhead = nhead  # the number of heads in the multiheadattention models
        self.dropout = dropout  # the dropout value
        self.emsize = ninp

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(num_embeddings=ntokens, embedding_dim=ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(in_features=ninp, out_features=ntokens)
        self.criterion = nn.CrossEntropyLoss()
        self.bptt = 35
        self.init_weights()

    def prepare_data(self):
        # called only on 1 GPU
        # ok, so maybe this needs to be called elsewhere...
        url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
        self.test_filepath, self.valid_filepath, self.train_filepath = extract_archive(
            download_from_url(url))
        self.tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(
            map(self.tokenizer, iter(io.open(self.train_filepath, encoding="utf8"))))
        self.vocab = vocab

    def data_process(self, raw_text_iter):
        data = [torch.tensor([self.vocab[token] for token in self.tokenizer(item)],
                             dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def setup(self, step):
        # step is either 'fit' or 'test' 90% of the time not relevant

        self.train_data = self.data_process(
            iter(io.open(self.train_filepath, encoding="utf8")))
        self.val_data = self.data_process(
            iter(io.open(self.valid_filepath, encoding="utf8")))
        self.test_data = self.data_process(
            iter(io.open(self.test_filepath, encoding="utf8")))

        self.batch_size = 20
        self.eval_batch_size = 10

        self.train_dataset = SequenceDataset(self.train_data)
        self.val_dataset = SequenceDataset(self.val_data)
        self.test_dataset = SequenceDataset(self.test_data)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=self.device))
                == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        #print('src.shape', src.shape)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        #print('encouder.shape', output.shape)
        output = self.decoder(output)
        #print('decoder.shape', output.shape)
        return output

    def training_step(self, batch, batch_idx):
        src_mask = self.generate_square_subsequent_mask(sz=self.bptt)
        #print('batch', batch)
        x, y = batch
        #print('x.shape', x.shape)
        # x=x.permute(1,0,2)
        if x.size(0) != self.bptt:
            src_mask = self.generate_square_subsequent_mask(
                x.size(0))
        output = self.forward(x, src_mask)
        #print('output.shape', output.shape, 'targets.shape', y.shape)
        #loss = self.criterion(output.view(-1, self.ntokens), y)
        loss = self.criterion(output.view(-1, self.ntokens), y.flatten())
        return loss

    def val_step(self, batch, batch_idx):
        loss = self.evaluate(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.evaluate(batch, batch_idx)
        return loss

    def evaluate(self, batch, batch_idx):
        total_loss = 0.
        src_mask = self.generate_square_subsequent_mask(self.bptt)

        data, targets = batch

        if data.size(0) != self.bptt:
            src_mask = self.generate_square_subsequent_mask(
                data.size(0))

        output = self(data, src_mask)
        print('output.shape', output.shape, 'targets.shape', targets.shape)
        output_flat = output.view(-1, self.ntokens)
        loss = len(data) * \
            self.criterion(output_flat, targets).item()

        return loss

    def configure_optimizers(self):
        lr = 5.0  # learning rate
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
test_filepath, valid_filepath, train_filepath = extract_archive(
    download_from_url(url))
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(
    map(tokenizer, iter(io.open(train_filepath, encoding="utf8"))))


ntokens = len(vocab.stoi)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value
ninp = emsize  # embedding size

autoencoder = TransformerModel(
    ntokens=ntokens, ninp=ninp, nhid=nhid, nlayers=nlayers, nhead=nhead, dropout=dropout)
trainer = pl.Trainer(gradient_clip_val=0.5)
trainer.fit(autoencoder)
