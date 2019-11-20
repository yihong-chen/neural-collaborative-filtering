import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Learner:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def train(self, train_data, train_tgt, ntokens, lr=1):
        # Turn on training mode which enables dropout.
        self.model.train()
        total_loss = 0.
        start_time = time.time()
        hidden = self.model.init_hidden(1)
        for i, batch in enumerate(train_data):
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.model.zero_grad()

            hidden = hidden.detach()
            scores, hidden = self.model(train_data[i], hidden)

            loss = self.criterion(scores.view(-1, ntokens), train_tgt[i].view(-1))
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),  0.25)
            for p in self.model.parameters():
                p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()

            if i % 1000 == 0 and i>0:
                cur_loss = total_loss / 1000
                elapsed = time.time() - start_time
                print('{:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    i, len(train_data), lr,
                    elapsed * 1000 / 10, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate(self, data, tgt, ntokens):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.

        hidden = self.model.init_hidden(1)
        with torch.no_grad():
            for i, batch in enumerate(data):
                output, hidden = self.model(data[i], hidden)
                hidden = hidden.detach()
                output_flat = output.view(-1, ntokens)
                total_loss += self.criterion(output_flat, tgt[i].view(-1)).item()
        return total_loss / (len(data) - 1)