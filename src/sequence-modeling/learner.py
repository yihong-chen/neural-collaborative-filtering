import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Learner:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loss = []
        self.val_loss = []

    def train(self, train_data, train_tgt, ntokens, batch_size = 10):
        # Turn on training mode which enables dropout.
#         print ("Training data:", train_data.shape, train_tgt.shape)
        self.model.train()
        losses = []
        start_time = time.time()
        hidden = self.model.init_hidden(batch_size)
        N = train_data.shape[0]
        num_batches = (N-1)//batch_size + 1;
        for i in range(0, num_batches-1):
#             print ("Batch", i*batch_size, "to" , min((i+1)*batch_size, N))
            data = train_data[i*batch_size:min((i+1)*batch_size, N)].transpose(0,1)
            tgt = train_tgt[i*batch_size:min((i+1)*batch_size, N)]
#             print ("Batch:", data.shape, tgt.shape)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.model.zero_grad()
            hidden = hidden.detach()
#             print ("Hidden:", hidden.shape)
            scores, hidden = self.model(data, hidden)
#             print ("Scores:", scores.shape) 
            loss = self.criterion(scores.view(-1,ntokens), tgt.view(-1))
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),  0.25)
            self.optimizer.step()

            losses.append(loss.item())

            if i*batch_size % 1000 == 0 and i>0:
                avg_batch_loss = np.mean(losses)
                print('{:5d}/{:5d} batches  | '
                        'loss {:5.2f}'.format(
                    i, num_batches, avg_batch_loss))
        self.train_loss.append(np.mean(losses))

    def evaluate(self, data, tgt, ntokens, batch_size=10, val=True):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.
#         print ("Evaluation Data:", data.shape, data.transpose(0,1).shape, tgt.shape)
    
        hidden = self.model.init_hidden(batch_size)
        with torch.no_grad():
            scores, hidden = self.model(data.transpose(0,1), hidden)
#             print ("Scores:", scores.shape)
#             hidden = hidden.detach()
            loss = self.criterion(scores.view(-1, ntokens), tgt.view(-1)).item()
        if val:
            self.val_loss.append(loss)
        return loss
    
    def generate_seq (self, init_movie_id, movie_embeddings, seq_length):
        cur_movie_id = init_movie_id
        hidden = self.model.init_hidden(1)
        seq = []
        seq.append(cur_movie_id)
        with torch.no_grad():
            for i in range(seq_length):
                emb = torch.Tensor(movie_embeddings[cur_movie_id]).view(1,1,-1)
                scores, hidden = self.model(emb, hidden)
                # Sample from the network as a multinomial distribution
                output_dist = scores.data.view(-1).div(0.8).exp()
                sample = torch.multinomial(output_dist, 1)[0]
    #             print ("Sampled: ", sample)
                seq.append(sample)
                cur_movie_id = sample
        return seq

    def plotLearningCurve(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.train_loss, label='Train')
        plt.plot(self.val_loss, label='Val')
        plt.legend(loc='lower left')