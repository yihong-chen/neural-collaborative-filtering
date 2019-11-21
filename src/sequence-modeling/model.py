import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, input_size, hidden_size, nlayers, dropout=0.5, temporal_batch_size=10):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.temporal_batch_size = temporal_batch_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        
        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, nlayers, nonlinearity='tanh')
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, nlayers)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, nlayers)
            
        self.decoder = nn.Linear(hidden_size, ntoken)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, emb, hidden):
        output, hidden = self.rnn(emb, hidden)
        scores = self.decoder(self.dropout(output))
#         print ("Hidden at the end of forward", hidden.shape)
        return scores, hidden

    def init_hidden(self):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.nlayers, self.temporal_batch_size, self.hidden_size),
                    weight.new_zeros(self.nlayers, self.temporal_batch_size, self.hidden_size))
        else:
            return weight.new_zeros(self.nlayers, self.temporal_batch_size, self.hidden_size)