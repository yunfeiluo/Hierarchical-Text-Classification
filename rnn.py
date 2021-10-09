import torch
from torch import nn
import numpy as np

class ResLSTM_block(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.res_shortcut = in_dim == hidden_dim

        self.rnn1 = nn.LSTM(in_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.rnn2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=0.2)

    def forward(self, x, h_0, c_0):
        packed_output, (hidden, cell) = self.rnn1(x, (h_0, c_0))
        packed_output, (hidden, cell) = self.rnn2(packed_output, (hidden, cell))

        if self.res_shortcut:
            return x + packed_output, (hidden, cell)
        return packed_output, (hidden, cell)

class ResLSTM(nn.Module):
    def __init__(self, params, hidden_dim):
        super().__init__()
        in_size = params['in_size']
        in_channel = params['in_channel']
        begin_channel = params['begin_channel']
        
        self.temp = nn.LSTM(in_channel, hidden_dim, batch_first=True, dropout=0.2)
        self.rnn1 = ResLSTM_block(hidden_dim, hidden_dim)
        self.rnn2 = ResLSTM_block(hidden_dim, hidden_dim)
        self.rnn3 = ResLSTM_block(hidden_dim, hidden_dim)

        self.final_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=0.2)
    
    def forward(self, input_seq):
        packed_output, (hidden, cell) = self.temp(input_seq)
        packed_output, (hidden, cell) = self.rnn1(packed_output, hidden, cell)
        packed_output, (hidden, cell) = self.rnn2(packed_output, hidden, cell)
        packed_output, (hidden, cell) = self.rnn3(packed_output, hidden, cell)
        packed_output, (hidden, cell) = self.final_rnn(packed_output, (hidden, cell))
        return hidden.squeeze()

class LSTM_layers(nn.Module):
    def __init__(self, params, hidden_dim):
        super().__init__()
        '''
        @param in_size: length of input sequence (e.g. word sequence)
        @param begin_channel: begin_channel
        '''
        in_size = params['in_size']
        in_channel = params['in_channel']
        begin_channel = params['begin_channel']

        self.rnn = nn.LSTM(in_channel, hidden_dim, batch_first=True, dropout=0.2)
        # self.rnn = nn.GRU(in_channel, hidden_dim, batch_first=True)
        # self.rnn = nn.RNN(in_channel, hidden_dim, batch_first=True)
    
    def forward(self, input_seq):
        packed_output, (hidden, cell) = self.rnn(input_seq)
        return hidden.squeeze()
