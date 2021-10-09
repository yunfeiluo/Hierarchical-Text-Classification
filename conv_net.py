import torch
from torch import nn
import numpy as np

class kmax_pooling(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.dim = dim
        self.k = k
    
    def forward(self, x):
        index = x.topk(self.k, dim = self.dim)[1].sort(dim = self.dim)[0]
        return x.gather(self.dim, index)

class ResBlock1d(nn.Module):
    def __init__(self, channel, out_channel=None):
        super().__init__()
        self.res = out_channel == None
        out_channel = channel if self.res else out_channel
        # first_stride = 1 if self.res else 2

        self.conv_liner = nn.Sequential(
            nn.Conv1d(channel, out_channel, 3, stride=1, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Conv1d(out_channel, out_channel, 3, stride=1, padding=1),
            nn.BatchNorm1d(out_channel)
        )
    
    def forward(self, x):
        out = self.conv_liner(x)
        return x + out if self.res else out

class conv_layers(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_size = params['in_size']
        in_channel = params['in_channel']
        begin_channel = params['begin_channel']

        self.conv_liner = nn.Sequential(
            nn.Conv1d(in_channel, begin_channel, 3, stride=1, padding=1),
            nn.BatchNorm1d(begin_channel),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
    
    def forward(self, x):
        x = x.permute(0,2,1) # switch the dimension, so the first dimension is the embedding vector dim
        return self.conv_liner(x)

class ResLayers1d(nn.Module):
    def __init__(self, params):
        super().__init__()
        '''
        @param in_size: length of input sequence (e.g. word sequence)
        @param begin_channel: begin_channel
        '''
        in_size = params['in_size']
        in_channel = params['in_channel']
        begin_channel = params['begin_channel']

        self.clip_liner = nn.Sequential(
            nn.Conv1d(in_channel, begin_channel, 3, stride=1, padding=1),
            nn.BatchNorm1d(begin_channel),
            nn.ReLU(),
        )

        self.res_liner = nn.Sequential(
            ResBlock1d(begin_channel),
            nn.ReLU(),
            ResBlock1d(begin_channel),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            ResBlock1d(begin_channel, out_channel=begin_channel*2),
            nn.ReLU(),
            ResBlock1d(begin_channel*2),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            ResBlock1d(begin_channel*2, out_channel=begin_channel*4),
            nn.ReLU(),
            ResBlock1d(begin_channel*4),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            ResBlock1d(begin_channel*4, out_channel=begin_channel*8),
            nn.ReLU(),
            ResBlock1d(begin_channel*8),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2, padding=1),
        )

        self.k_max_pool = kmax_pooling(dim=2, k=8)
        
    def forward(self, x):
        x = x.permute(0,2,1) # switch the dimension, so the first dimension is the embedding vector dim
        return self.k_max_pool(self.res_liner(self.clip_liner(x)))

if __name__ == '__main__':
    x = torch.randn(2, 3, 4)
    print(x)
    pool = kmax_pooling(2, 2)
    y = pool(x)
    print(y.shape)
    print(y)
