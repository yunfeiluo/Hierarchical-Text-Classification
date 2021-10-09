import torch
from torch import nn
import numpy as np

from conv_net import *
from rnn import *

class classifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=None):
        super().__init__()
        self.fc_liner = None
        if hidden_size == None:
            self.fc_liner = nn.Sequential(
                nn.Linear(input_size, num_classes)
            )
        else:
            self.fc_liner = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes)
            )
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.fc_liner(x)

class multitask_net(nn.Module):
    def __init__(self, tasks_size: dict, extractor, depth, vocab_size, embedding_dim, shared_out_size, params, embedding_weight=None):
        '''
        @param tasks_size: map: task_name -> output size of the task
        # 如果不是multitask的话，这里的task_size里面应该只有一个东西
        @param seq_len: length of input sequence
        @param embedding_dim: the embedded dimension
        @param shared_out_size: output size of the shared classifier
        @param params: parameter for feature extraction layers
        @param embedding_weight: if given, then the embedding layer will be freeze, else the embedding will be learned during training
                                 default: None
        '''
        super().__init__()

        ### embedding layer ######################################################################
        self.embedding_layer = None
        if embedding_weight == None:
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.embedding_layer = nn.Embedding.from_pretrained(embedding_weight, freeze=True)
        ##########################################################################################

        # ======================== split line ====================================================
        
        ### shared layers ########################################################################
        self.feature_extractor = None
        self.extractor_name = extractor
        extractor_out_size = None
        if extractor == "conv":
            if depth == "shallow":
                self.feature_extractor = conv_layers(params)
                extractor_out_size = params['begin_channel']*(params['in_size'] // 2)
            elif depth == "deep":
                self.feature_extractor = ResLayers1d(params)
                extractor_out_size = params['begin_channel']*8*8
            else:
                print('Wrong setting for depth.')
                exit()
        elif extractor == 'fc':
            self.feature_extractor = nn.Linear(params['in_size']*params['in_channel'], 4096)
            extractor_out_size = 4096
        elif extractor == "rnn":
            if depth == "shallow":
                extractor_out_size = 64
                self.feature_extractor = LSTM_layers(params, extractor_out_size)
                shared_out_size = 256
            elif depth == "deep":
                extractor_out_size = 64
                self.feature_extractor = ResLSTM(params, extractor_out_size)
                shared_out_size = 256
        else:
            print('Wrong setting for feature extractor.')
            exit()
        
        self.shared_liner = nn.Sequential(
            classifier(extractor_out_size, shared_out_size), 
            nn.BatchNorm1d(shared_out_size),
            nn.ReLU()
        )
        ##########################################################################################

        # ======================== split line ====================================================

        ### task specific layers #################################################################
        tasks_layer = dict() # map: task -> corresponding layers
        for task in tasks_size:
            tasks_layer[task] = classifier(shared_out_size, tasks_size[task], hidden_size=shared_out_size)
        
        self.tasks_layer = nn.ModuleDict(tasks_layer)
        ##########################################################################################
    
    def forward(self, task, input_seq):
        embedding_out = self.embedding_layer(input_seq)
        if self.extractor_name == 'fc':
            embedding_out = embedding_out.view(input_seq.shape[0], -1)
        feature_out = self.feature_extractor(embedding_out)
        shared_out = self.shared_liner(feature_out)
        return self.tasks_layer[task](shared_out)

# check output
if __name__ == '__main__':
    # Supposed we have 1 task
    tasks_size = {
        'task1': 3
    }

    # 举个栗子
    seq_len = 96 # sentence length（# token）
    vocab_size = 100
    embedding_dim = 16 # embedding dimension size
    shared_out_size = 2048 # shared layer size
    x = torch.randint(0, vocab_size, (5, seq_len)) # 5 samples, with length seq_len, with vocab_size as dictionary size

    # Construct model
    params = {
        'in_size': seq_len,
        'in_channel': embedding_dim, 
        'begin_channel': 64
    }
    model = multitask_net(tasks_size, 'rnn', 'deep', vocab_size, embedding_dim, shared_out_size, params)
    print(model)

    # forward
    print('input shape', x.shape)
    y = model('task1', x)
    print('output shape', y.shape) # If shape is correct. This is for checking the shape，here，output should have shape [5, 3] (5 samples, Every sample have prediction of 3 classes)