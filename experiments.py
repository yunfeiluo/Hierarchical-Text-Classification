# Import packages and modules
from utils import *
import sys

if __name__ == '__main__':
    ## INITIALIZATION SETTING HERE ###########################################################
    data_set_name = 'wikipedia'  # either amazon_review, tweets, or wikipedia
    feature_extractor = "conv"  # either "conv" or "fc". "rnn"'s implementation is not complete, use later
    depth_of_model = "deep"  # either "shallow" or "deep"
    level_of_tokens = 'word'  # either "word" or "char"
    # (conv, shallow, word), (conv, shallow, char), (fc, shallow, word), (fc, shallow, char), (conv, deep, word)训练时间不短， (conv, deep, char)训练时间长表现不是很好不推荐
    ##########################################################################################

    # brief check of setting
    if feature_extractor == 'fc' and depth_of_model == "deep":
        print("Deep version of FC layers is not supported. ")
        exit()

    # load data
    data_name = 'dataset/{}_{}_level.pkl'.format(data_set_name, level_of_tokens)
    data = read_data(data_name)

    print('data load complete. Here are the keys:')
    for key in data:
        print(key)
    print('seq_len', len(data['train_text'][0]))

    # specify output size for each task
    tasks_size = {
        'label1': len(data['level1_ind']),
        'label2': len(data['level2_ind']),
        'label3': len(data['level3_ind'])
    }

    tasks_weight = {
        'label1': 1e-2,
        'label2': 1e-1,
        'label3': 1e1
    }
    print(tasks_size)

    # # embedding
    # embedding_weight = None
    # if level_of_tokens == 'word':
    #     embedding_weight = word2_vec(data['train_text'], data['vocab'])
    #     print('embedding shape', embedding_weight.shape)
    #
    # # setup hyperparameters, then start experiment
    # use_pre_embed = False
    # if embedding_weight is not None:
    #     use_pre_embed = True
    # print('Use pre-trained word embedding vector:', use_pre_embed)
    #
    # # Hyper-parameter setting here
    # vocab_size = len(data['vocab'])
    # embedding_dim = 16 if not use_pre_embed else embedding_weight.shape[1]
    # shared_out_size = 2048
    #
    # print('vocab size', vocab_size)
    # print('embedding_dim', embedding_dim)
    #
    # # for convolutional layers. Fixed
    # params = {
    #     'in_size': len(data['train_text'][0]),
    #     'in_channel': embedding_dim,
    #     'begin_channel': 64
    # }
    #
    # # construct model, code details in models.py
    # model = multitask_net(
    #     tasks_size=tasks_size,
    #     vocab_size=vocab_size,
    #     extractor=feature_extractor,
    #     depth=depth_of_model,
    #     embedding_dim=embedding_dim,
    #     shared_out_size=shared_out_size,
    #     params=params,
    #     embedding_weight=embedding_weight  # if using pre-train word embedding. format: list of vector
    # )
    #
    # # declare loss function, here we use CrossEntropy (logistic + softmax)
    # loss_func = torch.nn.CrossEntropyLoss()
    #
    # # declare optimizer
    # # turning on epochs and lr
    # epochs = 15  # 10, 15, 20
    # lr = 1e-2
    #
    # # The following are fixed
    # batch_size = 64  # 32, 64, 128, 256, recommend 32 or 64
    # optimizer = optim.SGD(
    #     params=model.parameters(),
    #     lr=lr,
    #     momentum=0.9,
    #     #     nesterov=True
    # )
    # scheduler = StepLR(optimizer, step_size=1, gamma=1.0)  # if not decay lr, set gamma to 1

    # load previous trained model
    saved_filename = 'saved_models/wikipedia_conv_deep_word_level'
    saved_record = None
    with open(saved_filename, 'rb') as f:
        print('load', saved_filename)
        saved_record = pickle.load(f)

    model = saved_record["model"]
    optimizer = saved_record["optimizer"]
    scheduler = saved_record['scheduler']
    batch_size = saved_record['batch_size']
    loss_func = saved_record['loss_func']
    epochs = 2

    print(model)

    # start training and validating
    model, train_loss, val_loss, train_scores, val_scores = train_val(data,
                                                                      model,
                                                                      loss_func,
                                                                      tasks_weight,
                                                                      optimizer,
                                                                      scheduler,
                                                                      epochs,
                                                                      batch_size)

    # pack results
    pack_train_scores = {
        'f1_score': {
            'label1': [i['f1_score']['label1'] for i in train_scores],
            'label2': [i['f1_score']['label2'] for i in train_scores],
            'label3': [i['f1_score']['label3'] for i in train_scores]
        },
        'accuracy': {
            'label1': [i['accuracy']['label1'] for i in train_scores],
            'label2': [i['accuracy']['label2'] for i in train_scores],
            'label3': [i['accuracy']['label3'] for i in train_scores]
        },
        'auc_score': {
            'label1': [i['auc_score']['label1'] for i in train_scores],
            'label2': [i['auc_score']['label2'] for i in train_scores],
            'label3': [i['auc_score']['label3'] for i in train_scores]
        }
    }

    pack_val_scores = {
        'f1_score': {
            'label1': [i['f1_score']['label1'] for i in val_scores],
            'label2': [i['f1_score']['label2'] for i in val_scores],
            'label3': [i['f1_score']['label3'] for i in val_scores]
        },
        'accuracy': {
            'label1': [i['accuracy']['label1'] for i in val_scores],
            'label2': [i['accuracy']['label2'] for i in val_scores],
            'label3': [i['accuracy']['label3'] for i in val_scores]
        },
        'auc_score': {
            'label1': [i['auc_score']['label1'] for i in val_scores],
            'label2': [i['auc_score']['label2'] for i in val_scores],
            'label3': [i['auc_score']['label3'] for i in val_scores]
        }
    }

    # saving
    saved = {
        'model': model,
        'loss_func': loss_func,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'epoch': epochs,
        'batch_size': batch_size,
        'trends': (train_loss, val_loss, pack_train_scores, pack_val_scores)
    }

    # write to file (Remeber to change the filename if not want to overwritten previous results)
    model_file_name = 'saved_models/{}_{}_{}_{}_level'.format(data_set_name, feature_extractor, depth_of_model,
                                                              level_of_tokens)

    # testing
    best_model = model
    eval_metrics = test_model(data, best_model, loss_func)
    # print('test result: ', result)

    save = input('do you want to save the progress? (y/n)')
    if save == 'y':
        save_progress(model_file_name, saved)
        print('Complete saving results. ')
    else:
        print('Training complete. ')

# ============================ REMARKS BELOW =============================================

    '''
    数据预处理具体细节:
    
    对每个数据集，处理出来两个数据， word level和character level的，格式如下:
        - For word level: "datasetName_word_level.pkl"
        - For character level: "datasetName_char_level.pkl"

    datasetName 命名（我们的三个数据集，目前 amazon_review已经处理完了）： "amazon_review", "tweets", 和 "wikipedia". 
    所有处理的的数据文件保存在dataset文件夹里

    先解释下之后要用到的用到的表示符号：
    训练集数量： N_train
    验证集数量： N_val
    测试集数量： N_test
    数据长度（number of tokens）：L
    字典大小：V

    处理后的格式是个dictionary，每一项具体如下:
    
    dictionary: 
        {
            'train_text': 训练数据，矩阵形状是 (N_train, L),
            'train_label1': 第一层label，形状是 (, N_train),
            'train_label2': 第二层label，形状是 (, N_train),
            'train_label3': 第三层label，形状是 (, N_train), 注：如果是tweets数据集应该只有两层label，不需要这一项
            'val_text': 验证数据，矩阵形状是 (N_val, L),
            'val_label1': 第一层label，形状是 (, N_val),
            'val_label2': 第二层label，形状是 (, N_val),
            'val_label3': 第三层label，形状是 (, N_val), 注：如果是tweets数据集应该只有两层label，不需要这一项
            'test_text': 测试数据，矩阵形状是 (N_test, L),
            'test_label1': 第一层label，形状是 (, N_test),
            'test_label2': 第二层label，形状是 (, N_test),
            'test_label3': 第三层label，形状是 (, N_test), 注：如果是tweets数据集应该只有两层label，不需要这一项
            'vocab': 字典，也是个dictionary，格式是{token: index}, 假如字典是['a', 'b', 'c'], 那么vocab这里就是{'a': 0, 'b':1, 'c': 2},
            'level1_ind': label字典，也是个dictionary, 格式是（index: index）,假如有这几个label: ['a', 'b', 'c'], 那么level1_ind这里就是{'a': 0, 'b':1, 'c': 2},
            'level2_ind': 同上，
            'level3_ind': 同上，也是如果是tweets数据集只有两层label，不需要这一项
        }
    
    对于'train_text', 'val_text', 和 'test_text', 格式如下：
    假如我们有两个数据：['a'， 'b'], ['b', 'a'],
    然后我们的vocab是{'a': 0, 'b':1, 'c': 2},
    那么最终text是： [0, 1], [1, 0]

    text长度L是固定的，分析完数据后定好一个数，长的砍掉，短的padding.
    '''
