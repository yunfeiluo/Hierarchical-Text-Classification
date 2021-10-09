# Import packages and modules
import pickle
import random
import sys
from copy import deepcopy

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from models import *


#### Prepare Data ###################################################
# Read data
# read data, assume these are the processed data
def read_data(data_name):
    data = None
    with open(data_name, 'rb') as f:
        data = pickle.load(f)
    return data


# if using pre-trained or pre-defined word-embedding vectors, such as word2vec, word level one hot, etc
def word2_vec(texts, vocab):
    ind_vocab = dict()
    for v in vocab:
        ind_vocab[vocab[v]] = v

    # print(texts[:2])

    context = list()
    for text in texts:
        context.append([ind_vocab[i] for i in text])

    # skip-gram model 
    print('Training word2vec...')
    # print(context[:2])
    word2vec_model = Word2Vec(context, min_count=1, size=32, window=5, sg=1)

    embedding_weight = list()
    for word in vocab:
        embedding_weight.append(word2vec_model[word])
    return torch.FloatTensor(embedding_weight)


def one_hot_embedding(vocab):
    embedding_weight = np.eye(len(vocab)).tolist()
    return torch.FloatTensor(embedding_weight)


######################################################################

# ======================= SPLIT LINE =================================

#### evaluation metric ###############################################
def eval_accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()


def eval_f1_score(y_pred, y_true):
    return f1_score(y_true, y_pred, average='weighted')


def eval_auc_score(y_pred, y_true):
    mlb = MultiLabelBinarizer()
    mlb.fit([[i] for i in y_true])
    y_true = mlb.transform([[i] for i in y_true])
    y_pred = mlb.transform([[i] for i in y_pred])

    roc_weighted = None
    try:
        roc_weighted = roc_auc_score(y_true, y_pred, average='weighted')
    except:
        roc_weighted = 0.0
    return roc_weighted


######################################################################

# ======================= SPLIT LINE =================================

#### mini-batch helper function ######################################
def get_mini_batchs(batch_size, inds):
    batch_inds = list()
    np.random.shuffle(inds)
    i = 0
    while i < len(inds):
        batch_inds.append(inds[i:i + batch_size])
        i += batch_size
    return batch_inds


# training and validating
def train_val(data, model, loss_func, tasks_weight, optimizer, scheduler, epochs, batch_size):
    # prepare data to tensor:
    train_sample = torch.Tensor(data['train_text']).type(torch.long)
    train_inds = [i for i in range(len(train_sample))]
    val_sample = torch.Tensor(data['val_text']).type(torch.long)
    val_inds = [i for i in range(len(val_sample))]
    train_labels = dict()
    val_labels = dict()
    for task in model.tasks_layer:
        train_labels[task] = torch.Tensor(data['train_' + task]).type(torch.long)
        val_labels[task] = torch.Tensor(data['val_' + task]).type(torch.long)

    # declare variable storing the results
    best_model = None

    train_scores = list()
    val_scores = list()

    train_loss = list()
    val_loss = list()
    min_val_loss = np.inf

    # start training and validatin
    for epoch in range(epochs):
        train_metrics = {
            "accuracy": dict(),
            "f1_score": dict(),
            "auc_score": dict()
        }

        val_metrics = {
            "accuracy": dict(),
            "f1_score": dict(),
            "auc_score": dict()
        }

        train_total_loss = 0
        val_total_loss = 0

        # training
        model.train()

        # get mini batchs
        batch_inds = get_mini_batchs(batch_size, train_inds)

        print('Train epoch {} ...'.format(epoch))
        y_preds = dict()
        y_trues = dict()
        iters = 0
        for batch_ind in batch_inds:
            iters += len(batch_ind)
            # forward
            loss = 0
            for task in model.tasks_layer:
                output = model(task, train_sample[batch_ind])
                loss += tasks_weight[task] * loss_func(output, train_labels[task][batch_ind])

                y_pred = torch.argmax(output, dim=1).detach().tolist()
                y_true = train_labels[task][batch_ind].detach().tolist()

                if iters % 6400 == 0:
                    print('{} accuracy {}'.format(task, eval_accuracy(np.array(y_pred), np.array(y_true))))

                try:
                    y_preds[task] += y_pred
                except:
                    y_preds[task] = y_pred
                try:
                    y_trues[task] += y_true
                except:
                    y_trues[task] = y_true

            #                 print('accuracy', eval_accuracy(np.array(y_pred), np.array(y_true)))

            # update parameters
            model.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().item()
            train_total_loss += loss * len(batch_ind)

        train_loss.append(train_total_loss / len(train_inds))
        for task in y_preds:
            train_metrics["accuracy"][task] = eval_accuracy(np.array(y_preds[task]), np.array(y_trues[task]))
            train_metrics["f1_score"][task] = eval_f1_score(np.array(y_preds[task]), np.array(y_trues[task]))
            train_metrics["auc_score"][task] = eval_auc_score(np.array(y_preds[task]), np.array(y_trues[task]))
        train_scores.append(train_metrics)

        # validation
        model.eval()

        # get mini batchs
        batch_inds = get_mini_batchs(batch_size, val_inds)

        print('Validation...')
        y_preds = dict()
        y_trues = dict()
        for batch_ind in batch_inds:
            iters += len(batch_ind)
            loss = 0
            for task in model.tasks_layer:
                output = model(task, val_sample[batch_ind])
                loss += tasks_weight[task] * loss_func(output, val_labels[task][batch_ind])

                y_pred = torch.argmax(output, dim=1).detach().tolist()
                y_true = val_labels[task][batch_ind].detach().tolist()

                try:
                    y_preds[task] += y_pred
                except:
                    y_preds[task] = y_pred
                try:
                    y_trues[task] += y_true
                except:
                    y_trues[task] = y_true

            #                 print('accuracy', eval_accuracy(np.array(y_pred), np.array(y_true)))

            # clear cache
            model.zero_grad()
            loss.backward()
            model.zero_grad()

            loss = loss.detach().item()

            val_total_loss += loss * len(batch_ind)

        val_total_loss /= len(val_inds)
        # if val_total_loss < min_val_loss:
        #     min_val_loss = val_total_loss
        #     best_model = deepcopy(model)
        val_loss.append(val_total_loss)
        for task in y_preds:
            val_metrics["accuracy"][task] = eval_accuracy(np.array(y_preds[task]), np.array(y_trues[task]))
            val_metrics["f1_score"][task] = eval_f1_score(np.array(y_preds[task]), np.array(y_trues[task]))
            val_metrics["auc_score"][task] = eval_auc_score(np.array(y_preds[task]), np.array(y_trues[task]))
        val_scores.append(val_metrics)

        scheduler.step()
        # print every epoch
        print('================================================')
        print('epoch', epoch)
        print('train loss:', train_loss[-1])
        print('val loss:', val_loss[-1])
        print(' ')
        for metric in train_metrics:
            print(metric)
            for task in train_metrics[metric]:
                print('train, {}: {}'.format(task, train_metrics[metric][task]))
                print('val, {}: {}'.format(task, val_metrics[metric][task]))
            print(' ')
        print('================================================')
        print(' ')

    return (model, train_loss, val_loss, train_scores, val_scores)


def test_model(data, model, loss_func):
    print("Testing...")
    test_sample = torch.Tensor(data['test_text']).type(torch.long)
    test_inds = [i for i in range(len(test_sample))]
    test_labels = dict()
    for task in model.tasks_layer:
        test_labels[task] = torch.Tensor(data['test_' + task]).type(torch.long)

    test_metrics = {
        "accuracy": dict(),
        "f1_score": dict(),
        "auc_score": dict()
    }

    # get mini batchs
    batch_inds = get_mini_batchs(64, test_inds)

    model.eval()

    test_loss = 0
    y_preds = dict()
    y_trues = dict()
    for batch_ind in batch_inds:
        loss = 0
        for task in model.tasks_layer:
            output = model(task, test_sample[batch_ind])
            loss += loss_func(output, test_labels[task][batch_ind])

            y_pred = torch.argmax(output, dim=1).detach().tolist()
            y_true = test_labels[task][batch_ind].detach().tolist()

            try:
                y_preds[task] += y_pred
            except:
                y_preds[task] = y_pred
            try:
                y_trues[task] += y_true
            except:
                y_trues[task] = y_true

        # clear cache
        model.zero_grad()
        loss.backward()
        model.zero_grad()

        loss = loss.detach().item()

        test_loss += loss * len(batch_ind)

    test_loss /= len(test_inds)
    for task in y_preds:
        test_metrics["accuracy"][task] = eval_accuracy(np.array(y_preds[task]), np.array(y_trues[task]))
        test_metrics["f1_score"][task] = eval_f1_score(np.array(y_preds[task]), np.array(y_trues[task]))
        test_metrics["auc_score"][task] = eval_auc_score(np.array(y_preds[task]), np.array(y_trues[task]))

    # print
    print('================================================')
    print('test loss:', test_loss)
    print(' ')
    for metric in test_metrics:
        print(metric)
        for task in test_metrics[metric]:
            print('test, {}: {}'.format(task, test_metrics[metric][task]))
        print(' ')
    print('================================================')

    return test_metrics


# plot figures
def plot_trend(train_trend, val_trend, y_name):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(y_name + " trend")
    ax1.plot([i for i in range(len(train_trend))], train_trend)
    ax1.set(xlabel='epochs', ylabel=y_name)
    ax1.set_title('Training ' + y_name)
    ax2.plot([i for i in range(len(val_trend))], val_trend)
    ax2.set(xlabel='epochs', ylabel=y_name)
    ax2.set_title('Validation ' + y_name)
    fig.tight_layout(pad=3.0)
    plt.show()


# save the progress
def save_progress(filename, saved_content):
    last_progress = None
    try:
        with open(filename, 'rb') as f:
            last_progress = pickle.load(f)

        print('Updating last record...')
        l_train_loss, l_val_loss, l_pack_train_scores, l_pack_val_scores = last_progress['trends']
        c_train_loss, c_val_loss, c_pack_train_scores, c_pack_val_scores = saved_content['trends']

        train_loss = l_train_loss + c_train_loss
        val_loss = l_val_loss + c_val_loss

        for metric in l_pack_train_scores:
            for task in l_pack_train_scores[metric]:
                l_pack_train_scores[metric][task] += c_pack_train_scores[metric][task]
        for metric in l_pack_val_scores:
            for task in l_pack_val_scores[metric]:
                l_pack_val_scores[metric][task] += c_pack_val_scores[metric][task]

        saved_content['trends'] = (train_loss, val_loss, l_pack_train_scores, l_pack_val_scores)

        with open(filename, 'wb') as f:
            pickle.dump(saved_content, f)
    except:
        print('Saving New record...')
        with open(filename, 'wb') as f:
            pickle.dump(saved_content, f)
