import csv
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

'''
Covid-19 原始数据集：
Test:3798
    Level 1:
        1 Positive 1546
        2 Neutral 619
        3 Negative 1634
    Level 2:
        1 Extremely Positive 599
        2 Positive 947
        3 Neutral 619
        4 Negative 1041
        5 Extremely Negative 592
Train: 41157
    Level 1:
        1 Positive 18046
        2 Neutral 7713
        3 Negative 15399
    Level 2:
        1 Extremely Positive 6624
        2 Positive 11422
        3 Neutral 7713
        4 Negative 9917
        5 Extremely Negative 5481
Valid: 4116
'''
stopSet = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'[a-zA-Z]\w+\'?\w*')
dictionary = dict()


def tokenize(sentence):
    tokens = tokenizer.tokenize(sentence)
    stop_word = []
    for word in tokens:
        if word not in stopSet:
            stop_word.append(word.lower())
    return stop_word


# Tokenize出train和test的text的部分然后split train为train set和val set
original_text = []
original_label = []
original_test_text = []
original_test_label = []
with open('NLP_Train - Corona_NLP_train.csv', 'r', encoding="latin-1") as file:
    reader = csv.reader(file)
    for row in reader:
        original_text.append(tokenize(row[0]))
        original_label.append(row[2])
with open('NLP_Test - Corona_NLP_test.csv', 'r', encoding="latin-1") as file:
    reader = csv.reader(file)
    for row in reader:
        original_test_text.append(tokenize(row[0]))
        original_test_label.append(row[2])
original_text = original_text[1:]
original_label = original_label[1:]
original_test_text = original_test_text[1:]
original_test_label = original_test_label[1:]
original_train_text = original_text[:37041]
original_train_label = original_label[:37041]
original_val_text = original_text[37041:]
original_val_label = original_label[37041:]

# label type 转成数字
original_train_label = [int(i) for i in original_train_label]
original_val_label = [int(i) for i in original_val_label]
original_test_label = [int(i) for i in original_test_label]


# text转成16位
def textTranslate(text, old_text):
    for i in old_text:
        temp = []
        if len(i) > 32:
            for j in range(32):
                word = i[j]
                temp.append(word)
        else:
            for j in range(len(i)):
                word = i[j]
                temp.append(word)
            for k in range(32 - len(i)):
                temp.append(" ")
        text.append(temp)


train_temp_text = []
textTranslate(train_temp_text, original_train_text)

# check_v = set()
# for text in original_train_text:
#     for word in text:
#         check_v.add(word)
# print('CHECK HERE: ', len([i for i in check_v]))

val_temp_text = []
textTranslate(val_temp_text, original_val_text)
test_temp_text = []
textTranslate(test_temp_text, original_test_text)

# build vocab from train_text
vocab = dict()
vocab[" "] = 0
count = 1
for sentence in train_temp_text:
    for word in sentence:
        if word not in vocab:
            vocab[word] = count
            count += 1
dictionary["vocab"] = vocab

# text to num
def textToNum(text, temp_text):
    for i in temp_text:
        temp = []
        for j in i:
            if j in vocab:
                temp.append(vocab[j])
            else:
                temp.append(0)
        text.append(temp)


train_text = []
val_text = []
test_text = []
textToNum(train_text, train_temp_text)
textToNum(val_text, val_temp_text)
textToNum(test_text, test_temp_text)

train_text = np.array(train_text)
val_text = np.array(val_text)
test_text = np.array(test_text)
dictionary["train_text"] = train_text
dictionary["val_text"] = val_text
dictionary["test_text"] = test_text


# add label to dictionary
def getLabel1(label1, label2):
    for i in label2:
        if i == 0 or i == 1:
            label1.append(0)
        if i == 2:
            label1.append(1)
        if i == 3 or i == 4:
            label1.append(2)


train_label1 = []
val_label1 = []
test_label1 = []
getLabel1(train_label1, original_train_label)
getLabel1(val_label1, original_val_label)
getLabel1(test_label1, original_test_label)
dictionary["train_label2"] = np.array(original_train_label)
dictionary["val_label2"] = np.array(original_val_label)
dictionary["test_label2"] = np.array(original_test_label)
dictionary["train_label1"] = np.array(train_label1)
dictionary["val_label1"] = np.array(val_label1)
dictionary["test_label1"] = np.array(test_label1)

# dictionary level dictionary
dictionary['level1_ind'] = {"Positive": 1, "Neutral": 2, "Negative": 3}
dictionary['level2_ind'] = {"Extreme Positive": 1, "Positive": 2, "Neutral": 3, "Negative": 4, "Extreme Negative": 5}

# pickle to file
with open('tweets_word_level.pkl', 'wb') as file:
    pickle.dump(dictionary, file)

print(dictionary["train_text"].shape)
print(dictionary["train_label1"].shape)
print(dictionary["train_label2"].shape)
print(dictionary["val_text"].shape)
print(dictionary["val_label1"].shape)
print(dictionary["val_label2"].shape)
print(dictionary["test_text"].shape)
print(dictionary["test_label1"].shape)
print(dictionary["test_label2"].shape)
print(len(dictionary["vocab"]))
