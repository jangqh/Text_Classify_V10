#import codec
import random
import numpy as np
import multiprocessing
from collections import Counter
import os
import pickle
import h5py
import time
import json
import jieba
import tensorflow as tf
from model.config import Config
import pdb

#定义常量
LABEL_SPLITER = '__label__'

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
MASK_ID = 3
_PAD = "PAD"
_UNK = "UNK"
_CLS = "CLS"
_MASK = "MASK"

def build_chunk(lines, chunk_num=10):
    """
    """
    pass


def load_data_multilabel(data_path,training_data_path, valid_data_path, \
        test_data_path, vocab_word2index, label2index, sentence_len, \
        process_num=20, test_mode=False,tokenize_style='word', model_name=None):
    """加载word和标签
    """
    pass

def create_or_load_vocabulary(data_path, training_data_path, vocab_size, \
        test_mode=False, tokenize_style='word', fine_tuning_stage=False, \
        model_name=None):
    """
    加载单词和标签
    load from cache if exists, load data, count and get vocubulary and labels
    """
    tf.logging.info("data path: %s", data_path)
    tf.logging.info("training data path: %s", training_data_path)
    tf.logging.info("vocab size: %s", vocab_size)
    
    t1 = time.clock()
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    #1.if cache exists,load it; otherwise create it
    if model_name is not None:
        cache_path = data_path+model_name+'vocab_label.pik'
    else:
        cache_path = data_path+'vocab_label.pik'
    
    tf.logging.info("cache_path:", cache_path, "file_exists:", \
            os.path.exists(cache_path))
    if False and os.path.exists(cache_path):
        with open(cache_path, 'rb') as cache_f:
            print("to load cache file,vocab of words and labels")
            return pickle.load(cache_f)

    #2.load and shuffle raw data
    file_object = open(training_data_path, mode='r',encoding='utf-8')
    lines = file_object.readlines()
    file_object.close()

    random.shuffle(lines)
    if test_mode:
        lines = lines[0:20000]
    else:
        lines = lines[0:200*1000] #为了处理的快，只选择200klines 

    #3.loop each line, put to counter
    c_inputs = Counter()
    c_labels = Counter()
    for i,line in enumerate(lines):
        #print(line)
        input_list, input_label = get_input_string_and_labels(line, \
                tokenize_style=tokenize_style)
        c_inputs.update(input_list)
        c_labels.update(input_label)
        if i % 1000 == 0:
            print("_id:",i, "create_or_load_vocabulary line:", line)
            print("_id:",i, "label:",input_label, "input_list:", input_list)
    
    #4.get most frequency words and all labels
    if tokenize_style == 'char':
        vocab_size = 6000 
    vocab_word2index = {}
    vocab_list = c_inputs.most_common(vocab_size)
    vocab_word2index[_PAD] = PAD_ID
    vocab_word2index[_UNK] = UNK_ID
    vocab_word2index[_CLS] = CLS_ID
    vocab_word2index[_MASK] = MASK_ID
    #pdb.set_trace()
    for i,t in enumerate(vocab_list):
        word, freq = t
        vocab_word2index[word] = i+4
    
    label2index = {}
    label_list = c_labels.most_common()
    for i,t in enumerate(label_list):
        label_name, freq = t
        label_name = label_name.strip()
        label2index[label_name]=i

    #5.save to file system if vocabulary of words not exists
    if not os.path.exists(cache_path):
        with open(cache_path, 'ab') as f:
            print("going to save cache file of vocab of words and labels")
            pickle.dump((vocab_word2index, label2index), f)

    t2 = time.clock()
    print("create vocabulary ended time spent for generate training data:", \
            (t2-t1))
    return vocab_word2index, label2index


def  get_input_string_and_labels(line, tokenize_style='word'):
    """get input string and label
    """
    element_list = line.strip().split(LABEL_SPLITER)
    input_strings = element_list[0]
    input_list = token_string_as_list(input_strings,
            tokenize_style=tokenize_style)
    input_labels = element_list[1:]
    input_labels = [str(label).strip() for label in input_labels \
            if label.strip()]
    return input_list, input_labels

def token_string_as_list(string, tokenize_style='word'):
    if random.randint(0,500) == 1:
        print("toke_string-as_list.string:",string, 
                "tokenize_style:", tokenize_style)
    length = len(string)
    if tokenize_style == 'char':
        listt = [string[i] for i in range(length)]
    elif tokenize_style == 'word':
        listt = jieba.cut(string)
    listt = [x for x in listt if x.strip()]
    return listt


if __name__ == '__main__':
    data_path = './data/'
    training_data_path = data_path+'bert_train.txt'
    valid_data_path = data_path+'bert_test.txt'
    test_data_path=valid_data_path
    vocab_size=50000
    process_num=5
    test_mode=True
    sentence_len=200
    #vocab_word2index, label2index=create_or_load_vocabulary(data_path, \
    #        training_data_path,vocab_size,test_mode=False)
    create_or_load_vocabulary(data_path, \
            training_data_path,vocab_size,test_mode=False)
    #tf.logging.info("vocab_word2index: %d, ")



