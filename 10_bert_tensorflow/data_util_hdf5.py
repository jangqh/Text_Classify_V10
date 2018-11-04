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

LABEL_SPLITER = '__label__'


def build_chunk(lines, chunk_num=10):
    """
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
    if os.path.exists(cache_path):
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
        input_list, input_label = get_input_string_and_labels(line, \
                tokenize_style=tokenize_style)




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



