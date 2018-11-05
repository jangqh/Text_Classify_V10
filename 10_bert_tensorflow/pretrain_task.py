import jieba
from data_util_hdf5 import PAD_ID,UNK_ID,MASK_ID,_PAD,_UNK,_MASK, \
        create_or_load_vocabulary
import random
import re
import numpy as np
import os
import time
import pickle
import multiprocessing

splitter = '|&|'
eighty_percentage=0.8
nighty_percentage=0.9

def mask_language_model(source_file, target_file, index2word, \
        max_sentence_length=10,test_mode=False, process_num=10):
    """
    feed the input data through a deep tranformer encoder and then use the 
    final hidden states coresponding  to the masked positions to predict what 
    word was maked, exactly like train a language model.
    Input Sequence:  The man went to [masked] store with [masked] dog.
    Target Sequence:                 the                 his

    try prototype first, that is only select one word

    the training data generate choose 15% of tokens at random,e.g,. in the  
    sentence 'my dog is hairy it choose hairy'. It then seen performs the 
    following procedure:
    
    80% of time:Replace the word with the [MASK] token,e.g.
        my dog is hairy -> my dog is [mask]
    10% of time:Replace the word with a random word,e.g.
        my dog is hairy -> my dog is apple.
    10% of time:Keep the word unchanged
    
    The purpose of this is to bias the representation towards the actual  
        observed word.
    
    :paramter sentece_length: 
    :return :list of tuple.each tuple has a input_sequence, and target_sequence:
    (input_sequence, target_sentence)

    """
    #1. read source file
    t1 = time.clock()





if __name__ == '__main__':
    source_file = './data/zhihu/'
    data_path = './data'
    training_data = data_path + ''
    valid_data_path = data_path + ''
    test_data_path = valid_data_path
    vocab_size = 50000
    process_num = 5
    test_mode = True
    sentence_len = 200
