"""
1.load data(X, y)
2.create session
3.feed data
4.training
5.validation
6.prediction

big:   d_model=512, h=8, d_k=d_v=64
small: d_model=128, h=8, d_k=d_v=16
tiny:  d_model=64, h=8,  d_k=d_v=8
"""
import tensorflow as tf
import numpy as np
#from bert_model import BertModel  # todo
from bert_cnn_model import BertCNNModel as BertModel
from data_util_hdf5 import create_or_load_vocabulary, load_data_mutilabel
from data_util_hdf5 import assign_pretrained_word_embedding, set_config
import os
from evaluation_matrix import *
from pretrain_task import mask_language_model, mask_language_model_multi_processing
from config import Config
import random
from datetime import datetime

#configuation
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("test_mode", False, "如果是测试模式， 少量数据被用到")
tf.app.flags.DEFINE_string("data_path", './data', "训练数据目录")
tf.app.flags.DEFINE_string("mask_lm_source_file", './data/bert_train2.txt', "训练数据")
tf.app.flags.DEFINE_string("ckpt_dir", './checkpoint_lm', 'checkpoint location for the model')
tf.app.flags.DEFINE_integer("vocab_size", 60000, "maximum vocab size")



def main():
    """
    入口
    """

if __name__ == '__main__':
    tf.app.run()
