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
#from bert_cnn_model import BertCNNModel as BertModel
from data_util_hdf5 import create_or_load_vocabulary, load_data_mutilabel
#from data_util_hdf5 import assign_pretrained_word_embedding, set_config
import os
#from evaluation_matrix import *
#from pretrain_task import mask_language_model, \
#        mask_language_model_multi_processing
#from config import Config
import random
from datetime import datetime


#configuation
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("test_mode", False, \
        "如果是测试模式， 少量数据被用到")
tf.app.flags.DEFINE_string("data_path", './data', "训练数据目录")
tf.app.flags.DEFINE_string("mask_lm_source_file", './data/bert_train.txt',\
        "训练数据")
tf.app.flags.DEFINE_string("ckpt_dir", './checkpoint_lm', \
        'checkpoint location for the model')
tf.app.flags.DEFINE_integer("vocab_size", 60000, "maximum vocab size")
tf.app.flags.DEFINE_integer("d_model", 64, "dimension of model")
tf.app.flags.DEFINE_integer("num_layer", 6, "number of layer")
tf.app.flags.DEFINE_integer("num_header", 8, "number of header")
tf.app.flags.DEFINE_integer("d_k", 8, "dimension of k")
tf.app.flags.DEFINE_integer("d_v", 8, "dimension of v")

tf.app.flags.DEFINE_string("tokenize_style", "word", \
        "ckeckpoint location for the model")
tf.app.flags.DEFINE_integer("max_allow_sentence_length", 10, \
        "max length of allowed sentence for ")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size for training ")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay \
        learning rate")
tf.app.flags.DEFINE_float("decay_rate", 1.0,"rate of decay for learning rate")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.9, "precentage to keep")
tf.app.flags.DEFINE_integer("sequence_length",200,"max sentence length")
tf.app.flags.DEFINE_integer("sequence_length_lm",10,  \
        "max length for masked language model")
tf.app.flags.DEFINE_boolean("is_training",True, \
        "true:for training, false:for testing/infering")
tf.app.flags.DEFINE_boolean("is_fine_tuning",False, "True:说明是微调阶段")
tf.app.flags.DEFINE_integer("validate_every",1,"validation 1 epochs")
tf.app.flags.DEFINE_integer("num_epochs",30,"number of epochs to run")
tf.app.flags.DEFINE_boolean("use_pretained_embedding", False, \
        "whether to use embedding or not")
#data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5--->
#data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin--->
#sgns.merge.char
tf.app.flags.DEFINE_integer("process_num",35,"number of cpu process")
tf.app.flags.DEFINE_string("word2vec_model_path",  \
        "./data/Tecent_AILab_ChineseEmbedding.txt", \
        "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_integer("process_num",30,"number of cpu process")



def main(_):
    """ """
    vocab_word2index, _ = create_or_load_vocabulary(FLAGS.data_path, \
            FLAGS.mask_lm_source_file, FLAGS.vocab_size, \
            test_mode=FLAGS.test_mode, tokenize_style=FLAGS.tokenize_style)
    vocab_size = len(vocab_word2index)
    tf.logging.info("bert pretrain vocab size: %d", vocab_size)
    index2word = {v:k for k,v in vocab_word2index.items()}
    train, valid, test = mask_language_model(FLAGS.mask_lm_source_file, )

    
if __name__ == '__main__':
    tf.app.run()
