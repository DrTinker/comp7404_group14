import os
import time
import shutil


import torch
import numpy as np

import data
from vocab import Vocabulary, deserialize_vocab
from modules.BFAN import BFAN
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_xattn
from torch.autograd import Variable

import logging
import tensorboard_logger as tb_logger

import argparse

from modules.HEI import HEI
from dataloader.model_loader import loaddata
import time


def train(model, emb_i, emb_c, ms):
    model.train_start()
    model.train_hashing(emb_i, emb_c, ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/', help='path to datasets')
    parser.add_argument('--data_name', default='precomp', help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/', help='Path to saved vocabulary json files.')
    parser.add_argument('--logger_name', default='./runs/runX/log', help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint', help='Path to save the model.')
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--k', default=64, type=int, help='length of the hashing code.')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--embed_size', default=1024, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--use_BatchNorm', action='store_false', help='Whether to use BN.')
    parser.add_argument('--activation_type', default='tanh', help='choose type of activation functions.')
    parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate.')   
    parser.add_argument('--learning_rate', default=.0002, type=float, help='Initial learning rate.') 
    opt = parser.parse_args()

    emb_images, emb_caps, cap_lens, matching_scores =  loaddata("./models/model_best.pth.tar", "./data")
    # f = open('debugstoredata','wb')
    # pickle.dump(emb_images, f)
    # pickle.dump(emb_caps, f)
    # pickle.dump(cap_lens, f)
    # pickle.dump(matching_scores, f)
    # f.close()
    
    # f = open('debugstoredata','rb')
    # emb_images = pickle.load(f)
    # emb_caps = pickle.load(f)
    # cap_lens = pickle.load(f)
    # matching_scores = pickle.load(f)

    model = HEI(opt)

    for epoch in range(opt.num_epochs):
        print(epoch)
        train(model, emb_images, emb_caps, matching_scores)


if __name__ == '__main__':
    main()

