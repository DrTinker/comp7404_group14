import argparse
import json
from evaluation.evaluation import evalrank_without_plugin
import os
import sys
import time
import torch
import numpy as np
from utils.distance import hamming
from dataloader.model_loader import loaddata
from evaluation.evaluation_plugin import evlrank_with_plugin

from modules.HEI import HEI

def inference_with_plugin(opt):
    print('\n\ntest BFAN-HEI: \n')
    evlrank_with_plugin(opt)

def inference_without_plugin(opt):
    print('\n\ntest BFAN: \n')
    RUN_PATH = "./models/bfan/checkpoint/model_best.pth.tar"
    DATA_PATH = "./data"
    evalrank_without_plugin(RUN_PATH, data_path=DATA_PATH, split=opt.data_split, fold5=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/', help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp', help='{coco,f30k}_precomp')
    parser.add_argument('--data_split', default='test', help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/', help='Path to saved vocabulary json files.')
    parser.add_argument('--logger_name', default='./runs/runX/log', help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./models/hei', help='Path to save the model.')
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--skip_build', default=False, action='store_true', help='decide to generate hash code database or not')
    parser.add_argument('--k', default=64, type=int, help='length of the hashing code.')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--embed_size', default=1024, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--use_BatchNorm', action='store_false', help='Whether to use BN.')
    parser.add_argument('--activation_type', default='tanh', help='choose type of activation functions.')
    parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate.')   
    parser.add_argument('--learning_rate', default=.0002, type=float, help='Initial learning rate.')
    parser.add_argument('--img_region_num', default=36, type=int, help='img_region_num')
    parser.add_argument('--max_n_word', default=40, type=int, help='max_n_word')
    parser.add_argument('--result_num', default=0, type=int, help='result num')
    opt = parser.parse_args()

    inference_with_plugin(opt)
    inference_without_plugin(opt)

if __name__ == '__main__':
    main()