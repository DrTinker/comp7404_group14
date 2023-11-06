import argparse
import sys
import time
import torch
import numpy as np
from utils.distance import hamming
from dataloader.model_loader import loaddata

from modules.HEI import HEI

def gen_all_hash_code(model, path='./models'):
    emb_images, emb_caps, cap_len, _ =  loaddata("./models/model_best.pth.tar", "./data", split='test')
    img_codes, txt_codes = model.forward_hash(emb_images, emb_caps)
    data = []
    for i in range (cap_len):
        data.append([img_codes[i], txt_codes[i]])
    np.savetxt(path+'/code.txt', np.array(data))

def img_inference(model, path='./models'):
    emb_images, emb_caps, cap_len, _ =  loaddata("./models/model_best.pth.tar", "./data", split='test')
    img_codes, txt_codes = model.forward_hash(emb_images, emb_caps)
    # 遍历全部code，计算最高相似度
    data = np.loadtxt(path+'/code.txt')
    T1 = time.time()
    cnt = 0
    for i, code in enumerate(img_codes):
        min_dis = sys.maxsize
        flag = 0
        for j, line in enumerate(data):
            if min_dis>=hamming(line[0], code):
                flag = j
        if flag==i:
            cnt += 1
    T2 = time.time()
    print('img_inference')
    print('processing time is: %sms' % ((T2 - T1)*1000))
    print('accuracy is: ' + str(cnt/cap_len))

def txt_inference(model, path='./models'):
    emb_images, emb_caps, cap_len, _ =  loaddata("./models/model_best.pth.tar", "./data", split='test')
    img_codes, txt_codes = model.forward_hash(emb_images, emb_caps)
    # 遍历全部code，计算最高相似度
    data = np.loadtxt(path+'/code.txt')
    T1 = time.time()
    cnt = 0
    for i, code in enumerate(txt_codes):
        min_dis = sys.maxsize
        flag = 0
        for j, line in enumerate(data):
            if min_dis>=hamming(line[1], code):
                flag = j
        if flag==i:
            cnt += 1
    T2 = time.time()
    print('txt_inference')
    print('processing time is: %sms' % ((T2 - T1)*1000))
    print('accuracy is: ' + str(cnt/cap_len))

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

    model = HEI(opt)
    model.load_state_dict(torch.load(opt.model_name))

    gen_all_hash_code(model=model)
    img_inference(model=model)
    txt_inference(model=model)