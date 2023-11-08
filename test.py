import argparse
import json
import os
import sys
import time
import torch
import numpy as np
from utils.distance import hamming
from dataloader.model_loader import loaddata

from modules.HEI import HEI


def gen_all_hash_code(img_codes, txt_codes, ids, path='./models'):
    print('generate all hash code')
    # gen hash code
    img_data = []
    txt_data = []
    for i in range (len(img_codes)):
        pos = i*5
        img_data.append({ids[pos] : img_codes[i].detach().numpy().tolist()})
        for j in range(i*5, (i+1)*5):
            txt_data.append({ids[pos] : txt_codes[j].detach().numpy().tolist()})
    # save as json
    img_path = path+'/img_code.json'
    with open(img_path,"w") as f:
        json.dump(img_data, f)

    txt_path = path+'/txt_code.json'
    with open(txt_path,"w") as f:
        json.dump(txt_data, f)
    print('hash codes saved')

def img_inference(img_codes, path='./models'):
    # 遍历全部code，计算最高相似度

    txt_data = []
    img_data = []

    img_path = path+'/img_code.json'
    txt_path = path+'/txt_code.json'
    with open(txt_path,"r") as f:
        txt_data = json.load(f)
    with open(img_path,"r") as f:
        img_data = json.load(f)
    
    T1 = time.time()
    cnt = 0
    for code in img_codes:
        min_dis = sys.maxsize
        flag = 0
        for line in txt_data:
            if min_dis>=hamming(list(line.values())[0], code):
                flag = list(line.keys())[0]
        # search by flag in img_data
        for line in img_data:
            if list(line.keys())[0]==flag and list(line.values())[0]==code.detach().numpy().tolist():
                cnt += 1
    T2 = time.time()
    print('img_inference')
    print('processing time is: %sms' % ((T2 - T1)*1000))
    print('accuracy is: ' + str(cnt/len(img_codes)))

def txt_inference(txt_codes, path='./models'):
    
    # 遍历全部code，计算最高相似度
    txt_data = []
    img_data = []

    img_path = path+'/img_code.json'
    txt_path = path+'/txt_code.json'
    with open(txt_path,"r") as f:
        txt_data = json.load(f)
    with open(img_path,"r") as f:
        img_data = json.load(f)
    T1 = time.time()
    cnt = 0
    for code in txt_codes:
        min_dis = sys.maxsize
        flag = 0
        for line in img_data:
            if min_dis>=hamming(list(line.values())[0], code):
                flag = list(line.keys())[0]
        # search by flag in img_data
        for line in txt_data:
            if list(line.keys())[0]==flag and list(line.values())[0]==code.detach().numpy().tolist():
                cnt += 1
    T2 = time.time()
    print('txt_inference')
    print('processing time is: %sms' % ((T2 - T1)*1000))
    print('accuracy is: ' + str(cnt/len(txt_codes)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/', help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp', help='{coco,f30k}_precomp')
    parser.add_argument('--data_split', default='test', help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/', help='Path to saved vocabulary json files.')
    parser.add_argument('--logger_name', default='./runs/runX/log', help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint', help='Path to save the model.')
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--skip_build', default=False, action='store_true', help='decide to generate hash code database or not')
    parser.add_argument('--k', default=64, type=int, help='length of the hashing code.')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--embed_size', default=1024, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--use_BatchNorm', action='store_false', help='Whether to use BN.')
    parser.add_argument('--activation_type', default='tanh', help='choose type of activation functions.')
    parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate.')   
    parser.add_argument('--learning_rate', default=.0002, type=float, help='Initial learning rate.') 
    opt = parser.parse_args()
    print('loading model')
    emb_images, emb_caps, cap_len, _ =  loaddata("./models/model_best.pth.tar", "./data", split=opt.data_split)
    model = HEI(opt, emb_images.shape[1], emb_caps.shape[1])
    model.load_state_dict(torch.load(opt.model_name + '/result.pth.tar'))
    img_codes, txt_codes = model.forward_hash(emb_images, emb_caps)
    ids = []
    with open('%s%s/%s_ids.txt' % (opt.data_path, opt.data_name, opt.data_split), 'r', encoding='utf-8') as f:
        ids = f.readlines()
    print('finish loading')

    if not opt.skip_build:
        gen_all_hash_code(img_codes, txt_codes, ids)
    
    img_inference(img_codes)
    txt_inference(txt_codes)

if __name__ == '__main__':
    main()