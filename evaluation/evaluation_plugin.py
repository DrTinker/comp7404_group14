

import json
import sys
import time

import torch
from dataloader.model_loader import loaddata
from modules.HEI import HEI

from utils.distance import hamming

def evlrank_with_plugin(opt):
    print('loading model')
    emb_images, emb_caps, _ =  loaddata("./models/bfan/checkpoint/model_best.pth.tar", "./data", split=opt.data_split)
    model = HEI(opt)
    model.load_state_dict(torch.load(opt.model_name + '/result%d.pth.tar'%opt.result_num, map_location=torch.device('cpu')))
    img_codes, txt_codes = model.forward_hash(emb_images, emb_caps)
    ids = []
    with open('%s%s/%s_ids.txt' % (opt.data_path, opt.data_name, opt.data_split), 'r', encoding='utf-8') as f:
        ids = f.readlines()
    print('finish loading')

    if not opt.skip_build:
        gen_all_hash_code(img_codes, txt_codes, ids)
    
    img_inference(img_codes)
    txt_inference(txt_codes)


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

def img_inference(img_codes, path='./models', alpha=[0.33, 0.4, 0.5]):
    # 遍历全部code，计算最高相似度
    print('img_inference')
    txt_data = []
    img_data = []

    img_path = path+'/img_code.json'
    txt_path = path+'/txt_code.json'
    with open(txt_path,"r") as f:
        txt_data = json.load(f)
    with open(img_path,"r") as f:
        img_data = json.load(f)
    
    r1, r5, r10 = [], [], []
    T1 = time.time()
    for i, code in enumerate(img_codes):
        code = code.detach().numpy().tolist()
        sys.stdout.write('\r>> test images (cur: %d, total: %d)' % (i, len(img_codes)))
        min_dis = sys.maxsize
        flag = 0
        for line in txt_data:
            dis = hamming(list(line.values())[0], code)
            if min_dis>=dis:
                flag = list(line.keys())[0]
                min_dis = dis
        # search by flag in img_data
        for line in img_data:
            # if list(line.keys())[0]==flag and list(line.values())[0]==code:
            #     cnt += 1
            dis = hamming(list(line.values())[0], code)
            if list(line.keys())[0]==flag:
                if dis<=alpha[0]:
                    r1.append(dis)
                if dis<=alpha[1]:
                    r5.append(dis)
                if dis<=alpha[2]:
                    r10.append(dis)
                break
    T2 = time.time()
    
    average = (len(r1) + len(r5) + len(r10)) / 3 / len(img_codes)
    r = [len(r1)/len(img_codes), len(r5)/len(img_codes), len(r10)/len(img_codes)]
    print('\ncalculate i2t time: %.3fms' % ((T2 - T1)*1000))
    print('Average i2t Recall: %.3f' % average)
    print("Image to text: %.3f, %.3f, %.3f" % (r[0], r[1], r[2]))

def txt_inference(txt_codes, path='./models', alpha=[0.42, 0.5, 0.6]):
    print('txt_inference')
    # 遍历全部code，计算最高相似度
    txt_data = []
    img_data = []

    img_path = path+'/img_code.json'
    txt_path = path+'/txt_code.json'
    with open(txt_path,"r") as f:
        txt_data = json.load(f)
    with open(img_path,"r") as f:
        img_data = json.load(f)

    r1, r5, r10 = [], [], []
    T1 = time.time()
    for i, code in enumerate(txt_codes):
        code = code.detach().numpy().tolist()
        sys.stdout.write('\r>> test texts (cur: %d, total: %d)' % (i, len(txt_codes)))
        min_dis = sys.maxsize
        flag = 0
        for line in img_data:
            dis = hamming(list(line.values())[0], code)
            if min_dis>=dis:
                flag = list(line.keys())[0]
                min_dis = dis
        # search by flag in img_data
        for line in txt_data:
            # if list(line.keys())[0]==flag and list(line.values())[0]==code:
            #     cnt += 1
            dis = hamming(list(line.values())[0], code)
            if list(line.keys())[0]==flag:
                if dis<=alpha[0]:
                    r1.append(dis)
                if dis<=alpha[1]:
                    r5.append(dis)
                if dis<=alpha[2]:
                    r10.append(dis)
                break
    T2 = time.time()

    average = (len(r1) + len(r5) + len(r10)) / 3 / len(txt_codes)
    r = [len(r1)/len(txt_codes), len(r5)/len(txt_codes), len(r10)/len(txt_codes)]
    print('\ncalculate t2i time: %.3fms' % ((T2 - T1)*1000))
    print('Average t2i Recall: %.3f' % (average))
    print("Text to image: %.3f, %.3f, %.3f" % (r[0], r[1], r[2]))