import os
import time
import shutil


import torch

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
    parser.add_argument('--model_name', default='./models/hei', help='Path to save the model.')
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--k', default=64, type=int, help='length of the hashing code.')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--embed_size', default=1024, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--use_BatchNorm', action='store_false', help='Whether to use BN.')
    parser.add_argument('--activation_type', default='tanh', help='choose type of activation functions.')
    parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate.')   
    parser.add_argument('--learning_rate', default=.0002, type=float, help='Initial learning rate.') 
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    opt = parser.parse_args()

    emb_images, emb_caps, cap_lens, matching_scores =  loaddata("./models/model_best.pth.tar", "./data", split='short')
    print('img: ' + str(emb_images.shape) + ' caps: ' + str(emb_caps.shape))
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

    model = HEI(opt, emb_images.shape[1], emb_caps.shape[1])

    for epoch in range(opt.num_epochs):
        print(epoch)
        train(model, emb_images, emb_caps, matching_scores)
    # save
    torch.save(model.state_dict(), opt.model_name+'/result.pth.tar')

if __name__ == '__main__':
    main()

