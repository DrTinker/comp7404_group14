import logging
import os
import time
import shutil
import numpy as np


import torch

import argparse
from evaluation.evaluation import shard_xattn

import dataloader.loader

from modules.HEI import HEI
from dataloader.model_loader import BFAN_model_loader, get_embs, loaddata
import time

from utils.vocab import deserialize_vocab



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/', help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp', help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./data/vocab/', help='Path to saved vocabulary json files.')
    parser.add_argument('--logger_name', default='./runs/runX/log', help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./models/hei', help='Path to save the model.')
    parser.add_argument('--data_split', default='dev', help='data split')
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--img_region_num', default=36, type=int, help='img_region_num')
    parser.add_argument('--max_n_word', default=40, type=int, help='max_n_word')
    parser.add_argument('--k', default=64, type=int, help='length of the hashing code.')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--embed_size', default=1024, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--use_BatchNorm', default=False, action='store_true', help='Whether to use BN.')
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
    parser.add_argument('--lambda_softmax', default=20., type=float,
                        help='Attention softmax temperature.')
    parser.add_argument('--focal_type', default="equal",
                        help='equal|prob')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--orig_img_path', default='./data/', help='path to get the original image data')
    parser.add_argument('--orig_data_name', default='f30k', help='{coco,f30k}')
    opt = parser.parse_args()
    
    # emb_images, emb_caps, cap_lens, matching_scores =  loaddata("./models/model_best.pth.tar", "./data", split=opt.data_split)
    # print('img: ' + str(emb_images.shape) + ' caps: ' + str(emb_caps.shape))

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
    pre_model = BFAN_model_loader("./models/bfan/checkpoint/model_best.pth.tar")
    vocab = deserialize_vocab(os.path.join(
        opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)
    train_loader, _ = dataloader.loader.get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt, train_split=opt.data_split)
    
    model = HEI(opt)


    for epoch in range(opt.num_epochs):
        print('epoch: ' + str(epoch))
        train(model, pre_model, train_loader, opt, epoch)

        if not os.path.exists(os.path.dirname(opt.model_name)):
            os.mkdir(os.path.dirname(opt.model_name))
        torch.save(model.state_dict(), opt.model_name+'/result%d.pth.tar' %epoch)



def train(model, pre_model, train_loader, opt, epoch):
    for i, (images, captions, lengths, ids) in enumerate(train_loader):
        # get embs
        img_embs, cap_embs, cap_lens = get_embs(pre_model, images, captions, lengths, ids)

        # 5 fold
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs),5)])
        sims = shard_xattn(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        # print('img_embs: ' + str(img_embs.shape) + ' cap_embs: ' + str(cap_embs.shape))
        # Update the model
        model.train_start()
        model.train_hashing(img_embs, cap_embs, sims, epoch)

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                .format(
                    epoch, i, len(train_loader), e_log=str(model.logger)))

if __name__ == '__main__':
    main()

