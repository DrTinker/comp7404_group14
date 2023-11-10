import os
import numpy as np
from dataloader.loader import get_test_loader
from evaluation.evaluation import encode_data, shard_xattn
from modules.BFAN import BFAN

from utils.vocab import deserialize_vocab
import torch

def loaddata(model_path, data_path=None, split='dev', roll_n=5):
    checkpoint = None
    if not torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    if data_path is not None:
        opt.data_path = data_path

    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    model = BFAN(opt)
    model.load_state_dict(checkpoint['model'])
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.batch_size, opt.workers, opt)
    img_embs, cap_embs, cap_lens = encode_data(model, data_loader)

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), roll_n)])

    return img_embs, cap_embs, cap_lens


def BFAN_model_loader(model_path):
    checkpoint = None
    if not torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    model = BFAN(opt)
    model.load_state_dict(checkpoint['model'])

    return model

def get_embs(model, images, captions, lengths, ids):
    img_embs = None
    cap_embs = None
    cap_lens = None

    max_n_word = 0
    for l in lengths:
        max_n_word = max(max_n_word, l)

    # compute the embeddings
    img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths, volatile=False)

    if img_embs is None:
        if img_emb.dim() == 3:
            img_embs = np.zeros((len(images), img_emb.size(1), img_emb.size(2)))
        else:
            img_embs = np.zeros((len(images), img_emb.size(1)))
        cap_embs = np.zeros((len(images), max_n_word, cap_emb.size(2)))
        cap_lens = [0] * len(images)
    # cache embeddings
    ids = list(ids)
    img_embs[:len(ids)] = (img_emb.data.cpu().numpy().copy())
    cap_embs[:len(ids),:max(lengths),:] = cap_emb.data.cpu().numpy().copy()

    for j, _ in enumerate(ids):
        cap_lens[j] = cap_len[j]
    
    return img_embs, cap_embs, cap_lens
        
        

