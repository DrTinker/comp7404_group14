import os
import numpy as np
from dataloader.loader import get_test_loader
from evaluation.evaluation import encode_data, shard_xattn
from modules.BFAN import BFAN

from utils.vocab import deserialize_vocab
import torch

def loaddata(model_path, data_path=None, split='dev'):
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

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    sims = shard_xattn(img_embs, cap_embs, cap_lens, opt, shard_size=128)

    # batch_size = img_embs.shape[0]
    # r, rt = i2t(batch_size, sims, return_ranks=True)
    # ri, rti = t2i(batch_size, sims, return_ranks=True)
    # ar = (r[0] + r[1] + r[2]) / 3
    # ari = (ri[0] + ri[1] + ri[2]) / 3
    # rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]

    return img_embs, cap_embs, cap_lens, sims