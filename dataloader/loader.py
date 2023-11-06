import torch
import torch.utils.data as data

import os
import nltk

import numpy as np


from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


'''2) Flickr30k dataset'''


class PrecompDataset_Flickr30k(data.Dataset):
    """
    Load precomputed captions and image features for f30k dataset
    """

    def __init__(self, data_path, data_split, vocab, opt):
        self.vocab = vocab
        loc = data_path + '/'

        # 1) Captions
        self.captions = []
        with open(loc + '%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # 2) Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)
        self.length = len(self.captions)
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        if data_split == 'dev' or data_split == 'test':
            self.length = 5000

        self.data_split = data_split

        '''Load the image ids for loading the corresponding concept labels'''
        self.image_ids = []
        img_file_name = loc + '%s_ids.txt' % data_split
        with open(img_file_name, 'rb') as f:
            for line in f:
                line = int(line)
                self.image_ids.append(line)

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # remove the beginning mark
        sent = str(caption.strip())
        sent = sent.lstrip('b')  

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        tokens_sent = nltk.tokenize.word_tokenize(
            sent.lower())

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        # Load the concept labels
        image_id = self.image_ids[img_id]
        image_id = str(image_id)  # convert to string

        return image, target, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption, concept_label, concept_emb) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
            - attribute_label: concept label, torch tensor of shape (concept_num);
            - attribute_input_emb: initial concept embeddings, torch tensor of shape (concept_num, word_emb_dim);
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        attribute_label: torch tensor of shape (concept_num);
        attribute_input_emb: torch tensor of shape (concept_num, word_emb_dim);
        lengths: list; valid length for each padded caption.
        ids: index
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, ids, img_ids = zip(*data)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=20, drop_last=True):
    if 'f30k' in data_path:
        dset = PrecompDataset_Flickr30k(data_path,
                                        data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              drop_last=drop_last)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)

    # concept file path
    orig_dpath = os.path.join(opt.orig_img_path, opt.orig_data_name)

    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers, drop_last=True)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers, drop_last=False)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)

    orig_dpath = os.path.join(opt.orig_img_path, opt.orig_data_name)

    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers, drop_last=False)

    return test_loader
