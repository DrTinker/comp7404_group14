import torch
import torch.utils.data as data
import nltk

class PrecompTexts_Flickr30k(data.Dataset):
    
    def __init__(self, data_path, data_split, vocab,):
        self.vocab = vocab
        src = data_path + '/'
        # get all caption text
        self.captions = []
        with open(src + '%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())
    
    def __getitem__(self, index):
        caption = self.captions[index]
        vocab = self.vocab

        # remove the beginning mark
        words = str(caption.strip())
        words = words.lstrip('b')  

        # Convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        tokens = nltk.tokenize.word_tokenize(words.lower())

        # add tags
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return target, index
    
    def __len__(self):
        return self.length

    