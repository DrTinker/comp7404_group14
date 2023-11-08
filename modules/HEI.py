import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import tensorflow as tf
import torch
from modules.self_attention import V_single_modal_atten, T_single_modal_atten
torch.backends.cudnn.enabled = False

# 实现特征聚合(36,1024) -> (36,)
# w (1024, 1) v*w (36, 1)
class DenseLayer(nn.Module):
    def __init__(self, embed_size):
        super(DenseLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(embed_size, 1))
        if torch.cuda.is_available():
            self.w.cuda()
            cudnn.benchmark = True
    
    def forward(self, input):
        input = input.to(torch.float32)
        sum_vector = torch.mm(input, self.w)
        # caculate a and weighted value
        vector = []
        for i, feature in enumerate(input):
            weight_value = 0
            for j in range(len(feature)):
                if sum_vector[i]==0:
                    continue
                a = feature[j]*self.w[j] / sum_vector[i]
                weight_value += a * feature[j]
            vector.append(weight_value)
        
        return torch.Tensor(vector)


class HashingLayer(nn.Module):
    def __init__(self, input_size, hash_code_length):
        super(HashingLayer, self).__init__()
        self.fc = nn.Linear(input_size, hash_code_length)
        if torch.cuda.is_available():
            self.fc.cuda()
            cudnn.benchmark = True

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)
        return x


class HEI(object):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    def __init__(self, opt, img_region_num, max_n_word):
        self.grad_clip = opt.grad_clip
        self.code_length = opt.k
        
        # self.V_self_atten_enhance = V_single_modal_atten(img_region_num, opt.embed_size, opt.use_BatchNorm, opt.activation_type, opt.dropout_rate, img_region_num)
        # self.T_self_atten_enhance = T_single_modal_atten(opt.embed_size, opt.use_BatchNorm, opt.activation_type, opt.dropout_rate)
        self.dense_image = DenseLayer(opt.embed_size)
        self.dense_text = DenseLayer(opt.embed_size)
        self.hashing_image = HashingLayer(img_region_num, opt.k)
        self.hashing_text = HashingLayer(max_n_word, opt.k)
        
        if torch.cuda.is_available():
            # self.V_self_atten_enhance.cuda()
            # self.T_self_atten_enhance.cuda()
            self.dense_image.cuda()
            self.dense_text.cuda()
            self.hashing_image.cuda()
            self.hashing_text.cuda()
            cudnn.benchmark = True
        
        params = list(self.dense_image.parameters())
        params += list(self.dense_text.parameters())
        params += list(self.hashing_image.parameters())
        params += list(self.hashing_text.parameters())
        
        self.params = params
        
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

    def train_start(self):
        self.dense_image.train()
        self.dense_text.train()
        self.hashing_image.train()
        self.hashing_text.train()
    
    def state_dict(self):
        state_dict = [
            self.dense_image.state_dict(), self.dense_text.state_dict(),
            self.hashing_image.state_dict(), self.hashing_text.state_dict()
        ]
        return state_dict

    def load_state_dict(self, state_dict):
        self.dense_image.load_state_dict(state_dict[0])
        self.dense_text.load_state_dict(state_dict[1])
        self.hashing_image.load_state_dict(state_dict[2])
        self.hashing_text.load_state_dict(state_dict[3])
        # self.hashing_image.load_state_dict(state_dict[0])
        # self.hashing_text.load_state_dict(state_dict[1])

    def forward_hash(self, image_embs, cap_embs, volatile=False):
        binary_im = []
        binary_text = []
        
        image_embs = Variable(torch.from_numpy(image_embs), volatile=volatile)
        cap_embs = Variable(torch.from_numpy(cap_embs), volatile=volatile)
        if torch.cuda.is_available():
            image_embs.cuda()
            cap_embs.cuda()
        # img_means = torch.mean(image_embs, 1)
        # cap_means = torch.mean(cap_embs, 1)
        # img_means = Variable(img_means, volatile=volatile)
        # cap_means = Variable(cap_means, volatile=volatile)
        # if torch.cuda.is_available():
        #     img_means.cuda()
        #     cap_means.cuda()
        for i in range(image_embs.shape[0]):
            ins_img = self.dense_image(image_embs[i])
            if torch.cuda.is_available():
                ins_img.cuda()

            image_code = self.hashing_image(ins_img)
            if torch.cuda.is_available():
                image_code.cuda()

            binary_im.append(image_code)
            
        for i in range(cap_embs.shape[0]):
            ins_cap = self.dense_text(cap_embs[i])
            if torch.cuda.is_available():
                ins_cap = ins_cap.cuda()

            cap_code = self.hashing_text(ins_cap)
            if torch.cuda.is_available():
                cap_code.cuda()
            
            binary_text.append(cap_code)
        return binary_im, binary_text

    def forward_loss(self, bvs, bts, score):
        """
        bv: binary code of image
        bt: binary code of text
        s: matching_score sij is the matching score of i-th image and j-th text
        ic_map: image_caption map relationship, in which every image lists its matched captions (annotated in original dataset)
        """
        k = self.code_length
        loss = 0
        for i in range(len(bvs)):
            sum_of_loss_1 = 0
            sum_of_loss_2 = 0
            sum_of_indicator = 0
            for j in range(len(bts)):
                # print('bvs[i]: ' + str(bvs[i]) + ' bts[j]: ' + str(bts[j]))
                N_p = 0
                if j >= i*5 and j < i*5+5:
                    sum_of_loss_1 += ((1 / k) * bvs[i] @ bts[j] - 1) ** 2
                    N_p += 1
                else:
                    if (1 / k) * bvs[i] @ bts[j] - score[i][j] > 0:  # 如果hashing code相乘大于matching score sij, i=1,else i=0
                        indicator = 1  # indicator function
                    else:
                        indicator = 0  # indicator function
                    sum_of_loss_2 += indicator * ((1 / k) * bvs[i] @ bts[j] - score[i][j]) ** 2
                    sum_of_indicator += indicator
            if N_p==0:
                N_p = 1
            if sum_of_indicator==0:
                sum_of_indicator = 1
            loss += (sum_of_loss_1 / N_p + sum_of_loss_2 / sum_of_indicator)  #  Ni_plus = 
        return loss

    def train_hashing(self, images, captions, score):
        binary_im, binary_text = self.forward_hash(images, captions)
        # print('binary_im: ' + str(binary_im) + ' binary_text: ' + str(binary_text))
        self.optimizer.zero_grad()
        loss = self.forward_loss(binary_im, binary_text, score)
        print("loss:")
        print(loss)

        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
