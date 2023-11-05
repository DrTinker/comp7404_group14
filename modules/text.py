import torch
import torch.nn as nn
import torch.nn.init
import utils.norm as norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EncoderText(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # embed captions to same size
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=True)

        self.init_weights()
        
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        cap_emb = (cap_emb[:, :, :cap_emb.size(2)//2] +
                   cap_emb[:, :, cap_emb.size(2)//2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = norm.l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len

# Text self attention 3 Sequential layers
class AttentionText(nn.Module):
    def __init__(self, embed_dim, activation_type, dropout_rate):
        """
        param embed_dim: dim of embedding space
        """
        super(AttentionText, self).__init__()

        self.fc1 = nn.Linear(embed_dim, embed_dim)  # embed visual feature to common space
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # embed memory to common space
        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights
        
        if torch.cuda.is_available():
            self.fc1.cuda()
            self.fc2.cuda()
            self.fc3.cuda()
            cudnn.benchmark = True

        if activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))

            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))

            self.embedding_3 = nn.Sequential(self.fc3)
            if torch.cuda.is_available():
                self.embedding_1.cuda()
                self.embedding_2.cuda()
                self.embedding_3.cuda()
                cudnn.benchmark = True

        elif activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
            if torch.cuda.is_available():
                self.embedding_1.cuda()
                self.embedding_2.cuda()
                self.embedding_3.cuda()
                cudnn.benchmark = True

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, u_t, m_u):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """

        W_u = self.embedding_1(u_t)

        W_u_m = self.embedding_2(m_u)
        W_u_m = W_u_m.unsqueeze(1).repeat(1, W_u.size()[1], 1)

        h_u = W_u.mul(W_u_m)

        a_u = self.embedding_3(h_u)
        a_u = a_u.squeeze(2)
        weights = self.softmax(a_u)

        u_att = ((weights.unsqueeze(2) * u_t)).sum(dim=1)

        # l2 norm
        u_att = norm.l2norm(u_att)

        return u_att, weights
