import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn
import pdb
import numpy as np
from torch.nn.init import xavier_normal_, xavier_uniform_
import random

# Base Class

class Model(torch.nn.Module):

    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu
        self.BCEloss = torch.nn.BCELoss()
        self.LW = 1e-3

    def bilinear_initialization(self):
        #Initialize relation matrix

        Rel_init = np.zeros((self.n_r,self.embedding_rel_dim,self.embedding_rel_dim))
        for k in range(self.n_r):
            for i in range(self.embedding_rel_dim):
                for j in range(self.embedding_rel_dim):
                    if(i==j):
                        Rel_init[k][i][j] = 1+random.uniform(-0.2,0.2)
                    else:
                        Rel_init[k][i][j] = random.uniform(-0.2,0.2)
        Rel_init = Rel_init.reshape(-1, self.embedding_rel_dim**2)
        return Rel_init

    def normalize_embeddings(self):
        for e in self.embeddings:
            e.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def bce_loss(self, y_pred, y_true, average=True):
        
        norm_word = torch.norm(self.embed_words.weight, 2, 1)
        norm_rel = torch.norm(self.embed_rel.weight, 2, 1)
        y_true = Variable(torch.from_numpy(y_true.astype(np.float32)).cuda()) if self.gpu else Variable(torch.from_numpy(y_true.astype(np.float32)))
        y_pred = F.sigmoid(y_pred)
        loss = self.BCEloss(y_pred, y_true)
        # Penalize when embeddings norms larger than one
        nlp1 = torch.sum(torch.clamp(norm_word - 1, min=0))
        nlp2 = torch.sum(torch.clamp(norm_rel - 1, min=0))
        if average:
            nlp1 /= norm_word.size(0)
            nlp2 /= norm_rel.size(0)

        return loss + self.LW*nlp1 + self.LW*nlp2

    def predict_proba(self, pred_score):
        pred_score = pred_score.view(-1, 1)
        pred_prob = F.sigmoid(pred_score)
        return pred_prob.cpu().data.numpy() if self.gpu else pred_prob.data.numpy()


# Bilinear RESCAL Model
class BilinearModel(Model):

    def __init__(self, embedding_dim, embedding_rel_dim, weights, n_r, lw, batch_size, input_dropout=0.2, gpu=True):
        super(BilinearModel, self).__init__(gpu)
        self.embed_words = torch.nn.Embedding(len(weights), embedding_dim, padding_idx=0)
        self.embed_rel = torch.nn.Embedding(n_r, embedding_rel_dim**2, padding_idx=0)
        
        self.batch_size = batch_size
        self.embedding_rel_dim = embedding_rel_dim
        self.embedding_dim = embedding_dim
        self.n_r = n_r
        
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.gpu = gpu
        self.LW = lw
        self.transform_term = nn.Sequential(
            nn.Linear(embedding_dim, embedding_rel_dim),
            nn.Tanh(),
        )
        self.init(weights)

        if self.gpu:
            self.cuda()

    def init(self, weights):
        self.embed_words.weight.data.copy_(torch.from_numpy(weights))
        rel_mat = self.bilinear_initialization()
        self.embed_rel.weight.data.copy_(torch.from_numpy(rel_mat))
        # xavier_normal_(self.embed_rel.weight.data)
        # Xavier init
        for p in self.transform_term.modules():
            if isinstance(p, nn.Linear):
                in_dim = p.weight.size(0)
                p.weight.data.normal_(0, 1/np.sqrt(in_dim/2))

    def forward(self, s, o, p):
        freq_s = Variable(torch.from_numpy(s.astype(bool).sum(axis=1)).type(torch.FloatTensor), requires_grad=False)
        freq_o = Variable(torch.from_numpy(o.astype(bool).sum(axis=1)).type(torch.FloatTensor), requires_grad=False)
        freq_s = freq_s.cuda() if self.gpu else freq_s
        freq_o = freq_o.cuda() if self.gpu else freq_o

        s = Variable(torch.from_numpy(s)).long()
        s = s.cuda() if self.gpu else s
        
        o = Variable(torch.from_numpy(o)).long()
        o = o.cuda() if self.gpu else o
        
        p = Variable(torch.from_numpy(p)).long()
        p = p.cuda() if self.gpu else p
        
        s_embedded = self.embed_words(s).sum(dim=1)
        s_embedded = s_embedded.mul(freq_s.unsqueeze(1))
        o_embedded = self.embed_words(o).sum(dim=1)
        o_embedded = o_embedded.mul(freq_o.unsqueeze(1))

        s_embedded = self.transform_term(s_embedded)
        o_embedded = self.transform_term(o_embedded)

        s_embedded = s_embedded.view(-1, self.embedding_rel_dim, 1)
        o_embedded = o_embedded.view(-1, self.embedding_rel_dim, 1)
        p_embedded = self.embed_rel(p)
        p_embedded = p_embedded.view(-1, self.embedding_rel_dim, self.embedding_rel_dim)

        pred = torch.bmm(torch.transpose(s_embedded,1,2), p_embedded)
        pred = torch.bmm(pred, o_embedded)
        pred = pred.view(-1, 1)
        return pred


class LSTM_BilinearModel(Model):
    def __init__(self, embedding_dim, embedding_rel_dim, maxlen_s, maxlen_o, weights, n_r, lw, batch_size, input_dropout=0.2, gpu=True):
        super(LSTM_BilinearModel, self).__init__(gpu)
        self.embed_words = torch.nn.Embedding(len(weights), embedding_dim, padding_idx=0)
        self.embed_rel = torch.nn.Embedding(n_r, embedding_rel_dim**2, padding_idx=0)
        self.batch_size = batch_size
        self.maxlen_s = maxlen_s
        self.maxlen_o = maxlen_o
        self.n_r = n_r
        self.embedding_rel_dim = embedding_rel_dim
        self.lstm_s = nn.LSTM(input_size=embedding_dim, hidden_size=self.embedding_rel_dim, dropout=input_dropout)
        self.lstm_o = nn.LSTM(input_size=embedding_dim, hidden_size=self.embedding_rel_dim, dropout=input_dropout)
        self.LW = lw
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.loss = torch.nn.BCELoss()
        self.init(weights)
        self.gpu = gpu
        self.hidden1 = self.init_hidden1()
        self.hidden2 = self.init_hidden2()
        if self.gpu:
            self.cuda()

    def init(self, weights):
        self.embed_words.weight.data.copy_(torch.from_numpy(weights))
        rel_mat = self.bilinear_initialization()
        self.embed_rel.weight.data.copy_(torch.from_numpy(rel_mat))

    def init_hidden1(self):
        # the first is the hidden h
        # the second is the cell  c
        h = Variable(torch.zeros(1, self.batch_size, self.embedding_rel_dim))
        h = h.cuda() if self.gpu else h
        c = Variable(torch.zeros(1, self.batch_size, self.embedding_rel_dim))
        c = c.cuda() if self.gpu else c

        return (h, c)

    def init_hidden2(self):
        # the first is the hidden h
        # the second is the cell  c
        h1 = Variable(torch.zeros(1, self.batch_size, self.embedding_rel_dim))
        h1 = h1.cuda() if self.gpu else h1
        c1 = Variable(torch.zeros(1, self.batch_size, self.embedding_rel_dim))
        c1 = c1.cuda() if self.gpu else c1

        return (h1, c1)

    def forward(self, s, o, p):
        s = Variable(torch.from_numpy(s)).long()
        s = s.cuda() if self.gpu else s
        
        o = Variable(torch.from_numpy(o)).long()
        o = o.cuda() if self.gpu else o
        
        p = Variable(torch.from_numpy(p)).long()
        p = p.cuda() if self.gpu else p
        
        s_embedded = self.embed_words(s)
        o_embedded = self.embed_words(o)
        p_embedded = self.embed_rel(p)
        s_embedded = s_embedded.view(self.maxlen_s, self.batch_size, -1)
        o_embedded = o_embedded.view(self.maxlen_o, self.batch_size, -1)

        lstm_s_out, self.hidden1 = self.lstm_s(s_embedded, self.hidden1)
        lstm_o_out, self.hidden2 = self.lstm_o(o_embedded, self.hidden2)
        s_embedded = lstm_s_out[-1]
        o_embedded = lstm_o_out[-1]

        s_embedded = s_embedded.view(-1, self.embedding_rel_dim, 1)
        o_embedded = o_embedded.view(-1, self.embedding_rel_dim, 1)

        p_embedded = self.embed_rel(p).view(-1, self.embedding_rel_dim, self.embedding_rel_dim)

        pred = torch.bmm(torch.transpose(s_embedded,1,2), p_embedded)
        pred = torch.bmm(pred, o_embedded)
        pred = pred.view(-1, 1)
        return pred


class Avg_DistMult(Model):
    def __init__(self, embedding_dim, embedding_rel_dim, weights, n_r, lw, batch_size, input_dropout=0.2, gpu=True):
        super(Avg_DistMult, self).__init__(gpu)
        self.embed_words = torch.nn.Embedding(len(weights), embedding_dim, padding_idx=0)
        self.embed_rel = torch.nn.Embedding(n_r, embedding_rel_dim, padding_idx=0)
        self.batch_size = batch_size
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.loss = torch.nn.BCELoss()
        self.n_r = n_r

        self.gpu = gpu
        self.LW = lw
        self.transform_term = nn.Sequential(
            nn.Linear(embedding_dim, embedding_rel_dim),
            nn.ReLU(),
        )
        self.init(weights)

        if self.gpu:
            self.cuda()

    def init(self, weights):
        self.embed_words.weight.data.copy_(torch.from_numpy(weights))
        xavier_normal_(self.embed_rel.weight.data)
        # Xavier init
        for p in self.transform_term.modules():
            if isinstance(p, nn.Linear):
                in_dim = p.weight.size(0)
                p.weight.data.normal_(0, 1/np.sqrt(in_dim/2))

    def forward(self, s, o, p):
        freq_s = Variable(torch.from_numpy(s.astype(bool).sum(axis=1)).type(torch.FloatTensor), requires_grad=False)
        freq_o = Variable(torch.from_numpy(o.astype(bool).sum(axis=1)).type(torch.FloatTensor), requires_grad=False)
        freq_s = freq_s.cuda() if self.gpu else freq_s
        freq_o = freq_o.cuda() if self.gpu else freq_o

        s = Variable(torch.from_numpy(s)).long()
        s = s.cuda() if self.gpu else s
        
        o = Variable(torch.from_numpy(o)).long()
        o = o.cuda() if self.gpu else o
        
        p = Variable(torch.from_numpy(p)).long()
        p = p.cuda() if self.gpu else p
        
        s_embedded = self.embed_words(s).sum(dim=1)
        s_embedded = s_embedded.mul(freq_s.unsqueeze(1))
        o_embedded = self.embed_words(o).sum(dim=1)
        o_embedded = o_embedded.mul(freq_o.unsqueeze(1))
        p_embedded = self.embed_rel(p)

        s_embedded = self.transform_term(s_embedded)
        o_embedded = self.transform_term(o_embedded)
        s_embedded = self.inp_drop(s_embedded)
        o_embedded = self.inp_drop(o_embedded)
        p_embedded = self.inp_drop(p_embedded)
        pred = torch.sum(s_embedded * p_embedded * o_embedded, 1)
        
        return pred


class LSTM_DistMult(Model):
    def __init__(self, embedding_dim, embedding_rel_dim, weights, n_r, lw, batch_size, input_dropout=0.2, gpu=True):
        super(LSTM_DistMult, self).__init__(gpu)
        self.embed_words = torch.nn.Embedding(len(weights), embedding_dim, padding_idx=0)
        self.embed_rel = torch.nn.Embedding(n_r, embedding_rel_dim, padding_idx=0)
        self.batch_size = batch_size
        self.embedding_rel_dim = embedding_rel_dim
        self.maxlen_s = maxlen_s
        self.maxlen_o = maxlen_o
        self.lstm_s = nn.LSTM(embedding_dim, self.embedding_rel_dim)
        self.lstm_o = nn.LSTM(embedding_dim, self.embedding_rel_dim)
        self.LW = lw
        self.n_r = n_r
        
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.loss = torch.nn.BCELoss()
        self.hidden1 = self.init_hidden1()
        self.hidden2 = self.init_hidden2()
        self.init(weights)
        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def init(self, weights):
        self.embed_words.weight.data.copy_(torch.from_numpy(weights))
        xavier_normal_(self.embed_rel.weight.data)


    def init_hidden1(self):
        # the first is the hidden h
        # the second is the cell  c
        h = Variable(torch.zeros(1, self.batch_size, self.embedding_rel_dim))
        h = h.cuda() if self.gpu else h
        c = Variable(torch.zeros(1, self.batch_size, self.embedding_rel_dim))
        c = c.cuda() if self.gpu else c

        return (h, c)

    def init_hidden2(self):
        # the first is the hidden h
        # the second is the cell  c
        h1 = Variable(torch.zeros(1, self.batch_size, self.embedding_rel_dim))
        h1 = h1.cuda() if self.gpu else h1
        c1 = Variable(torch.zeros(1, self.batch_size, self.embedding_rel_dim))
        c1 = c1.cuda() if self.gpu else c1

        return (h1, c1)

    def forward(self, s, o, p):
        s = Variable(torch.from_numpy(s)).long()
        s = s.cuda() if self.gpu else s
        
        o = Variable(torch.from_numpy(o)).long()
        o = o.cuda() if self.gpu else o
        
        p = Variable(torch.from_numpy(p)).long()
        p = p.cuda() if self.gpu else p
        
        s_embedded = self.embed_words(s)
        o_embedded = self.embed_words(o)
        p_embedded = self.embed_rel(p)

        s_embedded = s_embedded.view(self.maxlen_s, self.batch_size, -1)
        o_embedded = o_embedded.view(self.maxlen_o, self.batch_size, -1)

        lstm_s_out, self.hidden1 = self.lstm_s(s_embedded, self.hidden1)
        lstm_o_out, self.hidden2 = self.lstm_o(o_embedded, self.hidden2)
        
        s_embedded = lstm_s_out[-1]
        o_embedded = lstm_o_out[-1]

        pred = torch.sum(s_embedded * p_embedded * o_embedded, 1)
        return pred


class LSTM_ERMLP(Model):
    def __init__(self, embedding_dim, embedding_rel_dim, mlp_hidden, maxlen_s, maxlen_o, weights, lw, n_r, input_dropout=0.2, gpu=True):
        super(LSTM_ERMLP, self).__init__(gpu)
        self.embed_words = torch.nn.Embedding(len(weights), embedding_dim, padding_idx=0)
        self.embed_rel = torch.nn.Embedding(n_r, embedding_rel_dim, padding_idx=0)
        self.gpu = gpu
        self.batch_size = batch_size
        self.embedding_rel_dim = embedding_rel_dim
        self.maxlen_s = maxlen_s
        self.maxlen_o = maxlen_o
        self.lstm_s = nn.LSTM(embedding_dim, self.embedding_rel_dim)
        self.lstm_o = nn.LSTM(embedding_dim, self.embedding_rel_dim)

        self.LW = lw
        self.n_r = n_r
        
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.loss = torch.nn.BCELoss()
        self.hidden1 = self.init_hidden1()
        self.hidden2 = self.init_hidden2()        
        self.init(weights)

        self.mlp = nn.Sequential(
            nn.Linear(2*self.lstm_dim+embedding_rel_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.inp_drop),
            nn.Linear(mlp_hidden, 1),
        )
        if self.gpu:
            self.cuda()

    def init(self, weights):
        self.embed_words.weight.data.copy_(torch.from_numpy(weights))
        xavier_normal_(self.embed_rel.weight.data)

        for p in self.mlp.modules():
            if isinstance(p, nn.Linear):
                in_dim = p.weight.size(0)
                p.weight.data.normal_(0, 1/np.sqrt(in_dim/2))

    def init_hidden1(self):
        # the first is the hidden h
        # the second is the cell  c
        h = Variable(torch.zeros(1, self.batch_size, self.k))
        h = h.cuda() if self.gpu else h
        c = Variable(torch.zeros(1, self.batch_size, self.k))
        c = c.cuda() if self.gpu else c

        return (h, c)

    def init_hidden2(self):
        # the first is the hidden h
        # the second is the cell  c
        h1 = Variable(torch.zeros(1, self.batch_size, self.embedding_rel_dim))
        h1 = h1.cuda() if self.gpu else h1
        c1 = Variable(torch.zeros(1, self.batch_size, self.embedding_rel_dim))
        c1 = c1.cuda() if self.gpu else c1

        return (h1, c1)

    def forward(self, s, o, p):
        s = Variable(torch.from_numpy(s)).long()
        s = s.cuda() if self.gpu else s
        
        o = Variable(torch.from_numpy(o)).long()
        o = o.cuda() if self.gpu else o
        
        p = Variable(torch.from_numpy(p)).long()
        p = p.cuda() if self.gpu else p
        
        s_embedded = self.embed_words(s)
        o_embedded = self.embed_words(o)
        p_embedded = self.embed_rel(p)

        s_embedded = s_embedded.view(self.maxlen_s, self.batch_size, -1)
        o_embedded = o_embedded.view(self.maxlen_o, self.batch_size, -1)

        lstm_s_out, self.hidden1 = self.lstm_s(s_embedded, self.hidden1)
        lstm_o_out, self.hidden2 = self.lstm_o(o_embedded, self.hidden2)
        
        s_embedded = lstm_s_out[-1]
        o_embedded = lstm_o_out[-1]

        phi = torch.cat([s_embedded, o_embedded, p_embedded], 1)
        pred = self.mlp(pred)        
        return pred

class ERMLP_avg(torch.nn.Module):
    def __init__(self, embedding_dim, embedding_rel_dim, weights, mlp_hidden, n_r, lw, batch_size, input_dropout=0.2, gpu=True):
        super(DistMult, self).__init__()
        self.embed_words = torch.nn.Embedding(len(weights), embedding_dim, padding_idx=0)
        self.embed_rel = torch.nn.Embedding(n_r, embedding_rel_dim, padding_idx=0)
        self.batch_size = batch_size
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.loss = torch.nn.BCELoss()
        self.gpu = gpu
        self.LW = lw
        self.transform_term = nn.Sequential(
            nn.Linear(embedding_dim, embedding_rel_dim),
            nn.ReLU(),
        )
        self.init(weights)

        self.mlp = nn.Sequential(
            nn.Linear(2*self.lstm_dim+embedding_rel_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.inp_drop),
            nn.Linear(mlp_hidden, 1),
        )
        if self.gpu:
            self.cuda()

    def init(self, weights):
        self.embed_words.weight.data.copy_(torch.from_numpy(weights))
        xavier_normal_(self.embed_rel.weight.data)
        # Xavier init
        for p in self.transform_term.modules():
            if isinstance(p, nn.Linear):
                in_dim = p.weight.size(0)
                p.weight.data.normal_(0, 1/np.sqrt(in_dim/2))


    def forward(self, s, o, p):
        freq_s = Variable(torch.from_numpy(s.astype(bool).sum(axis=1)).type(torch.FloatTensor), requires_grad=False)
        freq_o = Variable(torch.from_numpy(o.astype(bool).sum(axis=1)).type(torch.FloatTensor), requires_grad=False)
        freq_s = freq_s.cuda() if self.gpu else freq_s
        freq_o = freq_o.cuda() if self.gpu else freq_o

        s = Variable(torch.from_numpy(s)).long()
        s = s.cuda() if self.gpu else s
        
        o = Variable(torch.from_numpy(o)).long()
        o = o.cuda() if self.gpu else o
        
        p = Variable(torch.from_numpy(p)).long()
        p = p.cuda() if self.gpu else p
        
        s_embedded = self.embed_words(s).sum(dim=1)
        s_embedded = s_embedded.mul(freq_s.unsqueeze(1))
        o_embedded = self.embed_words(o).sum(dim=1)
        o_embedded = o_embedded.mul(freq_o.unsqueeze(1))
        p_embedded = self.embed_rel(p)

        s_embedded = self.transform_term(s_embedded)
        o_embedded = self.transform_term(o_embedded)
        s_embedded = self.inp_drop(s_embedded)
        o_embedded = self.inp_drop(o_embedded)
        p_embedded = self.inp_drop(p_embedded)
        phi = torch.cat([s_embedded, o_embedded, p_embedded], 1)
        pred = self.mlp(pred)
        return pred
