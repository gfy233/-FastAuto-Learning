#!/usr/bin/env python36
# -*- coding: utf-8 -*-

######################################################
# Elastic-Adjustment (EA) Mechanism #
######################################################

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch_position_embedding import PositionEmbedding


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size * self.world_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class Attention(Module):
    def __init__(self, hiddenSize, is_comp=False):
        super(Attention, self).__init__()
        self.hidden_size = hiddenSize
        self.is_comp = is_comp
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        if is_comp:
            self.linear_compress = nn.Linear(self.hidden_size * 2, 1, bias=True)
        else:
            self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

    def forward(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if self.is_comp:
            res = self.linear_compress(torch.cat([a, ht], 1))
        else:
            res = self.linear_transform(torch.cat([a, ht], 1))
        return res


class TAR_GNN(Module):
    def __init__(self, opt, n_node, len_max):
        super(TAR_GNN, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.opt = opt
        self.n_node = n_node
        self.len_max = len_max
        self.norm = opt.norm
        self.temperature = opt.temperature
        self.ta = opt.TA
        self.scale = opt.scale
        self.batch_size = opt.batchSize
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.position_embedding = PositionEmbedding(num_embeddings=len_max, embedding_dim=opt.position_embeddingSize,
                                                    mode=PositionEmbedding.MODE_ADD)
        self.transformerEncoderLayer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=opt.nhead,
                                                               dim_feedforward=self.hidden_size * opt.feedforward,
                                                               dropout=opt.dropout)
        self.transformerEncoder = TransformerEncoder(self.transformerEncoderLayer, opt.layer)

        self.out_size = opt.hiddenSize
        self.attention_mode = Attention(self.hidden_size, is_comp=False)
        self.attention_r = Attention(self.hidden_size, is_comp=False)
        self.attention_e = Attention(self.hidden_size, is_comp=False)
        self.attention_ht_e = Attention(self.hidden_size, is_comp=True)
        self.attention_ht_r = Attention(self.hidden_size, is_comp=True)
        self.w_re = Parameter(torch.Tensor(self.out_size, 2))

        if self.ta:
            self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # target attention

        self.head_items = []
        f = open("datasets/" + str(self.opt.split_num) + ".txt")
        hindex = np.zeros(19182)
        text = f.readline()
        while text != '':
            self.head_items.append(int(text))
            text = f.readline()
        f.close()

        for i in range(len(hindex)):
            if i + 1 in self.head_items:
                hindex[i] = 1
        self.t = trans_to_cuda(torch.as_tensor(hindex)).float()

        multilist = [[1.0 for col in range(self.hidden_size)] for row in range(self.n_node)]
        for item in self.head_items:
            l = multilist[item]
            for m in range(len(l)):
                l[m] = 0

        self.mod_mat = trans_to_cuda(torch.as_tensor(multilist))

        self.loss_function = FocalLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, items, mask):
        hidden = self.seq_hidden
        mask_sum = torch.sum(mask, 1)
        last_ht = hidden[torch.arange(mask.shape[0]).long(), mask_sum - 1]  # batch_size x latent_size
        last_two_index = mask_sum - 2
        last_two_index = (last_two_index == -1).long() + last_two_index
        last_two_ht = hidden[torch.arange(mask.shape[0]).long(), last_two_index]
        ht = (last_ht + last_two_ht) / 2

        b = self.embedding.weight[1:]

        sess_len = self.seq_hidden.shape[1]
        mod = torch.index_select(self.mod_mat, 0, trans_to_cuda(torch.as_tensor(items[0]).squeeze().long()))
        for i in np.arange(1, len(items)):
            mod = torch.cat(
                (mod, torch.index_select(self.mod_mat, 0, trans_to_cuda(torch.as_tensor(items[i]).squeeze().long()))),
                0)
        mod = torch.reshape(mod, [-1, sess_len, self.hidden_size])

        mod_hidden = self.seq_hidden + mod

        if self.norm:
            mod_shape = list(mod_hidden.size())
            mod_hidden = mod_hidden.view(-1, self.hidden_size)
            norms = torch.norm(mod_hidden, p=2, dim=1)  # l2 norm over session embedding
            mod_hidden = mod_hidden.div(norms.unsqueeze(-1).expand_as(mod_hidden))
            mod_hidden = mod_hidden.view(mod_shape)

        s_ht_r = self.attention_ht_r(mod_hidden, mask)
        head_r = torch.sigmoid(s_ht_r) #Adjustment Factor
        s_ht_e = self.attention_ht_e(mod_hidden, mask)
        head_e = torch.sigmoid(s_ht_e) #Adjustment Factor

        if self.norm:
            norms = torch.norm(self.embedding.weight, p=2, dim=1).data  # l2 norm over item embedding again for b
            self.embedding.weight.data = self.embedding.weight.data.div(
                norms.view(-1, 1).expand_as(self.embedding.weight))

        scores_r = torch.matmul(self.s_r, b.transpose(1, 0))
        scores_e = torch.matmul(self.s_e, b.transpose(1, 0))

        ht_r = head_r * self.t
        tt_r = (self.t - 1) * (head_r - 1)
        out_r = ht_r + tt_r
        ht_e = head_e * self.t
        tt_e = (self.t - 1) * (head_e - 1)
        out_e = ht_e + tt_e

        tar_scores = scores_r * self.pr * out_r + scores_e * self.pe * out_e
        san_scores = torch.matmul(ht, b.transpose(1, 0))

        if self.scale:
            tar_scores = 12 * tar_scores  # 12 is the delta factor
            san_scores = 12 * san_scores
        return tar_scores, san_scores

    def forward(self, inputs, mask, is_cl=False, pos_ebd=True):
        if self.norm and not is_cl:
            norms = torch.norm(self.embedding.weight, p=2, dim=1).data  # l2 norm over item embedding
            self.embedding.weight.data = self.embedding.weight.data.div(
                norms.view(-1, 1).expand_as(self.embedding.weight))

        seq_hidden = self.embedding(inputs)
        if pos_ebd:
            seq_hidden = self.position_embedding(seq_hidden)
        seq_hidden = seq_hidden.transpose(0, 1).contiguous()
        seq_hidden = self.transformerEncoder(seq_hidden)
        seq_hidden = seq_hidden.transpose(0, 1).contiguous()

        if self.norm and not is_cl:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.hidden_size)
            norms = torch.norm(seq_hidden, p=2, dim=1)  # l2 norm over session embedding
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1).expand_as(seq_hidden))
            seq_hidden = seq_hidden.view(seq_shape)

        self.seq_hidden = seq_hidden

        s_mode = self.attention_mode(seq_hidden, mask)
        p_re = F.softmax(torch.matmul(s_mode, self.w_re))
        p_re = p_re.transpose(1, 0)
        self.pr = torch.reshape(p_re[0], [p_re[0].shape[0], 1])
        self.pe = torch.reshape(p_re[1], [p_re[1].shape[0], 1])

        self.s_r = self.attention_r(seq_hidden, mask)
        self.s_e = self.attention_e(seq_hidden, mask)

        return seq_hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    inputs, mask, targets = data.get_slice(i)
    inputs = trans_to_cuda(torch.Tensor(inputs).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    model(inputs, mask)

    return targets, model.compute_scores(inputs, mask)


def contrastive_learning(model, i, data):
    inputs, mask, targets = data.get_slice(i)
    index = np.where((np.sum(mask, axis=1) > 1))  # filter sessions with length > 1
    inputs = inputs[index]
    filter_mask = mask[index]
    inputs_len = np.sum(filter_mask, axis=1).tolist()

    inputs_p1 = [inputs[i].tolist()[:inputs_len[i] // 2] + [0] * (model.len_max // 2 - inputs_len[i] // 2) for i in
                 range(len(inputs))]
    inputs_p2 = [inputs[i].tolist()[inputs_len[i] // 2:inputs_len[i]] + [0] * (
                model.len_max // 2 - (inputs_len[i] - inputs_len[i] // 2)) for i in range(len(inputs))]

    inputs_even = inputs[:, [k for k in range(0, model.len_max, 2)]]
    inputs_odd = inputs[:, [k for k in range(1, model.len_max, 2)]]
    filter_mask = trans_to_cuda(torch.Tensor(filter_mask).long())

    inputs_i = np.random.permutation(inputs.T).T
    inputs_j = np.random.permutation(inputs.T).T

    inputs_even = torch.Tensor(inputs_even).long()
    inputs_odd = torch.Tensor(inputs_odd).long()
    inputs_p1 = torch.Tensor(inputs_p1).long()
    inputs_p2 = torch.Tensor(inputs_p2).long()
    inputs_i = torch.Tensor(inputs_i).long()
    inputs_j = torch.Tensor(inputs_j).long()

    # move paddings to the end of the session
    x = (inputs_i == 0).float()
    _, idx = x.sort(1)
    inputs_i = inputs_i.gather(1, idx)

    x = (inputs_j == 0).float()
    _, idx = x.sort(1)
    inputs_j = inputs_j.gather(1, idx)

    inputs_i = trans_to_cuda(inputs_i)
    inputs_j = trans_to_cuda(inputs_j)
    inputs_even = trans_to_cuda(inputs_even)
    inputs_odd = trans_to_cuda(inputs_odd)
    inputs_p1 = trans_to_cuda(inputs_p1)
    inputs_p2 = trans_to_cuda(inputs_p2)

    mask_even = filter_mask[:, [k for k in range(0, model.len_max, 2)]]
    mask_odd = filter_mask[:, [k for k in range(1, model.len_max, 2)]]
    mask_p1 = [filter_mask[i].tolist()[:inputs_len[i] // 2] + [0] * (model.len_max // 2 - inputs_len[i] // 2) for i in
               range(len(inputs))]
    mask_p2 = [filter_mask[i].tolist()[inputs_len[i] // 2:inputs_len[i]] + [0] * (
                model.len_max // 2 - (inputs_len[i] - inputs_len[i] // 2)) for i in range(len(inputs))]

    mask_p1 = trans_to_cuda(torch.Tensor(mask_p1).long())
    mask_p2 = trans_to_cuda(torch.Tensor(mask_p2).long())

    out_r1 = model(inputs_i, filter_mask, is_cl=True, pos_ebd=False)
    out_r2 = model(inputs_j, filter_mask, is_cl=True, pos_ebd=False)
    out_even = model(inputs_even, mask_even, is_cl=True)
    out_odd = model(inputs_odd, mask_odd, is_cl=True)
    out_p1 = model(inputs_p1, mask_p1, is_cl=True)
    out_p2 = model(inputs_p2, mask_p2, is_cl=True)

    out_r1 = get_sess_ebd(out_r1, filter_mask)
    out_r2 = get_sess_ebd(out_r2, filter_mask)
    out_even = get_sess_ebd(out_even, mask_even)
    out_odd = get_sess_ebd(out_odd, mask_odd)
    out_p1 = get_sess_ebd(out_p1, mask_p1)
    out_p2 = get_sess_ebd(out_p2, mask_p2)

    # 计算对比损失
    criterion = NT_Xent(filter_mask.shape[0], model.temperature, 1)
    loss_even_odd = criterion(out_even, out_odd)
    loss_r1_r2 = criterion(out_r1, out_r2)
    #loss_p1_p2 = criterion(out_p1, out_p2)

    return loss_even_odd + loss_r1_r2

def get_sess_ebd(seq_hidden, mask):
    mask_sum = torch.sum(mask, 1)
    # get the embedding of the last item in the session
    out_last = seq_hidden[torch.arange(mask.shape[0]).long(), mask_sum - 1]
    last_two_index = mask_sum - 2
    last_two_index = (last_two_index == -1).long() + last_two_index  # change index -1 to 0
    out_last_two = seq_hidden[torch.arange(mask.shape[0]).long(), last_two_index]

    return (out_last + out_last_two) / 2


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    all_targets = []
    for i, j in tqdm(zip(slices, np.arange(len(slices))), total=len(slices)):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        tar_scores = scores[0]
        san_scores = scores[1]
        all_targets += [str(target) for target in targets]

        targets = trans_to_cuda(torch.Tensor(targets).long())
        tar_loss = model.loss_function(tar_scores, targets - 1)
        san_loss = model.loss_function(san_scores, targets - 1)
        rec_loss = 0.1 * tar_loss + 0.9 * san_loss
        cl_loss = contrastive_learning(model, i, train_data)
        loss = rec_loss + cl_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))

    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores_tuple = forward(model, i, test_data)
        tar_scores = scores_tuple[0]
        san_scores = scores_tuple[1]
        scores = 0.1 * tar_scores + 0.9 * san_scores
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                hit_index = np.where(score == target - 1)[0][0]
                rank = hit_index + 1
                mrr.append(1 / rank)
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr


