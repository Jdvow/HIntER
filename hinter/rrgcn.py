import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from rgcn.layers import UnionRGCNLayer
from hinter.model import BaseRGCN

class TimeEncode(nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = nn.Linear(1, dimension)

    # torch.nn.Parameter 对象能够自动进行梯度更新和优化
    self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output

class Copy_mode(nn.Module):
    def __init__(self, hidden_dim, output_dim, use_cuda, gpu):
        super(Copy_mode, self).__init__()
        self.hidden_dim = hidden_dim

        self.tanh = nn.Tanh()
        self.W_s = nn.Linear(hidden_dim * 2, output_dim)
        # self.bn0 = torch.nn.BatchNorm1d(output_dim)
        self.use_cuda = use_cuda
        self.gpu = gpu

    def forward(self, ent_embed, rel_embed, copy_vocabulary):
        
        m_t = torch.cat((ent_embed, rel_embed), dim=1)

        q_s = self.tanh(self.W_s(m_t))
        if self.use_cuda:
            encoded_mask = torch.Tensor(np.array(copy_vocabulary.cpu() == 0, dtype=float) * (-100))
            encoded_mask = encoded_mask.to(self.gpu)
        else:
            encoded_mask = torch.Tensor(np.array(copy_vocabulary == 0, dtype=float) * (-100))

        score_c = q_s + encoded_mask

        return F.softmax(score_c, dim=1)

class Generate_mode(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super(Generate_mode, self).__init__()
        # weights
        self.W_mlp = nn.Linear(hidden_size * 2, output_dim)

    def forward(self, ent_embed, rel_embed):
        m_t = torch.cat((ent_embed, rel_embed), dim=1)

        score_g = self.W_mlp(m_t)

        return F.softmax(score_g, dim=1)

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, 
                              activation=act, dropout=self.dropout, self_loop=self.self_loop, 
                              skip_connect=sc, rel_emb=self.rel_emb)


    def forward(self, g, init_ent_emb, init_rel_emb):
        # if self.encoder_name == "uvrgcn":
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        x, r = init_ent_emb, init_rel_emb
        for i, layer in enumerate(self.layers):
            layer(g, [], r[i])
        return g.ndata.pop('h')

class RecurrentRGCN(nn.Module):
    def __init__(self, alpha, num_ents, num_rels, h_dim, opn, num_bases=-1, num_basis=-1, num_hidden_layers=1, dropout=0, 
                 self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0, hidden_dropout=0, feat_dropout=0, 
                 use_cuda=False, gpu = 0, k=1):
        super(RecurrentRGCN, self).__init__()

        self.alpha = alpha
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.relation_evolve = False
        self.emb_rel = None
        self.use_cuda = use_cuda
        self.gpu = gpu
        self.k = k
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([6.0]), 
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / h_dim]), 
            requires_grad=False
        )

        # self.W_s = nn.Linear(hidden_dim * 2, output_dim)
        self.linear = nn.Linear(h_dim, h_dim // 2)
        # self.linear = nn.Linear(200, 100)

        self.time_encode = TimeEncode(h_dim)

        # 新加的 CyGnet 模型中的 copy_mode 和 generate_mode
        self.copy_mode = Copy_mode(h_dim, num_ents, use_cuda, gpu)
        self.generate_mode = Generate_mode(h_dim, num_ents)

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             self.opn,
                             self.emb_rel,
                             use_cuda)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)                                 

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)
        print("self.k = {}".format(self.k))
 

    def forward(self, g_list, use_cuda):

        self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)    # 第1层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  # 第2层输出==下一时刻第一层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            # 
            time_weight = torch.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1-time_weight) * self.h
            history_embs.append(self.h)
        return history_embs, self.h_0

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.emb_rel.pow(2)) + torch.mean(self.dynamic_emb.pow(2))
        return regularization_loss * reg_param

    def RE_negative_sample(self, triples, hist_mask):
        sample_res = torch.Tensor().to(self.gpu) if self.use_cuda else torch.Tensor()
        for i, row in enumerate(hist_mask):
            nonzero_index = torch.nonzero(row).squeeze()

            all_nodes = torch.arange(self.num_ents).to(self.gpu) if self.use_cuda else torch.arange(self.num_ents)
            remaining_nodes = all_nodes[~torch.isin(all_nodes, nonzero_index)]
            sampled_nodes = random.sample(remaining_nodes.tolist(), 10)
            sampled_nodes_tensor = torch.tensor(sampled_nodes).to(self.gpu) if self.use_cuda else torch.tensor(sampled_nodes)
            sample_res = torch.cat((sample_res, sampled_nodes_tensor.unsqueeze(0)), dim=0)
        
        return sample_res.to(torch.long)

    def RE(self, head, relation, tail):
        # head, relation, tail 都要是三维的 [batch_size, 1, h_dim]
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        relation = self.linear(relation)
        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)

        return score

    def RE_loss(self, head, relation, positive_tails, negative_tails):
        positive_score = self.RE(head, relation, positive_tails)
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)
        
        negative_score = self.RE(head, relation, negative_tails)
        negative_score = F.logsigmoid(-negative_score).mean(dim = 1)
        
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
        
        loss = (positive_sample_loss + negative_sample_loss)/2
        
        return loss

    def search_similary(self, node_embs, rel_embs, pred_triples, hist_mask):
        hist_mask_cp = hist_mask.clone()
        for i, triple in enumerate(pred_triples):
            head = node_embs[triple[0]]
            heads = head.repeat(self.num_ents, 1).unsqueeze(1)
            relation = rel_embs[triple[1]]
            relations = relation.repeat(self.num_ents, 1).unsqueeze(1)

            tails = node_embs.unsqueeze(1)

            p_score = self.RE(heads, relations, tails).view(-1)

            topk_val, topk_idx = torch.topk(p_score, k=self.k)
            hist_mask_cp[i][topk_idx] = 1

        return hist_mask_cp

    def get_loss(self, glist, pred_triples, hist_mask, use_cuda):
        evolve_embs, r_emb = self.forward(glist, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        r_emb = self.emb_rel
        pre_emb = self.dynamic_emb

        labels = pred_triples[:, 2]
        s_idx = pred_triples[:, 0]      # 查询（s, r, ?, t+1）中 s 的 id
        r_idx = pred_triples[:, 1]
        pred_sub_emb = pre_emb[s_idx]  # 查询（s, r, ?, t+1）中 r 的 embedding
        pred_rel_emb = r_emb[r_idx]    # 查询（s, r, ?, t+1）中 s 的 embedding

        hist_mask_cp = self.search_similary(pre_emb, r_emb, pred_triples, hist_mask)

        score_g = self.generate_mode(pred_sub_emb, pred_rel_emb)
        # score_c = self.copy_mode(pred_sub_emb, pred_rel_emb, hist_mask)
        score_c = self.copy_mode(pred_sub_emb, pred_rel_emb, hist_mask_cp)

        a = self.alpha
        score = score_c * a + score_g * (1-a)

        # return labels, score

        negative_samples = self.RE_negative_sample(pred_triples, hist_mask_cp)
        head = pre_emb[pred_triples[:, 0]].unsqueeze(1)
        relation = r_emb[pred_triples[:, 1]].unsqueeze(1)
        positive_tails = pre_emb[pred_triples[:, 2]].unsqueeze(1)
        negative_tails = pre_emb[negative_samples]

        rotate_loss = self.RE_loss(head, relation, positive_tails, negative_tails)


        return labels, score, rotate_loss