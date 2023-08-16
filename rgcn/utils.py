"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
import numpy as np
import torch
import dgl
import copy
from tqdm import tqdm
import rgcn.knowledge_graph as knwlgrh
from collections import defaultdict


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


#TODO filer by groud truth in the same time snapshot not all ground truth
def sort_and_rank_time_filter(batch_a, batch_r, score, target, total_triplets):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    for i in range(len(batch_a)):
        ground = indices[i]
    indices = indices[:, 1].view(-1)
    return indices


def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    for i in range(len(batch_a)):
        ans = target[i]
        b_multi = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ground = score[i][ans]
        score[i][b_multi] = 0
        score[i][ans] = ground
    _, indices = torch.sort(score, dim=1, descending=True)  # indices : [B, number entity]
    indices = torch.nonzero(indices == target.view(-1, 1))  # indices : [B, 2] 第一列递增， 第二列表示对应的答案实体id在每一行的位置
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score

def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        # print(h, r, t)
        # print(ans)
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel+num_rels].add(src)
        r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    """
    :param node_id: node id in the large graph
    :param num_rels: number of relation
    :param src: relabeled src id
    :param rel: original rel id
    :param dst: relabeled dst id
    :param use_cuda:
    :return:
    """
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    if use_cuda:
        g = g.to(gpu) 
        g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return g

def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict=0):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        if rel_predict==1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        if rel_predict:
            filter_score_batch = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def stat_ranks(rank_list, method):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float())
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
    return mrr


def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l

def UnionFindSet(m, edges):
    """

    :param m:
    :param edges:
    :return: union number in a graph
    """
    roots = [i for i in range(m)]
    rank = [0 for i in range(m)]
    count = m

    def find(member):
        tmp = []
        while member != roots[member]:
            tmp.append(member)
            member = roots[member]
        for root in tmp:
            roots[root] = member
        return member

    for i in range(m):
        roots[i] = i
    # print ufs.roots
    for edge in edges:
        print(edge)
        start, end = edge[0], edge[1]
        parentP = find(start)
        parentQ = find(end)
        if parentP != parentQ:
            if rank[parentP] > rank[parentQ]:
                roots[parentQ] = parentP
            elif rank[parentP] < rank[parentQ]:
                roots[parentP] = parentQ
            else:
                roots[parentQ] = parentP
                rank[parentP] -= 1
            count -= 1
    return count

def append_object(e1, e2, r, d):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]:
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def load_all_answers(total_data, num_rel):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    all_subjects, all_objects = {}, {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_subjects, num_rel=num_rel)
        add_object(s, o, r, all_objects, num_rel=0)
    return all_objects, all_subjects


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    # output_label_list = []
    # for all_ans in all_ans_list:
    #     output = []
    #     ans = []
    #     for e1 in all_ans.keys():
    #         for r in all_ans[e1].keys():
    #             output.append([e1, r])
    #             ans.append(list(all_ans[e1][r]))
    #     output = torch.from_numpy(np.array(output))
    #     output_label_list.append((output, ans))
    # return output_label_list
    return all_ans_list

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t:  # 同一时刻发生的三元组
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    # 加入最后一个shapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list


def slide_list(snapshots, k=1):
    """
    :param k: padding K history for sequence stat
    :param snapshots: all snapshot
    :return:
    """
    k = k  # k=1 需要取长度k的历史，在加1长度的label
    if k > len(snapshots):
        print("ERROR: history length exceed the length of snapshot: {}>{}".format(k, len(snapshots)))
    for _ in tqdm(range(len(snapshots)-k+1)):
        yield snapshots[_: _+k]



def load_data(dataset, bfs_level=3, relabel=False):
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(dataset)
    elif dataset in ['ICEWS18', 'ICEWS14', "GDELT", "SMALL", "ICEWS14s", "ICEWS05-15","YAGO",
                     "WIKI"]:
        return knwlgrh.load_from_local("../data", dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, r = test_triples[_][0], test_triples[_][1]
            if r < num_rels:
                predict_triples.append([test_triples[_][0], r, index])
            else:
                predict_triples.append([index, r-num_rels, test_triples[_][0]])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples

def construct_snap_r(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    # for _ in range(len(test_triples)):
    #     h, r = test_triples[_][0], test_triples[_][1]
    #     if (sorted_score[_][0]-sorted_score[_][1])/sorted_score[_][0] > 0.3:
    #         if r < num_rels:
    #             predict_triples.append([h, r, indices[_][0]])

    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, t = test_triples[_][0], test_triples[_][2]
            if index < num_rels:
                predict_triples.append([h, index, t])
                #predict_triples.append([t, index+num_rels, h])
            else:
                predict_triples.append([t, index-num_rels, h])
                #predict_triples.append([t, index-num_rels, h])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples


def dilate_input(input_list, dilate_len):
    dilate_temp = []
    dilate_input_list = []
    for i in range(len(input_list)):
        if i % dilate_len == 0 and i:
            if len(dilate_temp):
                dilate_input_list.append(dilate_temp)
                dilate_temp = []
        if len(dilate_temp):
            dilate_temp = np.concatenate((dilate_temp, input_list[i]))
        else:
            dilate_temp = input_list[i]
    dilate_input_list.append(dilate_temp)
    dilate_input_list = [np.unique(_, axis=0) for _ in dilate_input_list]
    return dilate_input_list

def emb_norm(emb, epo=0.00001):
    x_norm = torch.sqrt(torch.sum(emb.pow(2), dim=1))+epo
    emb = emb/x_norm.view(-1,1)
    return emb

def shuffle(data, labels):
    shuffle_idx = np.arange(len(data))
    np.random.shuffle(shuffle_idx)
    relabel_output = data[shuffle_idx]
    labels = labels[shuffle_idx]
    return relabel_output, labels


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t)
    return a


# 下面的三个函数是新加的
#
# 生成 hist_vocab 
def get_hist_vocab_from_triples(triples, num_ents, num_rels):
    ent_history_vocabulay = defaultdict(lambda: defaultdict(int))
    # rel_history_vocabulay = defaultdict(lambda: defaultdict(int))

    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    for s, r, o in zip(src, rel, dst):
        # subject + relation 作为主键，为了预测实体
        r = r - 2 * num_rels
        if (s, r) in ent_history_vocabulay:
            if o in ent_history_vocabulay[(s, r)]:
                ent_history_vocabulay[(s, r)][o] = ent_history_vocabulay[(s, r)][o] + 1
            else:
                ent_history_vocabulay[(s, r)][o] = 1
        else:
            ent_history_vocabulay[(s, r)][o] = 1
        
        # r = r + 2 * num_rels
        # o = o - num_ents
        # # subject + object 作为主键，为了预测关系
        # if (s, o) in rel_history_vocabulay:
        #     if r in rel_history_vocabulay[(s, o)]:
        #         rel_history_vocabulay[(s, o)][r] = rel_history_vocabulay[(s, o)][r] + 1
        #     else:
        #         rel_history_vocabulay[(s, o)][r] = 1
        # else:
        #     rel_history_vocabulay[(s, o)][r] = 1
        
    return ent_history_vocabulay

# 将两个 hist_vocab 合并
def merge_hist_vocab(last_vocab, this_vocab):
    if last_vocab == None:
        return this_vocab
    else:
        # 把 last_vocab 中的记录合并到 this_vocab 中
        for s_r, inner_dict in last_vocab.items():
            if s_r not in this_vocab:
                this_vocab[s_r] = inner_dict
            else:
                for o, count in inner_dict.items():
                    if o not in this_vocab[s_r]:
                        this_vocab[s_r][o] = count
                    else:
                        this_vocab[s_r][o] = this_vocab[s_r][o] + count
    return this_vocab

# 将一个 hist_vocab 中的内容从另一个中去除
def sub_hist_vocab(last_vocab, this_vocab):
    # this_vocab - last_vocab
    # 将 last_vocab 中的记录从 this_vocab 中删除
    for s_r, inner_dict in last_vocab.items():
        # s_r 一定会在 this_vocab 中
        for o, count in inner_dict.items():
            t = this_vocab[s_r][o] - count
            this_vocab[s_r][o] = t
            assert this_vocab[s_r][o] > -1, "sub hist_vocab error"
    return this_vocab

# 将一个 hist_vocab 列表合并
def merge_hist_vocab_with_length_limit(vocab):
    if len(vocab) == 1:
        return copy.deepcopy(vocab[0])
    else:
        res = copy.deepcopy(vocab[-1])
        for i in range(len(vocab) - 1):
            merge_hist_vocab(vocab[i], res)
        return res

# 合并三元组的逆
def merge_reverse(triples, num_rels):
    inverse_triples = triples[:, [2, 1, 0]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
    all_triples = np.concatenate((triples, inverse_triples), axis=0)
    return all_triples

def get_hist_mask(hist_vocab_dict, all_triples, num_ents, num_rels):
    factors_len = len(all_triples)
    hist_mask = np.zeros((factors_len, num_ents), dtype=float)
    for i, factor in enumerate(all_triples):
        tuple_key = (factor[0], factor[1] - 2 * num_rels)
        if tuple_key in hist_vocab_dict:
            for ent in hist_vocab_dict[tuple_key]:
                hist_mask[i][ent] = hist_vocab_dict[tuple_key][ent]

    return hist_mask


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

# return MRR (raw), and Hits @ (1, 3, 10)
def calc_raw_mrr(score, labels, hits=[]):
    with torch.no_grad():

        ranks = sort_and_rank(score, labels)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

def filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    filtered_h = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t))
    # Do not consider an object if it is part of a triplet to filter
    for h in range(num_entities):
        if (h, target_r, target_t) not in triplets_to_filter:
            filtered_h.append(h)
    return torch.LongTensor(filtered_h)

def filter_t(triplets_to_filter, target_h, target_r, target_t, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    filtered_t = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t))
    # Do not consider an object if it is part of a triplet to filter
    for t in range(num_entities):
        if (target_h, target_r, t) not in triplets_to_filter:
            filtered_t.append(t)
    return torch.LongTensor(filtered_t)

def get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter, entity):
    """ Perturb object in the triplets
    """
    num_entities = num_entity
    ranks = []

    for idx in range(test_size):
        target_h = h[idx]
        target_r = r[idx]
        target_t = t[idx]
        # print('t',target_t)
        if entity == 'object':
            filtered_t = filter_t(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_t_idx = int((filtered_t == target_t).nonzero())
            _, indices = torch.sort(score[idx][filtered_t], descending=True)
            rank = int((indices == target_t_idx).nonzero())
        if entity == 'subject':
            filtered_h = filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_h_idx = int((filtered_h == target_h).nonzero())
            _, indices = torch.sort(score[idx][filtered_h], descending=True)
            rank = int((indices == target_h_idx).nonzero())

        ranks.append(rank)
    return torch.LongTensor(ranks)

def calc_filtered_mrr(num_entity, score, train_triplets, valid_triplets, test_triplets, entity, hits=[]):
    with torch.no_grad():
        h = test_triplets[:, 0]
        r = test_triplets[:, 1]
        t = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        train_triplets = torch.Tensor(train_triplets[:, 0:3])
        valid_triplets = torch.Tensor(valid_triplets[:, 0:3])
        test_triplets = torch.Tensor(test_triplets[:, 0:3])

        # train_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in train_triplets])
        # valid_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets])
        # test_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in test_triplets])

        triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets]).tolist()

        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}

        ranks = get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter, entity)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

def calc_filtered_test_mrr(num_entity, score, train_triplets, valid_triplets, valid_triplets2, test_triplets, entity, hits=[]):
    with torch.no_grad():
        h = test_triplets[:, 0]
        r = test_triplets[:, 1]
        t = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        train_triplets = torch.Tensor(train_triplets[:, 0:3])
        valid_triplets = torch.Tensor(valid_triplets[:, 0:3])
        valid_triplets2 = torch.Tensor(valid_triplets2[:, 0:3])
        test_triplets = torch.Tensor(test_triplets[:, 0:3])

        # train_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in train_triplets])
        # valid_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets])
        # valid_triplets2 = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets2])
        # test_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in test_triplets])

        triplets_to_filter = torch.cat([train_triplets, valid_triplets, valid_triplets2, test_triplets]).tolist()

        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}

        ranks = get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter, entity)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

