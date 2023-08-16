import os
import sys
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.modules.rnn
sys.path.append("..")
from hinter.config import args
from hinter.rrgcn import RecurrentRGCN
from rgcn import utils
from rgcn.utils import build_sub_graph, get_hist_vocab_from_triples, merge_reverse, get_hist_mask, \
    calc_filtered_mrr, calc_filtered_test_mrr, merge_hist_vocab_with_length_limit, calc_raw_mrr
from hinter.logger import Logger

from tensorboardX import SummaryWriter

experiment_type = ["result", "ablation", "sensitivity"]
run_type = experiment_type[0]

# ablation_
logger = Logger('./{}/log/{}/lr:{}_alpha:{}_rotate_weight_{}_sim_len:{}_his_len:300_gpu:{}.log'.format(
    run_type, args.dataset, args.lr, args.alpha, args.rotate_weight, args.sim_len, args.gpu), True, True, logging.INFO, logging.INFO)

def test_my(model, train_data, valid_data, test_data, history_list, hist_vocab_dict, test_list, 
            num_rels, num_nodes, use_cuda, mode, model_name):
    
    raw_mrr, raw_hits1, raw_hits3, raw_hits10, filt_mrr, filt_hits1, filt_hits3, filt_hits10 = 0, 0, 0, 0, 0, 0, 0, 0

    if mode == "test":
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        logger.info("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        logger.info("----------------------start testing-----------------------------\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    test_loss = []
    
    test_snap_len = [len(snap) for snap in test_list]
    test_len = np.sum(test_snap_len)  # test_list 里有多少个三元组

    input_list = [snap for snap in history_list[-args.test_history_len:]]
    
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        hist_mask = get_hist_mask(hist_vocab_dict, test_snap, num_nodes, num_rels)
        hist_mask = torch.from_numpy(hist_mask).to(args.gpu) if use_cuda else torch.from_numpy(hist_mask)
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)
        labels, score, rotate_loss = model.get_loss(history_glist, test_triples_input, hist_mask, use_cuda)
        loss = torch.nn.functional.cross_entropy(score, labels) + args.rotate_weight * rotate_loss + model.regularization_loss(args.reg_param)
        
        test_loss.append(loss.item())

        if mode == "test":
            r_mrr, r_hits1, r_hits3, r_hits10 = calc_raw_mrr(score, labels, hits=[1, 3, 10])
            f_mrr, f_hits1, f_hits3, f_hits10 = calc_filtered_test_mrr(num_nodes,
                                                                               score,
                                                                               torch.LongTensor(train_data),
                                                                               torch.LongTensor(valid_data),
                                                                               torch.LongTensor(test_data),
                                                                               torch.LongTensor(test_snap),
                                                                               entity="object",
                                                                               hits=[1, 3, 10])
        else:
            r_mrr, r_hits1, r_hits3, r_hits10 = calc_raw_mrr(score, labels, hits=[1, 3, 10])
            # torch.LongTensor(np.vstack((train_data, valid_data)))
            f_mrr, f_hits1, f_hits3, f_hits10 = calc_filtered_mrr(num_nodes, 
                                                                        score, 
                                                                        torch.LongTensor(train_data),
                                                                        torch.LongTensor(valid_data),
                                                                        torch.LongTensor(test_snap),
                                                                        entity="object",
                                                                        hits=[1, 3, 10])

        filt_mrr += f_mrr * len(test_triples_input)
        filt_hits1 += f_hits1 * len(test_triples_input)
        filt_hits3 += f_hits3 * len(test_triples_input)
        filt_hits10 += f_hits10 * len(test_triples_input)
        
        raw_mrr += r_mrr * len(test_triples_input)
        raw_hits1 += r_hits1 * len(test_triples_input)
        raw_hits3 += r_hits3 * len(test_triples_input)
        raw_hits10 += r_hits10 * len(test_triples_input)
    
    raw_mrr = raw_mrr / test_len
    raw_hits1 = raw_hits1 / test_len
    raw_hits3 = raw_hits3 / test_len
    raw_hits10 = raw_hits10 / test_len

    filt_mrr = filt_mrr / test_len
    filt_hits1 = filt_hits1 / test_len
    filt_hits3 = filt_hits3 / test_len
    filt_hits10 = filt_hits10 / test_len
    return raw_mrr, raw_hits1, raw_hits3, raw_hits10, filt_mrr, filt_hits1, filt_hits3, filt_hits10, test_loss


def run_experiment(args):

    if args.dataset == "ICEWS14":
        his_len = 300
    elif args.dataset == "ICEWS18":
        his_len = 200
    elif args.dataset == "ICEWS05-15":
        his_len = 3000
    elif args.dataset == "GDELT":
        his_len = 2000

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    logger.info("train set timestamps: {}".format(len(train_list)))
    logger.info("valid set timestamps: {}".format(len(valid_list)))
    logger.info("test set timestamps: {}".format(len(test_list)))

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    # 在这里设置 history_vocabulary
    train_hist_vocab = []
    valid_hist_vocab = []
    test_hist_vocab = []
    for snap in train_list:
        train_hist_vocab.append(get_hist_vocab_from_triples(snap, num_nodes, num_rels))
    for snap in valid_list:
        valid_hist_vocab.append(get_hist_vocab_from_triples(snap, num_nodes, num_rels))
    for snap in test_list:
        test_hist_vocab.append(get_hist_vocab_from_triples(snap, num_nodes, num_rels))

    tlen = len(train_hist_vocab)
    vlen = len(valid_hist_vocab)
    elen = len(test_hist_vocab)
    all_hist_vocab = train_hist_vocab + valid_hist_vocab + test_hist_vocab

    ttt = []
    for i in range(tlen + vlen + elen):
        if i < his_len:
            ttt.append(merge_hist_vocab_with_length_limit(all_hist_vocab[0 : i+1]))
        else:
            ttt.append(merge_hist_vocab_with_length_limit(all_hist_vocab[i-his_len+1 : i+1]))

    train_hist_vocab_2 = ttt[0:tlen]
    valid_hist_vocab_2 = ttt[tlen:tlen+vlen]

    model_name = "{}-num-epochs{}-lr:{}-alpha:{}-gpu{}-weight:{}"\
        .format(args.dataset, args.n_epochs, args.lr, args.alpha, args.gpu, args.rotate_weight)
    model_state_file = './{}/model/{}/'.format(run_type, args.dataset) + model_name
    logger.info("Sanity Check: stat name : {}".format(model_state_file))
    logger.info("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    # create stat
    model = RecurrentRGCN(args.alpha,
                        num_nodes,
                        num_rels,
                        args.n_hidden,
                        args.opn,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        k=args.sim_len)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    writer = SummaryWriter(log_dir='./{}/scalar/{}'.format(run_type, args.dataset))

    if args.test and os.path.exists(model_state_file):
        logger.info("--------------------model exists, start testing----------------------")
        test_hist_vocab_dict = valid_hist_vocab_2[-1]
        
        raw_mrr, raw_hits1, raw_hits3, raw_hits10, filt_mrr, filt_hits1, filt_hits3, filt_hits10, test_loss = \
            test_my(model, data.train, data.valid, data.test, train_list+valid_list, test_hist_vocab_dict, test_list, 
                    num_rels, num_nodes, use_cuda, mode="test", model_name=model_state_file)
        
        logger.info("RAW -- Test Result: MRR : {:.6f} | Hits @ 1: {:.6f} | Hits @ 3: {:.6f} | Hits @ 10: {:.6f}\n\n".format(
            raw_mrr, raw_hits1, raw_hits3, raw_hits10))
        logger.info("filter -- Test Result: MRR : {:.6f} | Hits @ 1: {:.6f} | Hits @ 3: {:.6f} | Hits @ 10: {:.6f}\n\n".format(
            filt_mrr, filt_hits1, filt_hits3, filt_hits10))
        
    elif args.test and not os.path.exists(model_state_file):
        logger.info("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(
            model_state_file))
    else:
        logger.info("-----------------------------start training-----------------------------\n")
        best_mrr = 0
        n = 0
        for epoch in range(args.n_epochs):
            # train_list = train_list + valid_list
            logger.info("epoch: {}".format(epoch))
            model.train()
            train_loss = []

            idx = [_ for _ in range(tlen)]  # range 的范围也要修改
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0 : continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:train_sample_num]

                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = output[0]

                hist_vocab_dict = train_hist_vocab_2[train_sample_num - 1] 
                output = merge_reverse(output, num_rels)
                hist_mask = get_hist_mask(hist_vocab_dict, output, num_nodes, num_rels)
                output = torch.from_numpy(output).long().cuda() if use_cuda else torch.from_numpy(output).long()
                hist_mask = torch.from_numpy(hist_mask).cuda() if use_cuda else torch.from_numpy(hist_mask)

                labels, score, rotate_loss = model.get_loss(history_glist, output, hist_mask, use_cuda)
                loss = torch.nn.functional.cross_entropy(score, labels) + args.rotate_weight * rotate_loss + model.regularization_loss(args.reg_param)
                # labels, score = model.get_loss(history_glist, output, hist_mask, use_cuda)
                # loss = torch.nn.functional.cross_entropy(score, labels) + model.regularization_loss(args.reg_param)
                
                train_loss.append(loss.item())

                writer.add_scalar('{}_loss_with_rotate_weight_{}_lr_{}_alpha_{}_gpu_{}'.format(args.dataset, args.rotate_weight, args.lr, args.alpha, args.gpu), loss.item(), n)
                n = n + 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()  # 清空优化器中所有参数的梯度信息

            # validation
            if epoch >= args.valid_epoch and (epoch - args.valid_epoch) % args.eval_every == 0:
                # do valid
                test_hist_vocab_dict = train_hist_vocab_2[-1]  

                raw_mrr, raw_hits1, raw_hits3, raw_hits10, filt_mrr, filt_hits1, filt_hits3, filt_hits10, valid_loss = \
                    test_my(model, data.train, data.valid, data.test, train_list, test_hist_vocab_dict, valid_list, 
                            num_rels, num_nodes, use_cuda, mode="train", model_name=model_state_file)
                
                logger.info("RAW -- MRR : {:.6f} | Hits @ 1: {:.6f} | Hits @ 3: {:.6f} | Hits @ 10: {:.6f}".format(
                    raw_mrr, raw_hits1, raw_hits3, raw_hits10))
                logger.info("filter -- MRR : {:.6f} | Hits @ 1: {:.6f} | Hits @ 3: {:.6f} | Hits @ 10: {:.6f}".format(
                    filt_mrr, filt_hits1, filt_hits3, filt_hits10))
                logger.info("valid loss: {}".format(np.mean(valid_loss)))
                
                # if not args.relation_evaluation:  # entity prediction evalution
                if filt_mrr <= best_mrr:
                    if epoch >= args.n_epochs:
                        break
                else:
                    best_mrr = filt_mrr
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                
            logger.info("Epoch: {} | Avg Loss: {:.6f}\n".format(epoch, np.mean(train_loss)))
  
        writer.close()
        # 下面的这条语句是所有的训练 epoch 跑完之后才会执行测试
        logger.info("train finished")
        test_hist_vocab_dict = valid_hist_vocab_2[-1]
        raw_mrr, raw_hits1, raw_hits3, raw_hits10, filt_mrr, filt_hits1, filt_hits3, filt_hits10, test_loss = \
            test_my(model, data.train, data.valid, data.test, train_list+valid_list, test_hist_vocab_dict, test_list, 
                    num_rels, num_nodes, use_cuda, mode="test", model_name=model_state_file)
        
        logger.info("RAW -- Test Result: MRR : {:.6f} | Hits @ 1: {:.6f} | Hits @ 3: {:.6f} | Hits @ 10: {:.6f}\n\n".format(
            raw_mrr, raw_hits1, raw_hits3, raw_hits10))
        logger.info("filter -- Test Result: MRR : {:.6f} | Hits @ 1: {:.6f} | Hits @ 3: {:.6f} | Hits @ 10: {:.6f}\n\n".format(
            filt_mrr, filt_hits1, filt_hits3, filt_hits10))

    return None


if __name__ == '__main__':
    logger.info('dataset: {}, weight:{}, lr: {}, alpha: {}, gpu: {}, num_epochs: {}, valid_epoch: {}, eval_every: {}\n'.format(
        args.dataset, args.rotate_weight, args.lr, args.alpha, args.gpu, args.n_epochs, args.valid_epoch, args.eval_every
    ))
    logger.info(args)
    run_experiment(args)
