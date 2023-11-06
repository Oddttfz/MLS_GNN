#photo 9
#10的可视化,最经典的数据集5个
from __future__ import division
from __future__ import print_function
import time
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import argparse
import optuna
import torch
import copy
import torch.optim as optim
import dgl
import numpy as np
from data.utils import load_tensor_data, initialize_label, choose_path, parameters
from models.GCN import GCN
import torch.nn.functional as F
from loss import graph_intel_loss
from sklearn.metrics import accuracy_score


def distill_train(all_logits, dur, epoch, alpha, model, t1_model, t2_model, optimizer, conf, G, labels_one_hot, labels,
                   idx_train,
                  idx_val, idx_test, t1_optimizer, t2_optimizer,adj):
    t0 = time.time()
    model.train()
    logits,middle_feat_s = model(G.ndata['feat'],middle = True)
    logp = F.log_softmax(logits, dim=1)
    logp_soft = F.log_softmax(logits/conf['temperature'], dim=1)
    t1_logits, middle_feats_t1 = t1_model(G.ndata['feat'].float(), middle=True)
    t2_logits, middle_feats_t2 = t2_model(G.ndata['feat'].float(), middle=True)
    t1_logits_soft =F.softmax(t1_logits/conf['temperature'])
    t2_logits_soft =F.softmax(t2_logits/conf['temperature'])

    alpha1_l = torch.abs(torch.mean(torch.flatten(F.log_softmax(logp[idx_train].t().mm(t1_logits[idx_train])/logp[idx_train].shape[1]))))
    alpha2_l = torch.abs(torch.mean(torch.flatten(F.log_softmax(logp[idx_train].t().mm(t2_logits[idx_train])/logp[idx_train].shape[1]))))

    with torch.no_grad():
        beta1_l = alpha1_l/(alpha1_l + alpha2_l)
        beta2_l = alpha2_l/(alpha1_l + alpha2_l)

    cas_sum = (beta1_l *t1_logits_soft + beta2_l *t2_logits_soft)

    inter_loss =graph_intel_loss(model, t1_model, t2_model, middle_feat_s, G, G.ndata['feat'], epoch, conf,idx_train,beta1_l,beta2_l,adj)

    loss =F.kl_div(logp_soft[idx_train],cas_sum[idx_train],reduction='batchmean')+F.cross_entropy(logp[idx_train], labels[idx_train])+inter_loss


    acc_train = accuracy_score(logp[idx_train].max(1)[1].cpu().detach().numpy(),
                               labels[idx_train].cpu().detach().numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    dur.append(time.time() - t0)
    model.eval()
    logits = model(G.ndata['feat'])
    logp = F.log_softmax(logits, dim=1)
    # we save the logits for visualization later
    all_logits.append(logp.cpu().detach().numpy())
    loss_val = F.nll_loss(logp[idx_val], labels[idx_val])
    acc_val = accuracy_score(logp[idx_val].max(1)[1].cpu().detach().numpy(), labels[idx_val].cpu().detach().numpy())
    acc_test = accuracy_score(logp[idx_test].max(1)[1].cpu().detach().numpy(), labels[idx_test].cpu().detach().numpy())

    print(
        'Epoch %d | Loss: %.4f | loss_val: %.4f | acc_train: %.4f | acc_val: %.4f | acc_test: %.4f | inter_loss:%.4f |Time(s) %.4f' % (
            epoch, loss.item(), loss_val.item(), acc_train.item(), acc_val.item(), acc_test.item(), inter_loss,dur[-1]))
    return acc_val,acc_test




def model_train(conf, model, t1_model, t2_model, optimizer, G, labels_one_hot,
                labels, idx_train, idx_val, idx_test, t1_optimizer, t2_optimizer,adj):
    all_logits = []
    dur = []
    best = 0
    total_best=0
    cnt = 0
    epoch = 1
    alpha = 0
    record ={}

    while epoch < conf['max_epoch']:

        acc_val,acc_test = distill_train(all_logits, dur, epoch, alpha, model, t1_model, t2_model, optimizer, conf, G,
                                labels_one_hot, labels,
                                 idx_train, idx_val, idx_test, t1_optimizer, t2_optimizer,adj
                                )
        record[acc_val.item()] = acc_test.item()
        if acc_test>total_best:
            total_best=acc_test
        epoch += 1
        if acc_val >best:
            best = acc_val
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])

            cnt = 0
        else:
            if acc_val == best:
                state = dict([('model', copy.deepcopy(model.state_dict())),
                              ('optim', copy.deepcopy(optimizer.state_dict()))])
            cnt += 1

        if epoch == conf['max_epoch'] or cnt == conf['patience'] :
            print("Stop!!!")
            bit_list = sorted(record.keys())
            bit_list.reverse()
            acc_val_best =bit_list[0]
            acc_test_best = record[acc_val_best]
            for key in bit_list[:5]:
                value = record[key]
                print(key, value)
            break

    print("Optimization Finished!")
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    acc_test = distill_test(conf, model, G, labels, idx_test)
    if acc_test > conf['base_test']:
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, "./outputs/" + conf['dataset'] + "/GCN/model.pkl")
        print(state)
    print(acc_test)
    print(total_best)
    print("Optimization Finished!")

    return acc_val_best,acc_test_best


def distill_test(conf, model, G, labels, idx_test):
    model.eval()
    logits = model(G.ndata['feat'])
    logp = F.log_softmax(logits, dim=1)
    loss_test = F.nll_loss(logp[idx_test], labels[idx_test])
    acc_test = accuracy_score(logp[idx_test].max(1)[1].cpu().detach().numpy(), labels[idx_test].cpu().detach().numpy())
    print("Test set results:acc_test= {:.4f}".format(acc_test.item()))

    return acc_test




def raw_experiment(configs):
    output_dir, cascade_dir = choose_path(configs)
    adj, adj_sp, adj_calculate, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
        load_tensor_data(configs['seed'], configs['model_name'], configs['dataset'], configs['labelrate'],
                         configs['valrate'], configs['device'])
    byte_idx_train = torch.zeros_like(labels_one_hot, dtype=torch.bool).to(
        configs['device'])
    byte_idx_train[idx_train] = True
    G = dgl.graph((adj_sp.row, adj_sp.col)).to(configs['device'])
    G.ndata['feat'] = features

    ################################################################
    G.ndata['feat'].requires_grad_()
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    print('We have %d train_data.' % idx_train.shape)
    print('We have %d val_data.' % idx_val.shape)
    print('We have %d test_data.' % idx_test.shape)
    print('Loading cascades...')

    t1_model = model = GCN(
        g=G,
        in_feats=features.shape[1],
        n_hidden=256,  # 64
        n_classes=labels.max().item() + 1,  # 7
        n_layers=1,
        activation=F.relu,
        dropout=0.8)
    t2_model = model = GCN(
        g=G,
        in_feats=features.shape[1],
        n_hidden=256,  # 64
        n_classes=labels.max().item() + 1,  # 7
        n_layers=2,
        activation=F.relu,
        dropout=0.8)
    t3_model = model = GCN(
        g=G,
        in_feats=features.shape[1],
        n_hidden=256,  # 64
        n_classes=labels.max().item() + 1,  # 7
        n_layers=3,
        activation=F.relu,
        dropout=0.8)

    path_1 = "./outputs/" + configs['dataset'] + "/GCN/seed"+str(configs['teacher_seed'])+"/model_t1.pkl"#22
    path_2 = "./outputs/" + configs['dataset'] + "/GCN/seed"+str(configs['teacher_seed'])+"/model_t2.pkl"
    path_3 = "./outputs/" + configs['dataset'] + "/GCN/seed"+str(configs['teacher_seed'])+"/model_t3.pkl"

    checkpoint_t1 = torch.load(path_1)
    checkpoint_t2 = torch.load(path_2)
    checkpoint_t3 = torch.load(path_3)

    key_list = checkpoint_t1.keys()

    t1_model.load_state_dict(checkpoint_t1['t1_model'])
    t2_model.load_state_dict(checkpoint_t2['t2_model'])
    t3_model.load_state_dict(checkpoint_t3['t3_model'])

    t1_model = t1_model.to(configs['device'])
    t2_model = t2_model.to(configs['device'])
    t3_model = t3_model.to(configs['device'])

    t1_model.eval()
    t2_model.eval()
    t3_model.eval()

    t1_optimizer = optim.Adam(filter(lambda p: p.requires_grad, t1_model.parameters()), lr=0.01,
                              weight_decay=0.001)
    t2_optimizer = optim.Adam(filter(lambda p: p.requires_grad, t2_model.parameters()), lr=0.01,
                              weight_decay=0.001)
    t3_optimizer = optim.Adam(filter(lambda p: p.requires_grad, t3_model.parameters()), lr=0.01,
                              weight_decay=0.001)

    t1_optimizer.load_state_dict(checkpoint_t1['t1_optimizer'])
    t2_optimizer.load_state_dict(checkpoint_t2['t2_optimizer'])
    t3_optimizer.load_state_dict(checkpoint_t3['t3_optimizer'])


    t1_logits, middle_feats_t1 = t1_model(G.ndata['feat'].float(), middle=True)
    t2_logits, middle_feats_t2 = t2_model(G.ndata['feat'].float(), middle=True)
    t3_logits, middle_feats_t3 = t3_model(G.ndata['feat'].float(), middle=True)
    t1_logits_soft =F.softmax(t1_logits/configs['temperature'])
    t2_logits_soft =F.softmax(t2_logits/configs['temperature'])
    t3_logits_soft =F.softmax(t3_logits/configs['temperature'])
    print(accuracy_score(t1_logits_soft[idx_test].max(1)[1].cpu().detach().numpy(),labels[idx_test].cpu().detach().numpy()))
    print(accuracy_score(t2_logits_soft[idx_test].max(1)[1].cpu().detach().numpy(),labels[idx_test].cpu().detach().numpy()))
    print(accuracy_score(t3_logits_soft[idx_test].max(1)[1].cpu().detach().numpy(),labels[idx_test].cpu().detach().numpy()))

    output_dir, cascade_dir = choose_path(configs)

    model = GCN(
        g=G,
        in_feats=features.shape[1],
        # n_hidden=conf['hidden'],#64
        n_hidden=configs['emb_dim'],#256
        n_classes=labels.max().item() + 1,  # 7
        n_layers=1,
        activation=F.relu,
        dropout=configs['drop_out']).to(configs['device'])#citeseer=0.9


    print(f"number of parameter for student model: {parameters(model)}")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=configs['lr'],
                           weight_decay=configs['wd'])


    acc_val_best,acc_test_best = model_train(configs, model, t1_model, t2_model, optimizer, G, labels_one_hot,
                          labels, idx_train, idx_val, idx_test, t1_optimizer, t2_optimizer,adj)




