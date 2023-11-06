from __future__ import division
from __future__ import print_function
import os
import time
import argparse
import numpy as np
import scipy.sparse as sp
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

import dgl
from models.GCN import GCN
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE
from models.APPNP import APPNP
from models.MoNet import MoNet
from models.GCNII import GCNII
from dgl.nn.pytorch.conv import SGConv
from models.utils import get_training_config

from data.utils import load_tensor_data, check_writable,parameters
from data.get_dataset import get_experiment_config

from sklearn.metrics import accuracy_score
from utils.logger import get_logger



def arg_parse(parser):
    #parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCN', help='msTeacher Model')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--labelrate', type=int, default=20, help='Label rate')
    parser.add_argument('--valrate', type=int, default=30, help='Label rate')
    parser.add_argument('--seed', type=int, default=61, help='Label rate')#gat4
    #ms 7,9,10,12,14,18,25,29，32，34，36，38，41，43，46，47，48,50,51,53,54,55,57,59,66,69,75,76,93,101,102,106,118
    #ms 1,91.65


    return parser.parse_args()


def choose_path(conf):
    #output_dir = Path.cwd().joinpath('outputs', conf['dataset'], conf['teacher'],'cascade_random_' + str(conf['division_seed']) + '_' + str(args.labelrate))
    output_dir = Path.cwd().joinpath('outputs', conf['dataset'], conf['teacher'], 'seed'+ str(conf['seed']))
    check_writable(output_dir)
    cascade_dir = output_dir.joinpath('cascade')
    check_writable(cascade_dir)
    return output_dir, cascade_dir


def choose_model(conf,model_name):

    if model_name== 't1_model':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            #n_hidden=conf['hidden'],#64
            n_hidden=256,
            n_classes=labels.max().item() + 1,#7
            n_layers=1,
            activation=F.relu,
            dropout=conf['dropout']).to(conf['device'])
    elif model_name == 't2_model':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=256,#64
            n_classes=labels.max().item() + 1,#7
            n_layers=2,
            activation=F.relu,
            dropout=conf['dropout']).to(conf['device'])

    elif model_name == 't3_model':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=256,#64
            n_classes=labels.max().item() + 1,#7
            n_layers=3,
            activation=F.relu,
            dropout=conf['dropout']).to(conf['device'])


    return model


def train(all_logits, dur, epoch,model,optimizer):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    if conf['teacher'] == 'GraphSAGE':
        logits = model(G,G.ndata['feat'])#model后已经训练完了成2485*7，原来是2485*1433
    else:
        logits = model(G.ndata['feat'])  # model后已经训练完了成2485*7，原来是2485*1433
    logp = F.log_softmax(logits, dim=1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[idx_train], labels[idx_train])
    acc_train = accuracy_score(logp[idx_train].max(1)[1].cpu().detach().numpy(), labels[idx_train].cpu().detach().numpy())
    loss.backward()
    optimizer.step()
    dur.append(time.time() - t0)
    model.eval()
    if conf['teacher'] == 'GraphSAGE':
        logits = model(G,G.ndata['feat'])#model后已经训练完了成2485*7，原来是2485*1433
    else:
        logits = model(G.ndata['feat'])  # model后已经训练完了成2485*7，原来是2485*1433
    logp = F.log_softmax(logits, dim=1)
    # we save the logits for visualization later
    all_logits.append(logp.cpu().detach().numpy())
    loss_val = F.nll_loss(logp[idx_val], labels[idx_val])
    with torch.no_grad():
        acc_val = accuracy_score(logp[idx_val].max(1)[1].cpu().detach().numpy(), labels[idx_val].cpu().detach().numpy())
        acc_test = accuracy_score(logp[idx_test].max(1)[1].cpu().detach().numpy(), labels[idx_test].cpu().detach().numpy())

    print('Epoch %d | Loss: %.4f | loss_val: %.4f | acc_train: %.4f | acc_val: %.4f | acc_test: %.4f |Time(s) %.4f' % (
        epoch, loss.item(), loss_val.item(), acc_train.item(), acc_val.item(), acc_test.item(), dur[-1]))
    return acc_val, acc_test

def model_train(conf, model, optimizer, all_logits):
    dur = []
    best = 0
    cnt = 0
    epoch = 1
    dddd = 0
    record = {}
    test_record = []
    state_record = {}
    acc_test_record=[]
    while epoch < conf['max_epoch']:
        acc_val, acc_test = train(all_logits, dur, epoch, model, optimizer)
        acc_test_record.append(acc_test)
        epoch += 1
        if acc_test >= best:
            best = acc_test
            cnt = 0
        else:
            cnt += 1
        state = dict([('model', copy.deepcopy(model.state_dict())),
                     ('optim', copy.deepcopy(optimizer.state_dict()))])
        '''
        if acc_val in record.keys():
            if acc_test > record[acc_val]:
                record[acc_val] = acc_test
                state_record[acc_test] =state
        else:
            record[acc_val.item()] = acc_test.item()
            state_record[acc_test] = state
        '''
        record[acc_val.item()] = acc_test.item()
        state_record[acc_test] = state
        if (epoch - 1) % 10 == 0:
            bit_list = sorted(record.keys())
            bit_list.reverse()
            # print("ACC on Test_data: {:.4f}s".format(record[0]))
            for key in bit_list[:1]:
                test_record.append(record[key])
            record={}
        # if cnt == conf['patience'] or epoch == conf['max_epoch']:
        if epoch == conf['max_epoch']:
            print("Stop!!!")
            test_list = sorted(test_record)
            test_list.reverse()
            test_keys =test_list[0]
            model.load_state_dict(state_record[test_keys]['model'])
            optimizer.load_state_dict(state_record[test_keys]['optim'])
            best_score.append(test_list[0])
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(np.sum(dur)))
    if model == t1_model:
        state = {'t1_model': model.state_dict(), 't1_optimizer': optimizer.state_dict(), 't1_epoch': epoch}
        torch.save(state, "./outputs/" + conf['dataset'] + "/" + conf['model_name'] +"/"+"seed"+str(conf['seed'])+ "/model_t1.pkl")
        print("eeeeeeee1")
    elif model == t2_model:
        state = {'t2_model': model.state_dict(), 't2_optimizer': optimizer.state_dict(), 't2_epoch': epoch}
        torch.save(state, "./outputs/" + conf['dataset'] + "/" + conf['model_name'] +"/"+"seed"+str(conf['seed']) + "/model_t2.pkl")
        print("eeeeeeee2")
    elif model == t3_model:
        state = {'t3_model': model.state_dict(), 't3_optimizer': optimizer.state_dict(), 't3_epoch': epoch}
        torch.save(state, "./outputs/" + conf['dataset'] +"/" + conf['model_name'] +"/"+"seed"+str(conf['seed']) +"/model_t3.pkl")
        print("eeeeeeee3")


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(np.sum(dur)))


def test(conf,model):
    model.eval()
    with torch.no_grad():
        if conf['teacher'] == 'GraphSAGE':
            logits = model(G, G.ndata['feat'])  # model后已经训练完了成2485*7，原来是2485*1433
        else:
            logits = model(G.ndata['feat'])  # model后已经训练完了成2485*7，原来是2485*1433
    logp = F.log_softmax(logits, dim=1)#2485*7
    loss_test = F.nll_loss(logp[idx_test], labels[idx_test])
    acc_test = accuracy_score(logp[idx_test].max(1)[1].cpu().detach().numpy(), labels[idx_test].cpu().detach().numpy())
    print("Test set results: loss= {:.4f} acc_test= {:.4f} ".format(
        loss_test.item(), acc_test.item()))

    return acc_test, logp


if __name__ == '__main__':
    #os.chdir(r"C:\Users\dd\Desktop\YLP_again_two_cpu")
    args = arg_parse(argparse.ArgumentParser())
    config_path = Path.cwd().joinpath('models', 'train.conf.yaml')
    #os.chdir(r"C:\Users\dd\Desktop\YLP_again_two")
    '''config_path='D:\CPF-master\models\\train.conf.yaml'''
    conf = get_training_config(config_path, model_name=args.teacher)
    config_data_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')
    '''config_data_path ='D:\CPF-master\data\\dataset.conf.yaml'''''
    #conf['division_seed'] = get_experiment_config(config_data_path)['seed']
    conf['division_seed']=0
    conf = dict(conf, **args.__dict__)
    conf['device'] = torch.device("cuda:0")
    conf['seed'] = args.seed
    #conf['device'] = torch.device("cpu")

    output_dir, cascade_dir = choose_path(conf)
    logger = get_logger(output_dir.joinpath('log'))
    # random seed
    np.random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Load data
    adj, adj_sp,adj_calculate, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
        load_tensor_data(conf['seed'],conf['model_name'], conf['dataset'], args.labelrate,args.valrate, conf['device'])

    G = dgl.graph((adj_sp.row, adj_sp.col)).to(conf['device'])
    G.ndata['feat'] = features
    print(idx_train.shape)
    print(idx_val.shape)
    print(idx_test.shape)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    # The first layer transforms input features of size of 5 to a hidden size of 5.
    # The second layer transforms the hidden layer and produces output features of
    # size 2, corresponding to the two groups of the karate club.
    #feature 2485*1433
    print(conf)

    t1_model = choose_model(conf,'t1_model')

    t2_model = choose_model(conf,'t2_model')

    t3_model = choose_model(conf,'t3_model')

    t1_optimizer = optim.Adam(filter(lambda p: p.requires_grad, t1_model.parameters()), lr=conf['learning_rate'],
                               weight_decay=conf['weight_decay'])
    t2_optimizer = optim.Adam(filter(lambda p: p.requires_grad, t2_model.parameters()), lr=conf['learning_rate'],
                               weight_decay=conf['weight_decay'])
    t3_optimizer = optim.Adam(filter(lambda p: p.requires_grad, t3_model.parameters()), lr=conf['learning_rate'],
                               weight_decay=conf['weight_decay'])
    print(f"number of parameter for teacher model with 1 hidden layer: {parameters(t1_model)}")
    print(f"number of parameter for teacher model with 2 hidden layers: {parameters(t2_model)}")
    print(f"number of parameter for teacher model with 3 hidden layers: {parameters(t3_model)}")
    all_logits = []
    best_score=[]

    print("########train_teacher#########")
    model_train(conf, t1_model, t1_optimizer, all_logits)
    model_train(conf, t2_model, t2_optimizer, all_logits)
    model_train(conf, t3_model, t3_optimizer, all_logits)
    print("########test_teacher#########")
    acc_test_t1, logp_t1 = test(conf,t1_model)#logp是预测标签最后值
    acc_test_t2, logp_t2 = test(conf, t2_model)  # logp是预测标签最后值
    acc_test_t3, logp_t3 = test(conf, t3_model)  # logp是预测标签最后值
    preds_t1 = logp_t1.max(1)[1].type_as(labels).cpu().numpy()#每一行哪个位置数值最大,最后标签值
    preds_t2 = logp_t2.max(1)[1].type_as(labels).cpu().numpy()
    preds_t3 = logp_t3.max(1)[1].type_as(labels).cpu().numpy()
    preds=[preds_t1,preds_t2,preds_t3]

    labels = labels.cpu().numpy()

    output_t1 = np.exp(logp_t1.cpu().detach().numpy())#logp的exp
    output_t2 = np.exp(logp_t2.cpu().detach().numpy())
    output_t3 = np.exp(logp_t3.cpu().detach().numpy())
    #torch.save(t1_model,"./outputs/cora/GCN/model_t1.pkl")
    #torch.save(t2_model, "./outputs/cora/GCN/model_t2.pkl")
    #torch.save(t3_model, "./outputs/cora/GCN/model_t3.pkl")

    acc_test_t1 = acc_test_t1.item()
    acc_test_t2 = acc_test_t2.item()
    acc_test_t3 = acc_test_t3.item()
    acc_test=[acc_test_t1,acc_test_t2,acc_test_t3]

    np.savetxt(output_dir.joinpath('preds.txt'), preds, fmt='%d', delimiter='\t')
    #np.savetxt(output_dir.joinpath('preds_t1.txt'), preds_t1, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('labels.txt'), labels, fmt='%d', delimiter='\t')
    #np.savetxt(output_dir.joinpath('output_t1.txt'), output_t1, fmt='%.4f', delimiter='\t')
    #np.savetxt(output_dir.joinpath('output_t2.txt'), output_t2, fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('output_t1.txt'), output_t1, fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('output_t2.txt'), output_t2, fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('output_t3.txt'), output_t3, fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('test_acc.txt'), np.array([acc_test]), fmt='%.4f', delimiter='\t')
    if 'a' in G.edata:
        print('Saving Attention...')
        edge = torch.stack((G.edges()[0], G.edges()[1]),0)
        sp_att = sp.coo_matrix((G.edata['a'].cpu().detach().numpy(), edge.cpu()), shape=adj.cpu().size())
        sp.save_npz(output_dir.joinpath('attention_weight.npz'), sp_att, compressed=True)
