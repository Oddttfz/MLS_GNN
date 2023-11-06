import numpy as np
import scipy.sparse as sp
import torch
import os
import time
from pathlib import Path
from data.get_dataset import load_dataset_and_split
from models.utils import aug_normalized_adjacency,sparse_mx_to_torch_sparse_tensor
import pickle as pkl
import networkx as nx
import random
from torch.utils.data import DataLoader
from dgl.data.ppi import LegacyPPIDataset as PPIDataset
import dgl
def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("./data/datas/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "./data/datas/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # print(nx.from_dict_of_lists(graph).nodes)
    graph = nx.from_dict_of_lists(graph)
    edge_list = list(graph.edges)
    remove_list = random.sample(edge_list, len(edge_list) * 0 // 10)
    for edge in remove_list:
        graph.remove_edge(edge[0], edge[1])

    adj = nx.adjacency_matrix(graph)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, np.argmax(labels, 1), idx_train, idx_val, idx_test

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)





def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return mx


def normalize_adj(adj):
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj


def normalize_features(features):
    features = normalize(features)
    return features


def initialize_label(idx_train, labels_one_hot):
    labels_init = torch.ones_like(labels_one_hot) / len(labels_one_hot[0])
    labels_init[idx_train] = labels_one_hot[idx_train]
    return labels_init


def split_double_test(dataset, idx_test):
    test_num = len(idx_test)
    idx_test1 = idx_test[:int(test_num/2)]
    idx_test2 = idx_test[int(test_num/2):]
    return idx_test1, idx_test2


def preprocess_adj(model_name, adj):
    return normalize_adj(adj)


def preprocess_features(model_name, features):
    return features



def load_tensor_data(seed,model_name, dataset, labelrate, valrate,device):
    if dataset in ['composite', 'composite2', 'composite3']:
        adj, features, labels_one_hot, idx_train, idx_val, idx_test = load_composite_data(dataset)
    else:
        # config_file = os.path.abspath('data/dataset.conf.yaml')
        adj, features, labels_one_hot, idx_train, idx_val, idx_test = load_dataset_and_split(seed,labelrate,valrate,dataset)

    #adj_identity = sp.eye(adj.shape[0])
    #adj_1st = (adj + adj_identity).toarray() # adj+单位矩阵
    #adj_label = torch.FloatTensor(adj_1st)  # =邻接矩阵+单位矩阵
    #neg_num = pos_num = adj_label.sum().long()  # 临近矩阵的边总数
    #adj_calculate = aug_normalized_adjacency(adj_1st)
    adj = preprocess_adj(model_name, adj)
    adj_calculate = aug_normalized_adjacency(adj)
    adj_calculate = sparse_mx_to_torch_sparse_tensor(adj_calculate).float()  # D逆*A，先计算好
    features = preprocess_features(model_name, features)
    adj_sp = adj.tocoo()
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = labels_one_hot.argmax(axis=1)

    # labels, labels_init = initialize_label(idx_train, labels_one_hot)
    labels = torch.LongTensor(labels)
    labels_one_hot = torch.FloatTensor(labels_one_hot)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # adj_sp = sp.coo_matrix(adj.numpy(), dtype=float)
    print('Device: ', device)
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    adj_calculate=adj_calculate.to(device)
    labels_one_hot = labels_one_hot.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    return adj, adj_sp,adj_calculate, features, labels, labels_one_hot, idx_train, idx_val, idx_test


def load_composite_data(dataset):
    base_dir = Path.cwd().joinpath('data', dataset)
    adj = np.loadtxt(str(base_dir.joinpath('adj')))
    features = np.loadtxt(str(base_dir.joinpath('features')))
    labels_one_hot = np.loadtxt(str(base_dir.joinpath('labels')))
    idx_train = np.loadtxt(str(base_dir.joinpath('idx_train')))
    idx_val = np.loadtxt(str(base_dir.joinpath('idx_val')))
    idx_test = np.loadtxt(str(base_dir.joinpath('idx_test')))
    adj = sp.csr_matrix(adj)
    # adj = normalize_adj(adj)
    features = sp.csr_matrix(features)
    # features = normalize_features(features)
    # labels, labels_init = initialize_label(idx_train, labels_one_hot)

    return adj, features, labels_one_hot, idx_train, idx_val, idx_test




def matrix_pow(m1, n, m2):
    t = time.time()
    m1 = sp.csr_matrix(m1)
    m2 = sp.csr_matrix(m2)
    ans = m1.dot(m2)
    for i in range(n-2):
        ans = m1.dot(ans)
    ans = torch.FloatTensor(ans.todense())
    print(time.time() - t)
    return ans



def row_normalize(data):
    return (data.t() / torch.sum(data.t(), dim=0)).t()




def check_writable(dir, overwrite=True):
    import shutil
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif overwrite:
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        pass





def choose_path(conf):
    if 'assistant' not in conf.keys():
        teacher_str = conf['teacher']
    elif conf['assistant'] == 0:
        teacher_str = 'nasty_' + conf['teacher']
    elif conf['assistant'] == 1:
        teacher_str = 'reborn_' + conf['teacher']
    else:
        raise ValueError(r'No such assistant')
    if conf['student'] == 'PLP' and conf['ptype'] == 0:
        output_dir = Path.cwd().joinpath('outputs', conf['dataset'], teacher_str + '_' + conf['student'],
                                         'cascade_random_' + str(conf['division_seed']) + '_' + str(conf['labelrate']) + '_ind')
    elif conf['student'] == 'PLP' and conf['ptype'] == 1:
        output_dir = Path.cwd().joinpath('outputs', conf['dataset'], teacher_str + '_' + conf['student'],
                                         'cascade_random_' + str(conf['division_seed']) + '_' + str(conf['labelrate']) + '_tra')
    else:
        output_dir = Path.cwd().joinpath('outputs', conf['dataset'], teacher_str + '_' + conf['student'],
                                         'cascade_random_' + str(conf['division_seed']))
    check_writable(output_dir, overwrite=False)
    cascade_dir = Path.cwd().joinpath('outputs', conf['dataset'], teacher_str,
                                      'cascade_random_' + str(conf['division_seed']) + '_' + str(conf['labelrate']), 'cascade')
    # check_readable(cascade_dir)
    return output_dir, cascade_dir

def parameters(model):
    num_params = 0
    for params in model.parameters():
        cur = 1
        for size in params.data.shape:
            cur *= size
        num_params += cur
    return num_params

def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels



def get_data_loader(conf):
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=conf['batch_size'], collate_fn=collate, num_workers=0, shuffle=True)
    #collate_fn，自定义输入函数,graph, feats, labels
    fixed_train_dataloader = DataLoader(train_dataset, batch_size=conf['batch_size'], collate_fn=collate, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=conf['batch_size'], collate_fn=collate, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=conf['batch_size'], collate_fn=collate, num_workers=0)

    train_data_label=np.load('./ppi/train_labels.npy')
    n_classes = train_data_label.shape[1]
    train_data_num_feats=np.load('./ppi/train_feats.npy')
    num_feats = train_data_num_feats.shape[1]
    g = train_dataset.graph
    data_info = {}
    data_info['n_classes'] = n_classes
    data_info['num_feats'] = num_feats
    data_info['g'] = g
    return (train_dataloader, valid_dataloader, test_dataloader, fixed_train_dataloader), data_info


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device),False)
    print(f"Load model from {path}")


def save_checkpoint(model, path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), path)
    print(f"save model to {path}")