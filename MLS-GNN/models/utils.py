import yaml
import torch
import scipy.sparse as sp
import numpy as np

def get_training_config(config_path, model_name):
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    specific_config = dict(full_config['global'], **full_config[model_name])
    specific_config['model_name'] = model_name
    return specific_config


def check_device(conf):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(conf['device'])
    if conf['model_name'] in ['DeepWalk', 'GraphSAGE']:
        is_cuda = False
    else:
        is_cuda = not conf['no_cuda'] and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.manual_seed(conf['seed'])
        torch.cuda.manual_seed_all(conf['seed'])  # if you are using multi-GPU.
    device = torch.device("cuda:" + str(conf['device'])
                          ) if is_cuda else torch.device("cpu")
    return device


def aug_normalized_adjacency(adj):
   #adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))#行相加的和
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.#np.isinf就是看是否是正无穷或负无穷
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)#构建对角矩阵
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_row(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()