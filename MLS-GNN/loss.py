import torch
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
import torch.nn.functional as F


def graph_intel_loss(models, t1_model,t2_model,middle_feats_s, subgraph, feats, epoch, conf,idx_train,beta_1,beta_2,adj):
    t_model = t1_model
    with torch.no_grad():
        t_model.g = subgraph
        for layer in t_model.layers:
            layer.g = subgraph
        _, middle_feats_t1 = t_model(feats.float(), middle=True)
        middle_feats_t1s_0 = middle_feats_t1[0]
    t_model = t2_model
    with torch.no_grad():
        t_model.g = subgraph
        for layer in t_model.layers:
            layer.g = subgraph
        _, middle_feats_t2 = t_model(feats.float(), middle=True)
        middle_feats_t2s_0 = middle_feats_t2[0]
        middle_feats_t2s_1 = middle_feats_t2[1]

    middle_feats_ss = middle_feats_s[0]
    dist_s = inter_distance(subgraph, middle_feats_ss,adj)
    dist_t = beta_1 * inter_distance(subgraph, middle_feats_t1s_0,adj) + beta_2 * (inter_distance(subgraph, middle_feats_t2s_0,adj)+inter_distance(subgraph, middle_feats_t2s_1,adj))#111111*1
    return KLDiv(subgraph.to(torch.device("cuda:0")), dist_s, dist_t)



def inter_distance(graph,feats,adj):
    graph = graph.local_var().to(torch.device("cuda:0"))
    feats = feats.view(-1, 1, feats.shape[1])
    graph.ndata.update({'ftl': feats, 'ftr': feats})
    graph.apply_edges(fn.u_sub_v('ftl', 'ftr', 'diff'))
    e = graph.edata.pop('diff')
    row_sum = adj.sum(1)
    degree = row_sum.reshape(-1,1)
    degrees = torch.zeros(graph.num_nodes()).to("cuda:0")
    graph.ndata.update({'degreel':degree,'degreer':degrees})
    graph.apply_edges(fn.u_sub_v('degreel', 'degreer', 'edge'))
    edges_degree = graph.edata.pop('edge')

    e  = torch.exp((-1/100) * torch.sum(torch.abs(e), dim=-1))/edges_degree
    e = edge_softmax(graph, e)
    return e




def KLDiv(graph, edgex, edgey):
    with graph.local_scope():
        nnode = graph.number_of_nodes()
        #graph.ndata.update({'kldiv': torch.ones(nnode,1).to(edgex.device)})
        graph.ndata.update({'kldiv': torch.ones(nnode, 1).to("cuda:0")})#2485*1,全是1
        diff = edgey*(torch.log(edgey)-torch.log(edgex))
        graph.edata.update({'diff':diff})
        graph.update_all(fn.u_mul_e('kldiv', 'diff', 'm'), fn.sum('m', 'kldiv'))
        #return torch.mean(torch.flatten(graph.ndata['kldiv'][idx_train]))
        return torch.mean(torch.flatten(graph.ndata['kldiv']))



