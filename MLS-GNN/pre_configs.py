
predefined_configs = {
        'GCN': {
            'cora': {
                'emb_dim': 128,
                'lr': 0.003,#0.003,0.0025
                'wd': 0.01,#0.01
                'drop_out': 0.9,
                'temperature':5
            },
            'cora_full': {
                'num_layers': 8,
                'emb_dim': 128,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01,
                'temperature': 1
            },
            'citeseer': {
                'emb_dim': 128,
                'drop_out':0.9,#0.8/0.9
                'beta': 0,
                'lr': 0.003,#0.003
                'wd': 0.037,#0.037
                'temperature': 3#3
            },
            'pubmed': {
                #'num_layers': 6,#layers=6,feat0.7,att0.4##lr=0.01,wd=0.01
                #'emb_dim': 8,
                'emb_dim': 128,
                'lr': 0.01,#0.01
                'wd': 0.03,#0.01
                'drop_out': 0.6,
                'temperature': 3#4
            },
            'amazon_electronics_computers': {
                'emb_dim': 128,
                'lr': 0.005,
                'wd': 0.01,
                'drop_out': 0.8,
                'temperature':2
            },
            'amazon_electronics_photo': {
                #'num_layers': 6,
                'emb_dim': 128,
                'lr': 0.006,#0.003#0.005,6#####0.006
                'wd': 0.01,#0.025.0.01
                'drop_out': 0.9,#0.8
                'temperature': 2
            },
            'DBLP': {
                'emb_dim': 128,
                'lr': 0.005,  # 0.003#0.005,6#####seed=6 ,93.64
                'wd': 0.01,  # 0.025.0.01
                'drop_out': 0.8,  # 0.8
                'temperature': 1
            },
            'ms_academic_cs': {
                'emb_dim': 128,
                'lr': 0.001,
                'wd': 0.01,
                'drop_out': 0.8,  # 0.8
                'temperature': 1
            },
            'ms_academic_phy': {
                'num_layers': 8,
                'emb_dim': 128,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01,
                'temperature': 1
            }
        },
        'GAT': {
            'cora': {
                'num_layers': 5,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 5,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 6,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.2,
                'attn_drop': 0.5,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_photo': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.5,
                'beta': 1,
                'lr': 0.01,
                'wd': 0.01
            },
        },
        'APPNP': {
            'cora': {
                'num_layers': 5,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 7,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 8,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.2,
                'beta': 5,
                'lr': 0.005,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 7,
                'emb_dim': 64,
                'feat_drop': 0.2,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.01,
                'wd': 0.0005
            },
            'amazon_electronics_photo': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.8,
                'beta': 0,
                'lr': 0.01,
                'wd': 0.01
            },
        },
        'GraphSAGE': {
            'cora': {
                'num_layers': 9,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 7,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 9,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 8,
                'emb_dim': 64,
                'feat_drop': 0.2,
                'attn_drop': 0.5,
                'beta': 0,
                'lr': 0.01,
                'wd': 0.0005
            },
            'amazon_electronics_photo': {
                'num_layers': 9,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.8,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
        'SGC': {
            'cora': {
                'num_layers': 5,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 10,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 9,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 8,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.0005
            },
            'amazon_electronics_photo': {
                'num_layers': 6,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
        'GCNII': {
            'cora': {
                'num_layers': 7,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 8,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 9,
                'emb_dim': 8,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_computers': {
                'num_layers': 7,
                'emb_dim': 64,
                'feat_drop': 0.5,
                'attn_drop': 0.5,
                'lr': 0.001,
                'wd': 0.01
            },
            'amazon_electronics_photo': {
                'num_layers': 10,
                'emb_dim': 64,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'lr': 0.001,
                'wd': 0.01
            },
        },
        'GLP': {
            'cora': {
                'num_layers': 7,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'citeseer': {
                'num_layers': 5,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'pubmed': {
                'num_layers': 5,
                'emb_dim': 16,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        }
    }
