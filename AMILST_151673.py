import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from src.graph_func import graph_construction
from src.utils_func import mk_dir, load_ST_file, adata_preprocess
import anndata
from src.AMILST_train import AMILST_Train
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn import metrics

import os




warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
device = 'cpu'
print('===== Using device: ' + device)



# ################ Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph ')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--cell_feat_dim', type=int, default=200, help='Dim of PCA')
parser.add_argument('--X_dim', type=int, default=200, help='Dim of PCA')
parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.0.2')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
parser.add_argument('--lambda1', type=float, default=0.7, help='')
parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
# ______________ Eval clustering Setting ______________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.') 

params = parser.parse_args()
params.device = device

# ################ Path setting
data_root = './AMILST/data/DLPFC'
# all DLPFC folder list
proj_list = ['151673']


# set saving result path
save_root = './AMILST/output/DLPFC/'


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]
        
        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res


ARI_list = []
nmi_list = []
#acc_list = []
for proj_idx in range(len(proj_list)):
    data_name = proj_list[proj_idx]
    print('===== Project ' + str(proj_idx+1) + ' : ' + data_name)
    file_fold = f'{data_root}/{data_name}'

    data_path = "./AMILST/data/DLPFC"
    data_name = '151673'

    adata_h4 = load_ST_file(file_fold=file_fold)
    adata_h4.var_names_make_unique()

    adata_X = adata_preprocess(adata_h4, min_cells=3, pca_n_comps=params.cell_feat_dim)  
    graph_dict, mini_batch = graph_construction(adata_h4.obsm['spatial'], adata_h4.shape[0], params)

    params.save_path = mk_dir(f'{save_root}/{data_name}/AMILST')
    params.cell_num = adata_h4.shape[0]
    print('==== Graph Construction Finished')
    # ################## Model training
    amil_net = AMILST_Train(adata_X, graph_dict, mini_batch, params)

    amil_net.train_with_dec()
    amil_feat, _, _, _ = amil_net.process()

    adata_h4.obsm["amil_feat"] = amil_feat
    adata_amil = anndata.AnnData(amil_feat)
    adata_amil.uns['spatial'] = adata_h4.uns['spatial']
    adata_amil.obsm['spatial'] = adata_h4.obsm['spatial']

    sc.pp.neighbors(adata_amil, n_neighbors=params.eval_graph_n)
    sc.tl.umap(adata_amil)


    if data_name in ['151669', '151670', '151671', '151672']:
        n_clusters = 5
    else:
        n_clusters = 7
    eval_resolution = res_search_fixed_clus(adata_amil, n_clusters)

    sc.tl.leiden(adata_amil, key_added="AMILST_leiden", resolution=eval_resolution)

    sc.pl.spatial(adata_amil, img_key="hires", color=['AMILST_leiden'], show=False)
    plt.savefig(f'{params.save_path}/AMILST_leiden_plot.jpg', bbox_inches='tight', dpi=150)

    ###################################################################
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata_amil, color=['AMILST_leiden', 'AMILST_leiden'], title=['MMI (ARI=0.57)', "Ground Truth"])
    # plt.savefig(f'{params.save_path}/umap_plot.jpg', bbox_inches='tight', dpi=150)
    plt.savefig(f'{params.save_path}/umap_plot.jpg', bbox_inches='tight', dpi=150)
    ####################################################################
    sc.tl.paga(adata_amil, groups='AMILST_leiden')
    plt.rcParams["figure.figsize"] = (4,3)
    sc.pl.paga_compare(adata_amil, legend_fontsize=10, frameon=False, size=20,
                       title='151673_MMI', legend_fontoutline=2, show=False)
    plt.savefig(f'{params.save_path}/paga_plot.jpg', bbox_inches='tight', dpi=150)

    df_meta = pd.read_csv(f'{data_root}/{data_name}/metadata.tsv', sep='\t')
    df_meta['AMILST'] = adata_amil.obs['AMILST_leiden'].tolist()
    df_meta.to_csv(f'{params.save_path}/metadata.tsv', sep='\t', index=False)

    # #################### evaluation
    # ---------- Load manually annotation ---------------
    df_meta = df_meta[~pd.isnull(df_meta['layer_guess'])]
    ARI = metrics.adjusted_rand_score(df_meta['layer_guess'], df_meta['AMILST']) #true labels pred labels
    print('===== Project: {} ARI score: {:.3f}'.format(data_name, ARI))
    ARI_list.append(ARI)
    nmi = metrics.normalized_mutual_info_score(df_meta['layer_guess'], df_meta['AMILST'])
    print('===== Project: {} nmi score: {:.3f}'.format(data_name, nmi))
    nmi_list.append(nmi)
print('===== Project: AVG ARI score: {:.3f}'.format(np.mean(ARI_list)))
print('===== Project: AVG nmi score: {:.3f}'.format(np.mean(nmi_list)))
