#
import os
import scanpy as sc
import pandas as pd
from pathlib import Path
from scanpy.readwrite import read_visium
from scanpy._utils  import check_presence_download


def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def adata_preprocess(i_adata, min_cells=3, pca_n_comps=200):
    print('===== Preprocessing Data ')
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    ####
    adata_X = sc.pp.log1p(adata_X)
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X


def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_Adj=None):
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()

    if load_images is False:
        if file_Adj is None:
            file_Adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        adata_h5.obs.drop(columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)

    print('adata_h5: (' + str(adata_h5.shape[0]) + ', ' + str(adata_h5.shape[1]) + ')')
    return adata_h5


# from scanpy
def _download_visium_dataset(
    sample_id: str,
    spaceranger_version: str,
    base_dir='./data/',
):
    import tarfile

    url_prefix = f'https://cf.10xgenomics.com/samples/spatial-exp/{spaceranger_version}/{sample_id}/'

    sample_dir = Path(mk_dir(os.path.join(base_dir, sample_id)))

    # Download spatial data
    tar_filename = f"{sample_id}_spatial.tar.gz"
    tar_pth = Path(os.path.join(sample_dir, tar_filename))
    check_presence_download(filename=tar_pth, backup_url=url_prefix + tar_filename)
    with tarfile.open(tar_pth) as f:
        for el in f:
            if not (sample_dir / el.name).exists():
                f.extract(el, sample_dir)

    # Download counts
    check_presence_download(
        filename=sample_dir / "filtered_feature_bc_matrix.h5",
        backup_url=url_prefix + f"{sample_id}_filtered_feature_bc_matrix.h5",
    )


def load_visium_sge(sample_id='V1_Breast_Cancer_Block_A_Section_1', save_path='./data/'):
    if "V1_" in sample_id:
        spaceranger_version = "1.1.0"
    else:
        spaceranger_version = "1.2.0"
    _download_visium_dataset(sample_id, spaceranger_version, base_dir=save_path)
    adata = read_visium(os.path.join(save_path, sample_id))

    print('adata: (' + str(adata.shape[0]) + ', ' + str(adata.shape[1]) + ')')
    return adata


# !/usr/bin/env python
'''
# Author: ChangXu
# Create Time : 2021.4
# File Name :utils_func.py

'''

import os
import sys
import numpy as np
import anndata
import scanpy as sc
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from scipy.sparse import issparse, csr_matrix
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
# import stlearn
from src._compat import Literal
import scanpy
import scipy
import matplotlib.pyplot as plt

_QUALITY = Literal["fulres", "hires", "lowres"]
_background = ["black", "white"]


def read_10X_Visium(path,
                    genome=None,
                    count_file='filtered_feature_bc_matrix.h5',
                    library_id=None,
                    load_images=True,
                    quality='hires',
                    image_path=None):
    adata = sc.read_visium(path,
                           genome=genome,
                           count_file=count_file,
                           library_id=library_id,
                           load_images=load_images, )
    adata.var_names_make_unique()

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata


def read_SlideSeq(path,
                  library_id=None,
                  scale=None,
                  quality="hires",
                  spot_diameter_fullres=50,
                  background_color="white", ):
    count = pd.read_csv(os.path.join(path, "count_matrix.count"))
    meta = pd.read_csv(os.path.join(path, "spatial.idx"))

    adata = AnnData(count.iloc[:, 1:].set_index("gene").T)

    adata.var["ENSEMBL"] = count["ENSEMBL"].values

    adata.obs["index"] = meta["index"].values

    if scale == None:
        max_coor = np.max(meta[["x", "y"]].values)
        scale = 2000 / max_coor

    adata.obs["imagecol"] = meta["x"].values * scale
    adata.obs["imagerow"] = meta["y"].values * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"] = scale

    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres
    adata.obsm["spatial"] = meta[["x", "y"]].values

    return adata


def read_merfish(path,
                 library_id=None,
                 scale=None,
                 quality="hires",
                 spot_diameter_fullres=50,
                 background_color="white", ):
    counts = sc.read_csv(os.path.join(path, 'counts.csv')).transpose()
    locations = pd.read_excel(os.path.join(path, 'spatial.xlsx'), index_col=0)
    if locations.min().min() < 0:
        locations = locations + np.abs(locations.min().min()) + 100
    adata = counts[locations.index, :]
    adata.obsm["spatial"] = locations.to_numpy()

    if scale == None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "MERSEQ"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata


def read_seqfish(path,
                 library_id=None,
                 scale=1.0,
                 quality="hires",
                 field=0,
                 spot_diameter_fullres=50,
                 background_color="white", ):
    count = pd.read_table(os.path.join(path, 'counts.matrix'), header=None)
    spatial = pd.read_table(os.path.join(path, 'spatial.csv'), index_col=False)

    count = count.T
    count.columns = count.iloc[0]
    count = count.drop(count.index[0]).reset_index(drop=True)
    count = count[count["Field_of_View"] == field].drop(count.columns[[0, 1]], axis=1)
    spatial = spatial[spatial["Field_of_View"] == field]

    # cells = set(count[''])
    # obs = pd.DataFrame(index=cells)
    adata = AnnData(count)

    if scale == None:
        max_coor = np.max(spatial[["X", "Y"]])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = spatial["X"].values * scale
    adata.obs["imagerow"] = spatial["Y"].values * scale

    adata.obsm["spatial"] = spatial[["X", "Y"]].values

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "SeqFish"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata


def read_stereoSeq(path,
                   bin_size=100,
                   is_sparse=True,
                   library_id=None,
                   scale=None,
                   quality="hires",
                   spot_diameter_fullres=1,
                   background_color="white", ):
    from scipy import sparse
    count = pd.read_csv(os.path.join(path, "count.txt"), sep='\t', comment='#', header=0)
    count.dropna(inplace=True)
    if "MIDCounts" in count.columns:
        count.rename(columns={"MIDCounts": "UMICount"}, inplace=True)
    count['x1'] = (count['x'] / bin_size).astype(np.int32)
    count['y1'] = (count['y'] / bin_size).astype(np.int32)
    count['pos'] = count['x1'].astype(str) + "-" + count['y1'].astype(str)
    bin_data = count.groupby(['pos', 'geneID'])['UMICount'].sum()
    cells = set(x[0] for x in bin_data.index)
    genes = set(x[1] for x in bin_data.index)
    cellsdic = dict(zip(cells, range(0, len(cells))))
    genesdic = dict(zip(genes, range(0, len(genes))))
    rows = [cellsdic[x[0]] for x in bin_data.index]
    cols = [genesdic[x[1]] for x in bin_data.index]
    exp_matrix = sparse.csr_matrix((bin_data.values, (rows, cols))) if is_sparse else \
        sparse.csr_matrix((bin_data.values, (rows, cols))).toarray()
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)
    adata = AnnData(X=exp_matrix, obs=obs, var=var)
    pos = np.array(list(adata.obs.index.str.split('-', expand=True)), dtype=np.int)
    adata.obsm['spatial'] = pos

    if scale == None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 20 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "StereoSeq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata


# def ReadOldST(
#     count_matrix_file: Union[str, Path] = None,
#     spatial_file: Union[str, Path] = None,
#     image_file: Union[str, Path] = None,
#     library_id: str = "OldST",
#     scale: float = 1.0,
#     quality: str = "hires",
#     spot_diameter_fullres: float = 50,
# ) -> AnnData:

#     """\
#     Read Old Spatial Transcriptomics data
#     Parameters
#     ----------
#     count_matrix_file
#         Path to count matrix file.
#     spatial_file
#         Path to spatial location file.
#     image_file
#         Path to the tissue image file
#     library_id
#         Identifier for the visium library. Can be modified when concatenating multiple adata objects.
#     scale
#         Set scale factor.
#     quality
#         Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow']
#     spot_diameter_fullres
#         Diameter of spot in full resolution
#     Returns
#     -------
#     AnnData
#     """

#     adata = scanpy.read_text(count_matrix_file)
#     adata = stlearn.add.parsing(adata, coordinates_file=spatial_file)
#     stlearn.add.image(
#         adata,
#         library_id=library_id,
#         quality=quality,
#         imgpath=image_file,
#         scale=scale,
#         spot_diameter_fullres=spot_diameter_fullres,
#     )

#     return adata

def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0:num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred





