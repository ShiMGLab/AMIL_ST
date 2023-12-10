# [**AMIL-ST: a method for identifying spatial domains in spatial transcriptomics via adversarial mutual information learning**]
## Overview

![F1](/Users/lixuefeng/Desktop/MMI/结果 - 副本/107/108/F1.png)
A schematic outline of the algorithmic workflow. AMIL-ST capitalizes on spatial transcriptomics data to construct a specialized graph, in which gene expressions serve as node attributes and edges are delineated according to spatial coordinates. The initial stage involves decomposing the gene expression matrix through Principal Component Analysis (PCA). This is followed by the formulation of a graph on the basis of K-Nearest Neighbor (KNN) distances, computed from the spatial coordinates of each individual spot. The method incorporates two key neural network modules: a denoising autoencoder and a variational graph autoencoder. The overarching objective of these modules is the extraction of a latent feature, ![img](file:////Users/lixuefeng/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image002.png), which undergoes further refinement via an adversarial mutual information mechanism. This mechanism is specifically designed to optimize the mutual information between the input and output spaces of the VGAE, thereby enhancing the robustness and precision of the feature representation.

## Dependencies
- Python=3.9.16
- torch=1.10.1
- torchvision=0.11.2
- torch-geometric=2.1.0
- torch-scatter=2.0.9
- torch-sparse=0.6.13
- scikit-learn=1.2.1
- numpy=1.23.5
- scanpy=1.9.2
- seaborn=0.12.2
- scipy=1.10.1
- networkx=3.0
- pandas=1.5.3
- tqdm=4.64.1
- matplotlib=3.7.0
- seaborn=0.12.2
- anndata=0.8.0
- leidenalg=0.9.1


## Usage
Run `AMILST_151673.py for a clustering demo of slice 151673 of [spatialLIBD](http://spatial.libd.org/) dataset.

Demo uses [spatialLIBD](http://spatial.libd.org/) dataset. We have organized the file structure and put the data in at https://github.com/ShiMGLab/AMIL-ST. Please download and put it into `data` folder.
If you want to experiment with other data, you can arrange the file structure the same as it.

<!---

### Note
Due to the CUDA non-deterministic characteristic of the sparse tensor operations in [Pytorch Geometrics](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html), performance may vary with different runs or in different hardware devices. 
For optimal performance, it is advisable to execute the code multiple times or make minor adjustments to the parameters. To guarantee reproducibility, we intend to make the trained weights employed in this paper publicly available. Additionally, we have plans to enhance the code in future updates to mitigate this issue.
-->


## Acknowledgement
The code is partly adapted from [SEDR](https://github.com/JinmiaoChenLab/SEDR),  [SpaGCN](https://github.com/jianhuupenn/SpaGCN).
