# Experiments on using Kolmogorov-Arnold Networks (KAN) on Graph Learning

This repository contains some quick experimental results for comparing the performance of MLP, GN (GCN), KAN, and KAN+GNN on several benchmark datasets on graph learning (specifically, node classfication). 

## TL;DR (for now)
- **Using KANs or KAN + GNNs usually introduces a lot of model parameters.** This makes really skeptical to use KANs or KAN+GNNs compared to MLPs or GNNs. **(Perhaps we need a more effective way to merge KANs with GNNs)**
- Make the model (especially the KAN part) as light as possible.
- KAN+GNN generally performs great on homophilic datasets, but really suffers on heterophilic datasets (even worse than GCNs).
- KANs shines more on heterophilic datasets.
- Learning rate is the most important hyperparameter for KANs and KAN+GNNs.


## KAN and KAN+GNN (with reference to the original repo)
To build KAN and KAN+GNN, I have used the implementation of [Efficient-KAN](https://github.com/Blealtan/efficient-kan) for all KAN and KAN+GNN experiments. For KAN+GNN, I have combined the Efficient-KAN with [GraphKAN](https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks), which defines each KAN+GNN layer with (KAN $\rightarrow$ `torch.sparse.spmm` with the adjacency matrix). The detailed settings are all set as default unless mentinoed explicitly. The utility functions including data splits are also from [GraphKAN](https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks). (I do not claim any ownership of the Efficient-KAN and GraphKAN code.)

## Datasets
The following datasets are used in the experiments:
- `Cora`
- `Citeseer`
- `Pubmed`
- `Cornell`
- `Texas`
- `Wisconsin`

Note that `Cora`, `Citeseer`, and `Pubmed` are homophilic, while `Cornell`, `Texas`, and `Wisconsin` are heterophilic datasets.

## Hyperparameter tuning
The following hyperparameters are tuned for each model. For all cases, the maximum number of epochs is set to 1000 except for GNNs. For KAN and KAN+GNN, I have also considered the option of projecting the input features to the hidden dimension as the first step
### MLP
- Hidden dim: [16, 32, 64]
- Num. layers: [1, 2, 3]
- Learning rate: [0.01, 0.001, 0.0001]
### KAN
- Hidden dim: [16, 32, 64]
- Num. layers: [1, 2]
- Project with MLP to hidden dim as the first step (`Proj`): [True, False]
- Learning rate: [0.1, 0.01, 0.001, 0.0001]
### GNN
- Architecture: GCN
- Hidden dim: [16, 32, 64]
- Num. layers: [1, 2, 3]
- Learning rate: [0.1, 0.01, 0.001, 0.0001]
### KAN+GNN
- Hidden dim: [16, 32, 64]
- Num. layers for KAN in each layer: [1, 2]
- Num. layers for message passing (`spmm`) in each layer: [1, 2, 3]
- Project with MLP to hidden dim as the first step (`Proj`): [True, False]
- Learning rate: [0.1, 0.01, 0.001, 0.0001]

## Result 1: Best performers

Results after hyperparameter tuning for different datasets.

- KAN+GNN generally performs great on homophilic datasets, but really suffers on heterophilic datasets (even worse than GCNs).
- KANs shines more on heterophilic datasets.
- Using KANs or KAN + GNNs usually introduces a lot of model parameters. This makes really skeptical to use KANs or KAN+GNNs compared to MLPs or GNNs.

### Cora
| Model | Validation accuracy | Test accuracy | Number of parameters | Best epoch | Hidden dim | Num. layers | Learning rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.712177 | 0.737274 | 10,038 | 2 | 16 | 1 | 0.1 |
| KAN | 0.804428 | 0.760263 | 921,600 (`Proj`=`False`) | 84  | 64 | 2 | 0.001 |
| GCN | 0.889299 | 0.866995 | 95,936 | 18 | 64 | 2 | 0.1 |
| KAN+GNN | **0.907749** | **0.875205** | 458,560 (`Proj`=`False`) | 105 | 32 | 1 (KAN) / 1 (spmm) | 0.1 |

### Citeseer
| Model | Validation accuracy | Test accuracy | Number of parameters | Best epoch | Hidden dim | Num. layers | Learning rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.760902 | 0.723056 | 22,224 | 3 | 16 | 1 | 0.1 |
| KAN | 0.801504 | 0.757162 | 593,440 (`Proj`=`False`) | 65 | 16 | 2 | 0.01 |
| GCN | **0.831579** | **0.815825** | 119,584 | 38 | 32 | 2 | 0.01 |
| KAN+GNN | **0.831579** | 0.809004 | 458,560 (`Proj`=`False`) | 104 | 64 | 1 (KAN) / 1 (spmm) | 0.1 |

### Pubmed
| Model | Validation accuracy | Test accuracy | Number of parameters | Best epoch | Hidden dim | Num. layers | Learning rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.890439 | 0.885932 | 36,675 | 80 | 64 | 3 | 0.001 |
| KAN | 0.884098 | 0.881115 | 80,480 (`Proj`=`False`) | 319 | 16 | 2 | 0.01 |
| GCN | 0.887649 | 0.864639 | 8,560 | 191 | 16 | 3 | 0.1 |
| KAN+GNN | **0.906416** | **0.905703** | 80,480 (`Proj`=`False`) | 330 | 16 | 1 (KAN) / 2 (spmm) | 0.01 |


### Cornell
| Model | Validation accuracy | Test accuracy | Number of parameters | Best epoch | Hidden dim | Num. layers | Learning rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.918919 | **0.914894** | 27,381 | 37 | 16 | 2 | 0.001 |
| KAN | **0.972973** | 0.829787 | 1,093,120 (`Proj`=`False`) | 46 | 64 | 2 | 0.001 |
| GCN | 0.810811 | 0.723404 | 27,536 | 5 | 16 | 2 | 0.1 |
| KAN+GNN | 0.891892 | 0.617021 | 275,840 (`Proj`=`False`) | 78 | 16 | 1 (KAN) / 3 (spmm) | 0.001 |



### Wisconsin
| Model | Validation accuracy | Test accuracy | Number of parameters | Best epoch | Hidden dim | Num. layers | Learning rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | **0.98** | **0.9125** | 109,509 | 4 | 64 | 2 | 0.1 |
| KAN | **0.98** | **0.9125** | 546,560 (`Proj`=`False`) | 39 | 32 | 2 | 0.01 |
| GCN | 0.84 | 0.6125 | 55,584 | 3 | 32 | 2 | 0.1 |
| KAN+GNN | 0.82 | 0.65 | 32,368 (`Proj`=`True`) | 148 | 16 | 2 | 0.001 |



### Texas
| Model | Validation accuracy | Test accuracy | Number of parameters | Best epoch | Hidden dim | Num. layers | Learning rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.972973 | **0.852459** | 54,757 | 48 | 32 | 2 | 0.01 |
| KAN | **1.0** | 0.704918 | 1,093,120 (`Proj`=`False`) | 23 | 64 | 2 | 0.01 |
| GCN | 0.918919 | 0.754098 | 55,584 | 25 | 32 | 2 | 0.0001 |
| KAN+GNN | 0.918919 | 0.737705 | 74,976 (`Proj`=`True`) | 1 | 32 | 2 | 0.1 |


## Result 2 (SHAP analysis): Rule of thumb on hyperparameter settings

For this, I fit an XGBoost model to predict the test performance of each model based on the hyperparameters. Then, I have used the SHAP values to get the 'imporatnce' of each hyperparameter. Some trends are:

### KAN+GNN
- Learning rate is the hyperparameter to tune if you want the most bang for the buck.
- The number of KAN per layers is more important than the number of message passing layers. In general, make the model as light as possible.

Figure: SHAP analysis for Cora on KAN + GNN
![alt text](/images/Cora_KANGNN_SHAP.png "Cora_KANGNN_SHAP")

### KAN
- Similar to KAN+GNN, learning rate is the most important hyperparameter.
- Also similar to KAN+GNN, make the KAN as light as possible.

Figure: SHAP analysis for Citeseer on KAN
![alt text](/images/Citeseer_KAN_SHAP.png "Citeseer_KAN_SHAP")

## Result 3: Test performance vs. Number of parameters

I have also plotted the test performance vs. the number of parameters for all cases during hyperparameter tuning. Some notes on the figure:

- I have used the log scale for the x-axis (number of parameters) to make the plot more readable.
- During tuning, there may be some cases where there may have multiple models with the same number of parameters. In such cases, I have highlighted the best performer with the most non-transparent color.

Here are some observations:

- In general, it is very easy to build a heavy model using KANs or KAN+GNNs.
- For homophilic datasets, introduce GNNs to the mix. The performance usually depends on the specific dataset.
- For heterophilic datasets, non-GNN types (MLP, KAN) usually perform better with a larger margin.


Figure: Test performance vs. Number of parameters for Cora\
![alt text](/images/Cora_Param.png "Cora_Param.png")


Figure: Test performance vs. Number of parameters for Wisconsin\
![alt text](/images/Wisconsin_Param.png "Wisconsin_Param.png")


## Note

This is an ongoing investigation, and some results may change in the future. Thanks to all the authors of the Efficient-KAN and GraphKAN repositories for their awesome work! 
