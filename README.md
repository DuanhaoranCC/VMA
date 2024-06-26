# VMA

This repository is called CGRA in the previous version, I renamed the model name in the camera-ready version.

This repository is for the source code of the journal Neural Networks paper "Self-Supervised Contrastive Graph Representation with Node and Graph Augmentation."


## Dependencies

- torch==1.10.1+cu113
- torch_geometric==2.0.2
- scikit_learn==1.0.2

Install all dependencies using

```
pip install -r requirements.txt
```

## Usage

You can use the following command, and the parameters are given

```shell
python train.py --dataset Cora
```

The `--dataset` argument should be one of [Cora, CiteSeer, PubMed, DBLP].

## Reference link

The code refers to the following two papers. Thank them very much for their open source work.

[Deep Graph Contrastive Representation Learning(GRACE)](https://github.com/CRIPAC-DIG/GRACE)

[Directed Graph Contrastive Learning(DiGCL)](https://github.com/flyingtango/DiGCL)


## Cite
```
@article{DBLP:journals/nn/DuanXLT23,
  author       = {Haoran Duan and
                  Cheng Xie and
                  Bin Li and
                  Peng Tang},
  title        = {Self-supervised contrastive graph representation with node and graph
                  augmentation},
  journal      = {Neural Networks},
  volume       = {167},
  pages        = {223--232},
  year         = {2023},
}
```
