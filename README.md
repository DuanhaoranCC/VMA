# CGRA

The PyTorch implementation of Self-supervised Contrastive Graph Representation with Node and Graph Augmentation.

In this work, we propose a new graph augmentation method to generate an augmentation graph without changing any structures from the original graph. Meanwhile, a node augmentation method is proposed to augment the positive node pairs by discovering the most similar nodes in the same graph. 


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
python train.py --dataset DBLP
```

The `--dataset` argument should be one of [Cora, CiteSeer, PubMed, DBLP].

## Reference link

The code refers to the following two papers. Thank them very much for their open source work.

[Deep Graph Contrastive Representation Learning(GRACE)](https://github.com/CRIPAC-DIG/GRACE)

[Directed Graph Contrastive Learning(DiGCL)](https://github.com/flyingtango/DiGCL)

