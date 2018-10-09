# Graph Convolutional Matrix Completion

PyTorch based implementation of Graph Convolutional Matrix Completion for recommender systems, based on [Kipf and Welling](https://arxiv.org/abs/1706.02263) (2017) paper. We also implemented them based on their [source code](https://github.com/riannevdberg/gc-mc).

This code only covers the Movielens 1M Dataset https://grouplens.org/datasets/movielens/.

Preprocessing by ```Preprocess.ipynb``` is necessary.

## Requirements


  * Python 3.5
  * PyTorch (0.4)


## Usage

To reproduce the experiments mentioned in the paper you can run the following commands:

**Movielens 1M**
```bash
python train.py
```

Work In Progress.

## TODO
* Negative Sampling
* Hyper-parameters Settings
