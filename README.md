# Graph Convolutional Matrix Completion with GAN

PyTorch based implementation of Graph Convolutional Matrix Completion for recommender systems, based on [Kipf and Welling](https://arxiv.org/abs/1706.02263) (2017) paper. We also implemented them based on their [source code](https://github.com/riannevdberg/gc-mc).

This code only covers the Movielens 100K, 1M Dataset.  
After downloading [ml_1m](https://grouplens.org/datasets/movielens/) to the ```./data``` directory, you need to preprocess it by ```Preprocess.ipynb```.

## Requirements


  * Python 3.5
  * PyTorch (0.4)

## Usage


```bash
python main.py
```
## Performace


|               | Kipf and Welling  | Ours  |
| ------------- |:-------------:| -----:|
| ML-100K       | 0.910         | 0.935 |
| ML-100K (+Feat)| 0.905        | 0.932 |
| ML-1M         | 0.832         |    -  |
