# Dynamic Graph embedding algorithm: DynGEM
# Final university thesis
## General information
### Author

- **Student**: Xuan-Truong Hoang
- **ID**: 1612899
- **University**: HCM city University of Science
    
<!-- 
### Problem
Graph is kind of the natural type of data in the real world, such as: social network (Facebook, Twitter,..,), traffic network,
citation science paper, biology structure,... Analyzing graph to extract viral features, get insight information of the original network to necessary and important task.
**Graph embedding** is a method to facilitate the analyzing graph task more effective. 
Due to graph will be envolve
-->
## Code
This project is implement the code of Structural Deep Network Embedding (SDNE) and Deep Embedding Method for Dynamic Graphs (DynGEM).
Beside, evaluating the performance among three graph embedding algorithms are Node2Vec, SDNE and DynGEM.
I focus on the implement of DynGEM.

## Installation
Go to the folder and use `pip` to install `requirement.txt` file.

Run: `pip install -r requirement.txt`

## Basic usage
There are two task to evaluation: stability and link prediction.
To run stability:
1. Open `statbility_configuration.ini`
2. Change `dataset_name` is your folder dataset in *data* folder. Dataset should be in a folder and contain a list of `graph_0[index].edgelist` files. For example: `graph_00.edgelist`, `graph_01.edgelist`, `graph_23.edgelist`,... These files will be read by `networkx` library of Python.
3. There are 3 algorithms in second group: `is_dyge`, `is_node2vec` and `is_sdne`. Set any algorithm is `True` to run that algorithm.
4. There are 2 kinds of files are:
    - **train_[stability|link_pred].py**: for training model to getting weights, models.
    - **eval_[stability|link_pred].py**: for evaluating model with metric stability or link prediction (ROC AUC Score| Precision@K | mAP)
5. To run a file: `python train_link_pred.py`

### Example
**Note**:Make sure you have your dataset in *data* folder with your dataset have a list `graph_0*.edgelist` files. Remember change `dataset_name` in configuration file same name your graph dataset.

Run: `python train_link_pred.py` or `python eval_link_pred.py`

I had two dynamic graph dataset in examples folder. Run `get_soc_wiki_elec_data.py` or `get_cit_hepth_data.py` to download dataset.

### Option
### Output
You will get a folder **saved_data** which contains all your trained data: 
- Processed data for link prediction; 
- DynGEM, SDNE models; 
- DynGEM, SDNE, Node2Vec embedding.

If you run evaluation python file, you get the result of metric on your graph dataset with stability and link prediction.

To understand more about algorithms, checking references in the end of page.

## Model

# References
> @article{goyal2018dyngem,
  title={Dyngem: Deep embedding method for dynamic graphs},
  author={Goyal, Palash and Kamra, Nitin and He, Xinran and Liu, Yan},
  journal={arXiv preprint arXiv:1805.11273},
  year={2018}
}

> @inproceedings{wang2016structural,
  title={Structural deep network embedding},
  author={Wang, Daixin and Cui, Peng and Zhu, Wenwu},
  booktitle={Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={1225--1234},
  year={2016}
}

> @inproceedings{grover2016node2vec,
  title={node2vec: Scalable feature learning for networks},
  author={Grover, Aditya and Leskovec, Jure},
  booktitle={Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={855--864},
  year={2016}
}



