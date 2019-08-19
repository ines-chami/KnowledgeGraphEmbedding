"""
Run graph embedding experiments with varying parameters.
Retrieves best model based on validation score.
Retrieves average scores over multiple runs.
"""
import argparse
import datetime
import os
import subprocess
from itertools import product

import numpy as np
import scipy.stats as stats

USE_CUDA = True
GPU = 0
ADV = True

config_args = {
    # 'model': ['ReflectionE'],
    # 'model': ['RotationE'],
    'model': ['One2ManyTransE'],
    # 'data_path': ['/dfs/scratch0/advaw/OpenKE/KnowledgeGraphEmbedding/data/wn18rr'],
    'data_path': ['/dfs/scratch0/advaw/OpenKE/KnowledgeGraphEmbedding/data/FB15k-237'],
    'batch_size': [512],
    'negative_sample_size': [1024],
    'hidden_dim': [500, 1000],
    'gamma': [6.0],
    'adversarial_temperature': [0.5],
    'learning_rate': [0.000025, 0.00005],
    'max_steps': [80000],
    'test_batch_size': [8],
    'dropout': [0, 0.25, 0.5],
    'p_norm': [1],
    'cpu_num': [10],
    'save_checkpoint_steps': [10000],
    'valid_steps': [10000],
    'log_steps': [100],
    'test_log_steps': [1000],
    'entity_embedding_multiple': [1],
    'relation_embedding_multiple': [2],
    'regularization': [0.000005],
    'nsib': [1, 10, 25],
    'rho': [1, 10]
    # TODO: early stop, lr_decay
}


def launch():
    all_params = product(*config_args.values())
    k=0
    for i, params in enumerate(all_params):
        #if k>0:
            #return True
        params = dict(zip(config_args.keys(), params))
        args = ' '.join(["--{} {} ".format(x, str(params[x])) for x, _ in params.items()])
        train_command = 'CUDA_VISIBLE_DEVICES={} python -u run.py --do_train --do_valid --do_test '.format(
            GPU) + args
        if USE_CUDA:
            train_command += ' --cuda'
        if ADV:
            train_command += ' -adv'
        print(train_command)
        subprocess.call(train_command, shell=True)
        #k+=1
    return True


if __name__ == '__main__':
    launch()
