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
DOUBLE_ENTITY = False
DOUBLE_RELATIONS = False
ADV = True

config_args = {
    'model': ['TransE'],
    'data_path': ['/dfs/scratch0/chami/KnowledgeGraphEmbedding/data/wn18rr'],
    'batch_size': [512],
    'negative_sample_size': [1024],
    'hidden_dim': [500, 1000],
    'gamma': [6.0],
    'adversarial_temperature': [0.5],
    'learning_rate': [0.00005],
    'max_steps': [80000],
    'test_batch_size': [8],
    'dropout': [0, 0.25, 0.5],
    #'regularization': [],
    'cpu_num': [10],
    # 'save_path': [None],
    # 'warm_up_steps': [None],
    'save_checkpoint_steps': [10000],
    'valid_steps': [10000],
    'log_steps': [100],
    'test_log_steps': [1000],
    # early stop, L2 reg, lr_decay
}

def launch():
    all_params = product(*config_args.values())
    for i, params in enumerate(all_params):
        params = dict(zip(config_args.keys(), params))
        args = ' '.join(["--{} {} ".format(x, str(params[x])) for x, _ in params.items()])
        train_command = 'CUDA_VISIBLE_DEVICES={} python -u codes/run.py --do_train --do_valid --do_test '.format(GPU) + args
        if DOUBLE_ENTITY:
            train_command += ' -de'
        if DOUBLE_RELATIONS:
            train_command += ' -dr'
        if USE_CUDA:
            train_command += ' --cuda'
        if ADV:
            train_command += ' -adv'  # TODO: check + adversarial temperature
        print(train_command)
        subprocess.call(train_command, shell=True)
    return True
    
if __name__ == '__main__':
    launch()
