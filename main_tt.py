from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils_tt import *
from utils.core_utils_tt import train   
from datasets.dataset_generic_tt import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    #all_val_auc = []
    all_train_auc = []
    all_test_acc = []
    #all_val_acc = []
    all_train_acc = []
    folds = np.arange(start, end)

    # 分割保存的地址
    basic_src = "/media/zw/Elements1/lvyp/splits/Recurrence_metastasis_vs_normal_70_balance/"

    for i in folds:
        seed_torch(args.seed)
        # 设置使用h5文件加载患者图像特征
        dataset.load_from_h5(True)
        # 根据create_splits.py分割好的数据集 进行数据集划分
        # train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        train_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                         csv_path=f"/media/zw/Elements1/lvyp/splits/Recurrence_metastasis_vs_normal_70_balance/splits_{i}.csv")
        print("-----------------dataset.use_h5", dataset.use_h5)
        print('training: {}, testing: {}'.format(len(train_dataset), len(test_dataset)))
        datasets = (train_dataset, test_dataset)
        results, test_auc, test_acc,  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        #all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        #all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
         'test_acc': all_test_acc })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=300,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)') # 权重衰减
parser.add_argument('--seed', type=int, default=5, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='sgd')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='big', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['Recurrence_metastasis_vs_normal'])

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

print('\nLoad Dataset')

if args.task == 'Recurrence_metastasis_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/media/zw/Elements1/lvyp/dataset_csv/dataset_balance.csv',
                                  data_dir="/media/zw/Elements1/lvyp/data_features/FEATURES_DIRECTORY/", 
                                  # data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                                  shuffle = False, 
                                  seed = args.seed, 
                                  print_info = True,
                                  label_dict = {'0':0, '1':1},
                                  patient_strat=False
                                  )

else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


