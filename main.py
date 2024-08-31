import os
import argparse
import random
import torch

from constants import *
from tasks.vqmil import vqmil_with_pseudo_bags

def get_args_parser():
    parser = argparse.ArgumentParser('Experiments', add_help=False)
    parser.add_argument('--task', default='vqmil', type=str)
    parser.add_argument('--dataset', default='camelyon16', type=str)
    parser.add_argument('--feature_dir', type=str)
    parser.add_argument('--pseudo_feature_dir', type=str)
    parser.add_argument('--split_dir', default='splits/task_camelyon16', type=str)
    parser.add_argument('--label_csv', default='labels/labels_all.csv', type=str)
    return parser

def main(args):

    if args.task == 'vqmil':
        if args.dataset == 'camelyon16':
            for i in range(5):
                vqmil_with_pseudo_bags(split=i, feature_dir=args.feature_dir, pseudo_feature_dir=args.pseudo_feature_dir,
                    split_csv=os.path.join(args.split_dir, 'splits_%d.csv'%i), label_csv=args.label_csv, 
                    model_save_dir='checkpoints/vqmil_s%d_n16d512lr1e-4.pth'%i,
                    dim=512, num_embeddings=32,
                    lr=1e-4, epoch=400)

    elif args.task == 'sampling':
        from utils.partition import collect_patches
        for i in range(5):
            collect_patches(feat_dir=args.feature_dir,
                            split_csv=os.path.join(args.split_dir, 'splits_%d.csv'%i), split=i, label_csv=args.label_csv)


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    args = get_args_parser()
    args = args.parse_args()
    main(args)