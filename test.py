import torch
import random
import argparse
import math

import matplotlib.pyplot as plt

from tasks.metrics import draw_aggfeat_tsne, draw_feat_tsne, check_codeword_tumor_ratio, aggregated_feature_distribution, get_slide
from tasks.vqmil import vqmil_with_pseudo_bags_test, abmil_with_pseudo_bags_test, abmil_test
from tasks.visualization import visualize_patches
from models.abmil import ABMIL_withA
from models.clam import CLAM_SB
from models.dtfd_mil import Classifier_1fc, Attention_Gated, DimReduction, Attention_with_Classifier, DTFD_MIL

#kmeans_clustering()

random.seed(42)
torch.manual_seed(42)

#get_slide(ckpt_ab='checkpoints/abmil_s3_d512lr1e-4.pth', ckpt_vq='checkpoints/vqmil_400/vqmil_s3_p_n32d512lr1e-4.pth', split_csv='splits/task_camelyon16/splits_3.csv')


#aggregated_feature_distribution(ckpt_ab='checkpoints/abmil_s3_d512lr1e-4.pth', ckpt_vq='checkpoints/vqmil_400/vqmil_s3_p_n32d512lr1e-4.pth', split_csv='splits/task_camelyon16/splits_3.csv')

#draw_aggfeat_tsne(feat_ab='experiment_results/data/aggregated_feat_ab_test.pth', feat_vq='experiment_results/data/aggregated_feat_vq_test.pth')

for slide in {'test_099', 'test_048', 'tumor_060', 'test_097', 'test_102', 'test_011', 'test_052', 'test_004', 'test_033', 'test_066', 'test_116', 'tumor_027', 'test_038', 'test_046', 'test_051', 'test_079', 'test_030', 'test_117', 'test_013', 'test_110', 'test_084'}:
    try:
        draw_feat_tsne(slide=slide, ckpt_vq='checkpoints/vqmil_400/vqmil_s3_p_n32d512lr1e-4.pth')
    except:continue



#counts = check_codeword_tumor_ratio(ckpt_vq='checkpoints/vqmil_400/vqmil_s3_p_n32d512lr1e-4.pth', 
                           #split_csv='splits/task_camelyon16/splits_3.csv',
                           #)



'''
counts = [[59, 1453], [12746, 3222], [3365, 16208], [3327, 2887], [0, 0], [9961, 3431], [4086, 36767], [1882, 335], [0, 0], [41, 330], [5790, 1070], [146, 11282], [2, 2], [5091, 110120], [4006, 209668], [22379, 1606], [0, 0], [0, 0], [843, 63018], [9722, 14244], [49, 2698], [836, 51290], [4078, 243777], [142, 2124], [0, 0], [0, 0], [4913, 1470], [80, 6276], [0, 0], [0, 0], [0, 0], [0, 0]]

from models.aggregator import FeatureEncoder, ABMIL_Head
from models.vq import VQMIL
import numpy as np
import torch.nn as nn
ckpt = 'checkpoints/vqmil_400/vqmil_s3_p_n32d512lr1e-4.pth'
dim = 512
encoder = FeatureEncoder(out_dim=dim)
head = ABMIL_Head(M=dim, L=dim)
model = VQMIL(encoder=encoder, head=head, dim=dim, num_embeddings=32)
state_dict = torch.load(ckpt)
model.load_state_dict(state_dict, strict=True)

encoder = model.encoder
vqer = model.vqer
model = nn.Sequential(encoder, vqer)

codewords = vqer._embedding.weight.data

p, A, _ = head(codewords)
A = torch.log(A.detach().cpu())
A = A - torch.min(A)
A = list((A / torch.max(A)).squeeze())
A = [a.item() for a in A]

min_color = np.array([0, 1, 0])   # Cyan
max_color = np.array([1, 0, 0]) 
colors = [min_color + (max_color - min_color) * a for a in A]
colors = [(c[0], c[1], c[2]) for c in colors]

new_counts = []
for i in range(len(counts)):
    tc, nc = counts[i]
    a = A[i]
    c = colors[i]
    if tc + nc != 0:
        ratio = tc / (tc + nc)
        size = tc + nc
        new_counts.append((ratio, size, a, c))

new_counts = sorted(new_counts, key=lambda x:x[0])

ys = []
xs = []
cs = []



for i in range(len(new_counts)):
    ratio, size, a, c = new_counts[i]
    ys.append(ratio)
    xs.append(a)
    cs.append(c)
    #sizes.append(math.log(size, 2))
    #sizes.append(size)

plt.scatter(xs, ys, c=cs, s=40)
plt.xlabel('attention scores')
plt.ylabel('ratio of tumor patches')
plt.savefig('experiment_results/figures/codewords.png')



'''





















'''
vqmil_with_pseudo_bags_test(split=4, feature_dir='data_feat/res50_imgn/Camelyon16_patch224_ostu_train/pt_files',
                       split_csv='splits/task_camelyon16/splits_4.csv', 
                       label_csv='labels/labels_all.csv', 
                       model_save_dir='checkpoints/test/split4_no_commitment.pth', 
                       dim=512, epoch=301, commitment_cost=4)
'''

'''
vqmil_with_pseudo_bags_test(split=4, feature_dir='data_feat/res50_imgn/Camelyon16_patch224_ostu_train/pt_files',
                       split_csv='splits/task_camelyon16/splits_4.csv', 
                       label_csv='labels/labels_all.csv', 
                       model_save_dir='checkpoints/test/split4_n256.pth', 
                       dim=512, epoch=601, num_embeddings=256,
                       commitment_cost=0.25)
'''


'''
vqmil_with_pseudo_bags_test(split=4, feature_dir='data_feat/res50_imgn/Camelyon16_patch224_ostu_train/pt_files',
                       split_csv='splits/task_camelyon16/splits_4.csv', 
                       label_csv='labels/labels_all.csv', 
                       model_save_dir='checkpoints/test/split4_n256.pth', 
                       dim=512, epoch=601, num_embeddings=256,
                       commitment_cost=0.25, pseudo_bag_size=128)
'''

'''
vqmil_with_pseudo_bags_test_V(split=4, feature_dir='data_feat/res50_imgn/Camelyon16_patch224_ostu_train/pt_files',
                              split_csv='splits/task_camelyon16/splits_4.csv', 
                              label_csv='labels/labels_all.csv', 
                              model_save_dir='checkpoints/test/split4_n256_V.pth',
                              dim=512, epoch=601, num_embeddings=256, 
                              commitment_cost=0.25)
'''

