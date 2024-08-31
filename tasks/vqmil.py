import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from constants import DEVICE
from dataset import PatchDataset
from models.vq import VectorQuantizer, VQMIL
from models.aggregator import FeatureEncoder, ABMIL_Head
from utils.partition import get_split_slidenames


def vqmil_with_pseudo_bags(split:int, feature_dir, pseudo_feature_dir, split_csv, label_csv, model_save_dir, 
                           dim=512, num_embeddings=32, lr=1e-4, epoch=400, pseudo_bag_size:int=256):
    encoder = FeatureEncoder(out_dim=dim)
    head = ABMIL_Head(M=dim, L=dim)
    model = VQMIL(encoder=encoder, head=head, dim=dim, num_embeddings=num_embeddings).to(DEVICE)

    train_slides, val_slides, test_slides = get_split_slidenames(split_csv, label_csv)

    normal_pseudo_feature_dir = os.path.join(pseudo_feature_dir, 'split%d/normal.pt'%split)
    tumor_pseudo_feature_dir = os.path.join(pseudo_feature_dir, 'split%d/tumor.pt'%split)

    normal_dataset = PatchDataset(features_pt=normal_pseudo_feature_dir, label=0)
    tumor_dataset = PatchDataset(features_pt=tumor_pseudo_feature_dir, label=1)
    normal_loader = DataLoader(dataset=normal_dataset, batch_size=pseudo_bag_size, shuffle=True)
    tumor_loader = DataLoader(dataset=tumor_dataset, batch_size=pseudo_bag_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    def get_features(slidename, feature_dir):
        feature_path = os.path.join(feature_dir, slidename + '.pt')
        feature = torch.load(feature_path)
        return feature

    for e in range(epoch):

        #Additional train
        tumor_label = torch.tensor(tumor_loader.dataset.label).to(DEVICE)
        normal_label = torch.tensor(normal_loader.dataset.label).to(DEVICE)
        model.train()
        for _ in range(1):
            for normal_batch, tumor_batch in zip(normal_loader, tumor_loader):
                tumor_batch = tumor_batch.to(DEVICE)
                vq_loss, pred, encodings, A, Z = model(tumor_batch)
                cls_loss = loss_fn(pred, tumor_label)
                loss = vq_loss + cls_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                normal_batch = normal_batch.to(DEVICE)
                vq_loss, pred, encodings, A, Z = model(normal_batch)
                cls_loss = loss_fn(pred, normal_label)
                loss = vq_loss + cls_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        #Train
        model.train()
        random.shuffle(train_slides)
        for slidename, label in train_slides:
            optimizer.zero_grad()
            feature = get_features(slidename, feature_dir).to(DEVICE)
            label = torch.tensor(label).to(DEVICE)
            vq_loss, pred, encodings, A, Z = model(feature)
            cls_loss = loss_fn(pred, label) 
            loss = vq_loss + cls_loss
            loss.backward()
            optimizer.step()

        #Val
        model.eval()
        preds = []
        pred_labels = []
        labels = []
        cls_losses = []
        vq_losses = []
        for slidename, label in val_slides:
            feature = get_features(slidename, feature_dir).to(DEVICE)
            label = torch.tensor(label).to(DEVICE)
            vq_loss, pred, encodings, A, Z = model(feature)
            cls_loss = loss_fn(pred, label)
            preds.append(pred[1].item())
            pred_labels.append(torch.argmax(pred).item())
            labels.append(label.item())
            cls_losses.append(cls_loss.item())
            vq_losses.append(vq_loss.item())
        val_acc = accuracy_score(labels, pred_labels)
        val_f1 = f1_score(labels, pred_labels)
        val_auc = roc_auc_score(labels, preds)
        if e + 1 % 100 == 0:
            print('Epoch %d'%e + 1)
            print('val acc : ', val_acc)
            print('val f1 : ', val_f1)
            print('val auc : ', val_auc)
            print('\n')
    
    #save the model
    torch.save(model.state_dict(), f=model_save_dir)

    #test
    model.eval()
    preds = []
    pred_labels = []
    labels = []
    for slidename, label in test_slides:
        feature = get_features(slidename, feature_dir).to(DEVICE)
        label = torch.tensor(label).to(DEVICE)
        vq_loss, pred, encodings, A, Z = model(feature)
        cls_loss = loss_fn(pred, label)
        preds.append(pred[1].item())
        pred_labels.append(torch.argmax(pred).item())
        labels.append(label.item())
    test_acc = accuracy_score(labels, pred_labels)
    test_f1 = f1_score(labels, pred_labels)
    test_auc = roc_auc_score(labels, preds) 
    print('Split%d'%split)
    print('test acc : ', test_acc)
    print('test f1 : ', test_f1)
    print('test auc : ', test_auc)
    print('\n')








